import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess as sb
import torch
import os, sys, itertools
from astropy.io import fits
import casacore.tables as ctab

# append script path
sys.path.append(os.path.relpath("../calibration"))

from calibration_tools import *
import generate_data
from generate_data import simulate_data

# (try to) use a GPU for computation?
use_cuda = True
if use_cuda and torch.cuda.is_available():
    mydevice = torch.device("cuda")
else:
    mydevice = torch.device("cpu")

# action direction: range [0,1], select source if >0.5
LOW = 0.0
HIGH = 1.0

# scaling of input data to prevent saturation
META_SCALE = 1e-3
MAX_DATA = 1e18
MAX_AIC = 1e6

EPS = 0.01  # to make 1/(x+EPS) when x->0 not explode
NCOMP_SCALE = 0.01  # scale source degrees of freedom


class DemixingFullEnv(gym.Env):
    """Custom Environment that follows gym interface
    DemixingEnv: only selects the directions
    DemixingFullEnv: selects the directions and also for each direction, source model"""

    metadata = {"render.modes": ["human"]}

    # K=number of directions (target+outliers)
    # state(observation): sep,az,el(all K),log(freq),stations
    # residual map: use to calculate reward
    # reward: decrease in residual/number of directions demixed
    # Npix=residual map dimension Npix x Npix
    def __init__(self, K=2, Nf=3, Npix=1024, Tdelta=10, provide_hint=False):
        super(DemixingFullEnv, self).__init__()
        # Define action and observation space
        # action: probabilities of being selected: K-1
        # for each direction, which source to select: K-1,
        # fractional noise reduction: 1
        # total space dim=(K-1)+(K-1)+ 1
        # possible source models (clusted id, source counts, name, ra, dec):
        self.source_models = (
            (
                2,
                24,
                "CasA_Gaussian",
                6.124365,
                1.026384,
            ),
            (
                3,
                3500,
                "CasA_highres",
                6.122555,
                1.026502,
            ),
            (
                4,
                9,
                "CasA_lowres",
                6.123273,
                1.026748,
            ),
            (
                5,
                12,
                "CygA_Gaussian",
                5.233856,
                0.710903,
            ),
            (
                6,
                2082,
                "CygA_highres",
                5.233955,
                0.710790,
            ),
            (
                7,
                5,
                "CygA_lowres",
                5.233838,
                0.710912,
            ),
            (
                8,
                697,
                "CygA_midres",
                5.233814,
                0.710917,
            ),
            (
                9,
                5000,
                "HerA_highres",
                4.412175,
                0.087064,
            ),
            (
                10,
                3000,
                "HerA_lowres",
                4.412048,
                0.087195,
            ),
            (
                11,
                58875,
                "HerA_new",
                4.413080,
                0.085669,
            ),
            (
                12,
                76,
                "TauA_Gaussian",
                1.459453,
                0.384229,
            ),
            (
                13,
                8781,
                "TauA_highres",
                1.459570,
                0.383962,
            ),
            (
                14,
                26,
                "TauA_lowres",
                1.459697,
                0.383912,
            ),
            (
                15,
                241,
                "VirA_Gaussian",
                3.276048,
                0.216260,
            ),
            (
                16,
                50000,
                "VirA_highres",
                3.276019,
                0.216299,
            ),
            (
                17,
                24,
                "VirA_lowres",
                3.274993,
                0.215277,
            ),
            (
                18,
                19011,
                "VirA_midres",
                3.276019,
                0.216299,
            ),
        )
        # CasA: source_models[0:3] CygA: source_models[3:7]
        # HerA: source_models[7:10] TauA: source_models[10:13]
        # VirA: source_models[13:17]
        self.source_idx = (
            np.arange(0, 3),
            np.arange(3, 7),
            np.arange(7, 10),
            np.arange(10, 13),
            np.arange(13, 17),
        )
        self.K = K
        self.Npix = Npix
        # actions: 0,1,..,K-1 : action ((K-1)+(K-1)+1)x1 each in [-1,1]
        self.action_space = spaces.Box(
            low=np.ones((2 * self.K - 1, 1)) * (-1),
            high=np.ones((2 * self.K - 1, 1)) * (1),
            dtype=np.float32,
        )
        # observation (state space): residual
        # metadata: separation,azimuth,elevation (K values),frequency,n_stations
        self.observation_space = spaces.Dict(
            {
                "metadata": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3 * self.K + 2, 1),
                    dtype=np.float32,
                )
            }
        )
        # frequencies for the simulation
        self.Nf = Nf
        # time resolution
        self.Tdelta = Tdelta
        # Full sky model and clusters, including all K directions
        self.skymodel = "./sky.txt"
        self.cluster_full = "./cluster.txt"
        self.Clus = None  # dict of parsed full cluster
        # Cluster file with all source models for all outliers
        self.complete_cluster = "./complete.cluster"
        self.Clus_complete = readcluster(self.complete_cluster)
        # for each episode, cluster file (subset of full cluster)
        self.cluster = "./cluster_epi.txt"
        # selected clusters, initialized to include target only
        self.clus_id = list()
        self.clus_id.append(self.K - 1)  # make sure target is last in the list
        # ADMM rho text file, full clusters
        self.in_admm_rho = "./admm_rho.txt"
        # output ADMM rho
        self.out_admm_rho = "./admm_rho_epi.txt"
        self.rho = np.ones(self.K, dtype=np.float32)
        # cluster id
        self.rho_id = np.ones(self.K, dtype=int)
        self.elevation = None
        self.separation = None
        # extra fields for reset()
        self.freq_low = None
        self.freq_high = None
        self.ra0 = None
        self.dec0 = None

        # standard deviation of target map (raw data and residual)
        self.std_data = 0
        self.std_residual = 0
        self.metadata = np.zeros(3 * self.K + 2, dtype=np.float32)
        self.N = 1
        self.prev_clus_id = None
        self.reward0 = 0

        self.hint = None
        self.provide_hint = provide_hint
        self.tau = 100  # temperature, divide AIC by 1/tau before softmin()

    def step(self, action):
        # action : 0:K-1 values, 0,1,..,K-2 : direction probabilities (excluding target)
        # action : K-1:2(K-1): which source model to use for each K-1 direction (0 if the direction is not selected)
        # action : 2*(K-1): expected noise reduction
        assert len(action) == 2 * (self.K - 1) + 1
        done = False  # make sure to return True at some point
        self.std_residual = self.evaluate_full_hint(action)
        # metadata 0:K-1 are separations,
        # For the directions included in calibration, set separation to 0
        metadata_update = self.metadata.copy()
        observation = {"metadata": metadata_update * META_SCALE}

        # calculate reward, subtract from default step reward
        Kselected = np.sum(action[: self.K - 1] > 0.5)
        reward = self.calculate_reward_(Kselected) - self.reward0
        print(
            "STD %f %f K %d N %d reward %f"
            % (self.std_data, self.std_residual, Kselected, self.N, reward)
        )
        info = {}
        # calculate and store the hint for future use
        if self.provide_hint:
            if self.hint is None:
                self.hint = self.get_hint()
            return observation, reward, False, done, self.hint, info
        else:
            return observation, reward, False, done, info

    def reset(self):
        # run input simulations, for debugging use e.g., Tdelta=300,do_image=True
        separation, azimuth, elevation, freq_low, freq_high, ra0, dec0, N, _ = (
            simulate_data(Nf=self.Nf)
        )
        # remember stations
        self.N = N
        self.elevation = elevation
        self.separation = separation
        # freq range (MHz)
        self.freq_low = freq_low / 1e6
        self.freq_high = freq_high / 1e6
        self.ra0 = ra0
        self.dec0 = dec0
        # read full cluster
        self.Clus = readcluster(self.cluster_full)
        self.initialize_rho_()
        # select only the target
        self.clus_id = list()
        self.clus_id.append(self.K - 1)
        self.print_clusters_()
        self.output_rho_()

        self.maxiter = 20  # maximum ADMM iterations (fixed)
        # run calibration, use --oversubscribe if not enough slots are available
        sb.run(
            "mpirun -np 3 --oversubscribe "
            + generate_data.sagecal_mpi
            + " -f 'L_SB*.MS'  -A "
            + str(self.maxiter)
            + " -P 2 -s sky.txt -c "
            + self.cluster
            + " -I DATA -O MODEL_DATA -p zsol -G "
            + self.out_admm_rho
            + " -n 4 -t "
            + str(self.Tdelta)
            + " -E 1 > calibration.out",
            shell=True,
        )

        self.std_data = self.get_noise_(col="DATA")
        self.std_residual = self.get_noise_(col="MODEL_DATA")

        # caculate baseline reward for this episode, with only 1 direction calibrated
        self.reward0 = self.calculate_reward_(1)
        # concatenate metadata
        metadata = np.zeros(3 * self.K + 2, dtype=np.float32)
        metadata[: self.K] = separation
        metadata[self.K : 2 * self.K] = azimuth
        metadata[2 * self.K : 3 * self.K] = elevation
        metadata[-2] = np.log(freq_low)
        metadata[-1] = N
        self.metadata = metadata
        observation = {"metadata": metadata * META_SCALE}
        # remember current action taken
        self.prev_clus_id = self.clus_id.copy()

        self.hint = None
        return observation, {}

    # return the average std of data, by making images
    def get_image_noise_(self, col="DATA"):
        fits_std = np.zeros(self.Nf)
        for ci in range(self.Nf):
            MS = "L_SB" + str(ci) + ".MS"
            sb.run(
                generate_data.excon
                + " -m "
                + MS
                + " -p 20 -x 0 -c "
                + col
                + " -d "
                + str(self.Npix)
                + " > /dev/null",
                shell=True,
            )
            hdu = fits.open(MS + "_I.fits")
            fitsdata = np.squeeze(hdu[0].data[0])
            hdu.close()
            fits_std[ci] = fitsdata.std()
        return np.sqrt(np.mean(fits_std**2))

    # return the average std of data
    def get_noise_(self, col="DATA"):
        fits_std = np.zeros(self.Nf)
        for ci in range(self.Nf):
            MS = "L_SB" + str(ci) + ".MS"
            fits_std[ci] = self.get_noise_var_(MS, col)
        return np.sqrt(np.mean(fits_std**2))

    # extract noise info
    def get_noise_var_(self, msname, col="DATA"):
        tt = ctab.table(msname, readonly=True)
        t1 = tt.query("ANTENNA1 != ANTENNA2", columns=col + ",FLAG")
        data0 = t1.getcol(col)
        # set nans/infs to 0
        data0[~np.isfinite(data0)] = 0
        flag = t1.getcol("FLAG")
        data = data0 * (1 - flag)
        tt.close()
        # form IQUV
        sI = (data[:, :, 0] + data[:, :, 3]) * 0.5
        return sI.std()

    def print_clusters_(self, clus_id=None):
        # print only the clusters in clus_id
        ff = open(self.cluster, "w+")
        ff.write("## Cluster selection\n")
        for ci in self.Clus.keys():
            if clus_id:
                if ci in clus_id:
                    ff.write(self.Clus[ci])
            else:
                if ci in self.clus_id:
                    ff.write(self.Clus[ci])
        ff.close()

    def initialize_rho_(self):
        # initialize from text file
        ci = 0
        with open(self.in_admm_rho, "r") as fh:
            for curline in fh:
                if (not curline.startswith("#")) and len(curline) > 1:
                    curline1 = curline.split()
                    # id hybrid rho
                    self.rho[ci] = float(curline1[2])
                    self.rho_id[ci] = int(curline1[0])
                    # ignore hybrid parameter (==1)
                    ci += 1

    def output_rho_(self, clus_id=None):
        # write rho to text file
        fh = open(self.out_admm_rho, "w+")
        fh.write("## format\n")
        fh.write("## cluster_id hybrid admm_rho_spectral admm_rho_spatial\n")
        if clus_id:
            for ci in clus_id:
                fh.write(
                    str(self.rho_id[ci])
                    + " "
                    + str(1)
                    + " "
                    + str(self.rho[ci])
                    + " 1.0\n"
                )
        else:
            for ci in self.clus_id:
                fh.write(
                    str(self.rho_id[ci])
                    + " "
                    + str(1)
                    + " "
                    + str(self.rho[ci])
                    + " 1.0\n"
                )
        fh.close()

    @staticmethod
    def scalar_to_kvec(n, K=5):
        # convert integer to binary bits, return array of size K
        ll = [1 if digit == "1" else 0 for digit in bin(n)[2:]]
        a = np.zeros(K)
        a[-len(ll) :] = ll
        return a

    def get_hint(self):
        # if data are not valid (either simulation, calibration error), return dummy hint
        if (
            ~np.isfinite(self.std_data)
            or ~np.isfinite(self.std_residual)
            or (self.std_data > MAX_DATA)
            or (self.std_residual > MAX_DATA)
        ):
            print(f"Invalid data {self.std_data} residual {self.std_residual}")
            return np.zeros(2 * (self.K - 1) + 1)

        # iterate over all possible actions
        AIC = np.zeros(2 ** (self.K - 1))
        for index in range(2 ** (self.K - 1)):
            action = self.scalar_to_kvec(index, self.K - 1)
            # check if there are -ve elevation clusters
            # if so, give a high AIC
            chosen_el = itertools.compress(self.elevation[:-1], action)
            any_neg_dir = any([x < 0 for x in chosen_el])
            if any_neg_dir:
                AIC[index] = MAX_AIC
            else:
                indices = np.where(action > 0)
                clus_id = np.unique(indices[0]).tolist()
                clus_id.append(self.K - 1)
                Kselected = len(clus_id)
                self.print_clusters_(clus_id)
                self.output_rho_(clus_id)
                sb.run(
                    "mpirun -np 3 --oversubscribe "
                    + generate_data.sagecal_mpi
                    + " -f 'L_SB*.MS' -A "
                    + str(self.maxiter)
                    + " -P 2 -s sky.txt -c "
                    + self.cluster
                    + " -I DATA -O MODEL_DATA -p zsol -G "
                    + self.out_admm_rho
                    + " -n 4 -t "
                    + str(self.Tdelta)
                    + " -E 1 > calibration.out",
                    shell=True,
                )
                # calculate noise
                std_residual = self.get_noise_(col="MODEL_DATA")
                if np.isfinite(std_residual) and (std_residual <= self.std_data):
                    AIC[index] = (
                        self.N * std_residual / self.std_data
                    ) ** 2 + Kselected * self.N
                else:
                    AIC[index] = MAX_AIC
        # take softmin
        probs = np.exp(-AIC / self.tau) / np.sum(np.exp(-AIC / self.tau))
        # map 2**(K-1) to K-1 vector
        hint = np.zeros(self.K - 1)
        for ci in range(2 ** (self.K - 1)):
            hint += probs[ci] * self.scalar_to_kvec(ci, self.K - 1)
        # transform [0,1] back to [-1,1] space
        hint = (hint - (HIGH + LOW) / 2) * (2 / (HIGH - LOW))

        # manually check source 0 and 1
        if self.separation[0] < 25 and hint[0] < 0:
            hint[0] = 1
        if self.separation[1] < 25 and hint[1] < 0:
            hint[1] = 1
        for ci in range(2, self.K - 1):
            if self.separation[ci] < 10 and self.elevation[ci] > 0 and hint[ci] < 0:
                hint[ci] = 1

        # now we have the directions to select,
        # need to try out possible source models
        assert self.K - 1 == len(self.source_idx)
        # append remaining hint K-1 values for source model, 1 value for expected noise reduction (%)
        hint_full = np.zeros(2 * (self.K - 1) + 1)
        source_model = np.zeros(self.K - 1)
        # if all(hint) <0, move up largest negative (closest to zero) to positive
        if np.max(hint) < 0:
            hint += np.abs(np.max(hint)) + EPS

        hint_full[0 : self.K - 1] = hint

        for ci in range((self.K - 1)):
            # if this source is selected
            if hint[ci] > 0.0:
                residual_vals = np.zeros(len(self.source_idx[ci]))
                source_components = np.zeros(len(self.source_idx[ci]))
                max_id = len(self.source_idx[ci])
                cj = 0
                for si in self.source_idx[ci]:
                    source_components[cj] = self.source_models[si][1]
                    # create cluster file with only target and this cluster
                    clus_id = list()
                    clus_id.append(self.K - 1)
                    self.print_clusters_(clus_id)
                    # append this cluster to self.cluster file
                    with open(self.cluster, "a") as ff:
                        ff.write("#\n")
                        ff.write(self.Clus_complete[si])
                        ff.write("\n")
                    self.output_rho_(clus_id)
                    # append this cluster to self.out_admm_rho
                    with open(self.out_admm_rho, "a") as ff:
                        ff.write("#\n")
                        ff.write(str(self.source_models[si][0]) + " 1 1.0 1.0")
                        ff.write("\n")
                    # run calibration
                    sb.run(
                        "mpirun -np 3 --oversubscribe "
                        + generate_data.sagecal_mpi
                        + " -f 'L_SB*.MS' -A "
                        + str(self.maxiter)
                        + " -P 2 -s sky.txt -c "
                        + self.cluster
                        + " -I DATA -O MODEL_DATA -p zsol -G "
                        + self.out_admm_rho
                        + " -n 4 -t "
                        + str(self.Tdelta)
                        + " -E 1 > calibration.out",
                        shell=True,
                    )
                    residual_vals[cj] = self.get_noise_(col="MODEL_DATA")
                    cj += 1
                # AIC for model parameters noise reduction by log(source_components)
                # so AIC ~ normalized_residual +  log(n_components)*scale_factor
                selection_criterion = (
                    residual_vals / (self.std_data + EPS)
                ) ** 2 + NCOMP_SCALE * np.log(source_components)
                # map choice 0,1,2,.., max_cluster-1 -> [-1,1]
                print(
                    f"{ci} selecting {np.argmin(selection_criterion)} out of {max_id}"
                )
                hint_full[self.K - 1 + ci] = self.source_to_hint(np.argmin(selection_criterion), max_id)
            else:
                print(f"{ci} not selected (using source model with lowest components)")
                hint_full[self.K - 1 + ci] = 0
        # expected noise reduction (ratio)
        hint_full[-1] = min(1.0, self.evaluate_full_hint(hint_full) / self.std_data)
        return hint_full

    def hint_to_source(self, hint, max_id):
        """
        convert hint in [-1,1] to a valid source id in 0...max_id-1
        """
        source_id = min(max_id - 1, int((hint + 0.5) * max_id))
        return source_id

    def source_to_hint(self, source_id, max_id):
        """
        convert source in [0,max_id-1] to hint in [-1,1]
        """
        hint =  source_id / max_id - 0.5
        return hint

    def evaluate_full_hint(self, hint):
        """
        hint:0...K-1, > 0.5 means source is selected
        hint:K..2*K-1, source model index in [-0.5,0.5] to use for selected sources
        """
        # create cluster file with only target and this cluster
        clus_id = list()
        clus_id.append(self.K - 1)
        self.print_clusters_(clus_id)
        self.output_rho_(clus_id)
        # iterate over hint
        for ci in range((self.K - 1)):
            if hint[ci] > 0.0:
                # find how many possible source models for this cluster
                max_id = len(self.source_idx[ci])
                # get back from [-1,1] to real id in 0:max_id
                source_id = self.hint_to_source(hint[self.K - 1 + ci], max_id)

                cluster_id0 = self.source_idx[ci][
                    source_id
                ]  # ordinal (row) in cluster file
                cluster_id = self.source_models[cluster_id0][0]  # actual cluster
                print(
                    f"Including {ci} source model {source_id} row {cluster_id0} cluster {cluster_id} {self.source_models[cluster_id0][2]}"
                )
                # append this cluster to self.cluster file
                with open(self.cluster, "a") as ff:
                    ff.write("#\n")
                    ff.write(self.Clus_complete[cluster_id0])
                    ff.write("\n")
                # append this cluster to self.out_admm_rho
                with open(self.out_admm_rho, "a") as ff:
                    ff.write("#\n")
                    ff.write(str(cluster_id) + " 1 1.0 1.0")
                    ff.write("\n")
        # run calibration
        sb.run(
            "mpirun -np 3 --oversubscribe "
            + generate_data.sagecal_mpi
            + " -f 'L_SB*.MS' -A "
            + str(self.maxiter)
            + " -P 2 -s sky.txt -c "
            + self.cluster
            + " -I DATA -O MODEL_DATA -p zsol -G "
            + self.out_admm_rho
            + " -n 4 -t "
            + str(self.Tdelta)
            + " -E 1 > calibration.out",
            shell=True,
        )
        residual_std = self.get_noise_(col="MODEL_DATA")

        return residual_std

    def calculate_reward_(self, Kselected):
        # reward ~ 1/(noise (variance) reduction) /(clusters calibrated)
        # penalty ~ maxiter (negative reward)
        data_var = self.std_data * self.std_data
        noise_var = self.std_residual * self.std_residual
        # AIC = -log(likelihood) + 2 deg_of_freedom
        # log(likelihood) ~ (normalized residual_sum_sqr)x baselines,
        # deg_of_freeedom ~ n_stations x n_directions
        # so AIC ~ n_stations^2 x normalized_residual + n_stations x n_directions
        # use -AIC as reward
        reward = -self.N * self.N * noise_var / (data_var + EPS) - Kselected * self.N
        # normalize reward (mean/variance found by the initial ~3000 reward values)
        reward = (reward - (-859)) / 3559.0

        # penalty: increased iterations result in increased penalty
        penalty = -self.maxiter / 100

        return reward + penalty

    def render(self, mode="human"):
        print("%%%%%%%%%%%%%%%%%%%%%%")
        print("Render not implemented")
        print("%%%%%%%%%%%%%%%%%%%%%%")

    def close(self):
        pass

    def __del__(self):
        for ci in range(self.Nf):
            MS = "L_SB" + str(ci) + ".MS"
            try:
                sb.run("rm -rf " + MS, shell=True)
            except:
                pass

#dem=DemixingFullEnv(K=6,Nf=3,Npix=1024,Tdelta=10)
#obs,_=dem.reset()
#hint=dem.get_hint()
# print(hint)
# action=np.zeros(5+1)
# action[-1]=0 # max iterations
# action[0]=1
# obs,reward,_,_,_=dem.step(action=action)
# sb.run('mv influenceI.fits inf1.fits',shell=True)
# action[1]=1
# obs,reward,_,_,_=dem.step(action=action)
# sb.run('mv influenceI.fits inf2.fits',shell=True)
# action[2]=1
# obs,reward,_,_,_=dem.step(action=action)
# sb.run('mv influenceI.fits inf3.fits',shell=True)
# action[3]=1
# obs,reward,_,_,_=dem.step(action=action)
# sb.run('mv influenceI.fits inf4.fits',shell=True)
# action[4]=1
# obs,reward,_,_,_=dem.step(action=action)
# sb.run('mv influenceI.fits inf5.fits',shell=True)
