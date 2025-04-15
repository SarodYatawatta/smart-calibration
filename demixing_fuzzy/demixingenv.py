import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess as sb
import torch
import os,sys,itertools
from astropy.io import fits
import casacore.tables as ctab
# append script path
sys.path.append(os.path.relpath('../calibration'))

from calibration_tools import *
import generate_data
from generate_data import simulate_data
from demix_controller import DemixController

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

# scaling of input data to prevent saturation
INF_SCALE=1e-3
META_SCALE=1e-5

EPS=0.01 # to make 1/(x+EPS) when x->0 not explode

class DemixingEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  # K=number of directions (target+outliers)
  # state(observation): sep,az,el(all K),freq,target influence map
  # residual map: use to calculate reward, influence map: feed to network
  # reward: decrease in residual/number of directions demixed + influence map penalty
  # Ninf=influence map dimensions NinfxNinf (set in doinfluence.sh)
  # Npix=residual map dimension Npix x Npix
  def __init__(self, K=2, Nf=3, Ninf=128, Npix=1024, Tdelta=10, provide_hint=False):
    super(DemixingEnv, self).__init__()
    # Define action and observation space
    # action space dim=number of outlier clusters=vector (K-1)x1, K: number of directions + 1 for max ADMM iterations
    self.K=K
    self.Ninf=Ninf
    self.Npix=Npix
    # actions: 20 each in [-1,1]
    self.n_fuzzy=20
    self.action_space = spaces.Box(low=np.ones((self.n_fuzzy,1))*(-1),high=np.ones((self.n_fuzzy,1))*(1),dtype=np.float32)
    # observation (state space): residual and influence maps
    # metadata: separation,azimuth,elevation,log_fluxes (K values),frequency,n_stations (2 values)
    self.n_metadata=4*self.K+2
    self.observation_space = spaces.Dict({
       'infmap': spaces.Box(low=-np.inf,high=np.inf,shape=(Ninf,Ninf),dtype=np.float32),
       'metadata': spaces.Box(low=-np.inf,high=np.inf,shape=(self.n_metadata,1),dtype=np.float32)
       })
    # frequencies for the simulation
    self.Nf=Nf
    # time resolution
    self.Tdelta=Tdelta
    # Full sky model and clusters, including all K directions
    self.skymodel='./sky.txt'
    self.cluster_full='./cluster.txt'
    self.Clus=None # dict of parsed full cluster
    # for each episode, cluster file (subset of full cluster)
    self.cluster='./cluster_epi.txt'
    # selected clusters, initialized to include target only
    self.clus_id=list()
    self.clus_id.append(self.K-1) # make sure target is last in the list
    # ADMM rho text file, full clusters
    self.in_admm_rho='./admm_rho.txt'
    # output ADMM rho
    self.out_admm_rho='./admm_rho_epi.txt'
    # max ADMM iterations
    self.maxiter=15
    self.rho=np.ones(self.K,dtype=np.float32)
    # cluster id
    self.rho_id=np.ones(self.K,dtype=int)
    self.elevation=None
    self.separation=None
    self.log_fluxes=None
    self.target_flux=0
    # extra fields for reset()
    self.freq_low=None
    self.freq_high=None
    self.ra0=None
    self.dec0=None
    # shell script and command names
    self.cmd_calc_influence=None
    
    # standard deviation of target map (raw data and residual)
    self.std_data=0
    self.std_residual=0
    self.metadata=np.zeros(self.n_metadata,dtype=np.float32)
    self.N=1
    self.prev_clus_id=None
    self.reward0=0

    self.hint=None
    self.provide_hint=provide_hint
    self.tau=100 # temperature, divide AIC by 1/tau before softmin()

    # fuzzy controller (initialized at each reset())
    self.ctrl=DemixController(n_action=self.n_fuzzy)

  def step(self, action):
    # action: fuzzy set boundaries, prioriy cutoff for being selected
    # action : self.n_fuzzy x 1: in [-1,1] 
    # map from [-1,1] to [0,1]
    action_scaled=action*0.5+0.5
    done=False # make sure to return True at some point
    # update fuzzy controller based on action
    self.ctrl.update_limits(action_scaled)
    self.ctrl.create_controller()
    # evaluate sky using controller
    flux_ratio=np.exp(self.log_fluxes)/self.target_flux
    priority=self.ctrl.evaluate(self.elevation,self.separation,self.log_fluxes,flux_ratio)
    priority_cutoff=self.ctrl.get_high_priority()
    # find indices of selected directions
    indices=np.where(priority >= priority_cutoff)
    if len(indices) > 0:
      self.clus_id=np.unique(indices[0]).tolist()
    else:
      self.clus_id=list()
    self.clus_id.append(self.K-1)
    # check if current cluster selection (action) is same as previous
    if self.prev_clus_id==self.clus_id:
        #done=True
        pass
    else:
        self.prev_clus_id=self.clus_id.copy()
    Kselected=len(self.clus_id)
    # create cluster file clusters based on clus_id indices
    self.print_clusters_()
    self.output_rho_()
    # run calibration, use --oversubscribe if not enough slots are available
    sb.run('mpirun -np 3 --oversubscribe '+generate_data.sagecal_mpi+' -f \'L_SB*.MS\'  -A '+str(self.maxiter)+' -P 2 -s sky.txt -c '+self.cluster+' -I DATA -O MODEL_DATA -p zsol -G '+self.out_admm_rho+' -n 4 -t '+str(self.Tdelta)+' > calibration.out',shell=True)

    # calculate influence (update the command)
    sb.run(self.cmd_calc_influence,shell=True)
    hdul = fits.open('./influenceI.fits')
    infdata=hdul[0].data[0,:,:,:]
    infdata=infdata.astype(np.float32)
    hdul.close()
    # metadata 0:K-1 are separations,
    # For the directions included in calibration, set separation to 0
    metadata_update=self.metadata.copy()
    metadata_update[self.clus_id]=0
    observation={
      'infmap': infdata*INF_SCALE,
      'metadata': metadata_update*META_SCALE }

    self.std_residual=self.get_noise_(col='MODEL_DATA')

    # calculate reward, subtract from default step reward
    reward=self.calculate_reward_(Kselected)-self.reward0
    influence_std=infdata.std()/100 # arbitray scaling, because influence is alreade scaled in calculation
    print('STD %f %f Inf %f K %d N %d reward %f'%(self.std_data,self.std_residual,influence_std,Kselected,self.N,reward))
    info={}
    # calculate and store the hint for future use
    if self.provide_hint:
        if self.hint is None:
          self.hint=self.get_hint()
        return observation, reward, done, self.hint, info
    else:
        return observation, reward, done, info

  def reset(self):
    # run input simulations, for debugging use e.g., Tdelta=300,do_image=True
    separation,azimuth,elevation,freq_low,freq_high,ra0,dec0,N,fluxes=simulate_data(Nf=self.Nf)
    # remember stations
    self.N=N
    self.elevation=elevation
    self.separation=separation
    self.log_fluxes=np.log(fluxes)
    self.target_flux=fluxes[-1]
    # freq range (MHz)
    self.freq_low=freq_low/1e6
    self.freq_high=freq_high/1e6
    self.ra0=ra0
    self.dec0=dec0
    # read full cluster
    self.Clus=readcluster(self.cluster_full)
    self.initialize_rho_()
    # select only the target
    self.clus_id=list()
    self.clus_id.append(self.K-1)
    self.print_clusters_()
    self.output_rho_()

    self.maxiter=15 # provide a reasonable value
    # run calibration, use --oversubscribe if not enough slots are available
    sb.run('mpirun -np 3 --oversubscribe '+generate_data.sagecal_mpi+' -f \'L_SB*.MS\'  -A '+str(self.maxiter)+' -P 2 -s sky.txt -c '+self.cluster+' -I DATA -O MODEL_DATA -p zsol -G '+self.out_admm_rho+' -n 4 -t '+str(self.Tdelta)+' > calibration.out',shell=True)

    # calculate influence (image at ./influenceI.fits)
    self.cmd_calc_influence='./doinfluence.sh '+str(self.freq_low)+' '+str(self.freq_high)+' '+str(self.ra0)+' '+str(self.dec0)+' '+str(self.Tdelta)+' > influence.out'
    sb.run(self.cmd_calc_influence,shell=True)
    self.std_data=self.get_noise_(col='DATA')
    self.std_residual=self.get_noise_(col='MODEL_DATA')

    # caculate baseline reward for this episode, with only 1 direction calibrated
    self.reward0=self.calculate_reward_(1)
    # concatenate metadata
    metadata=np.zeros(self.n_metadata,dtype=np.float32)
    metadata[:self.K]=separation
    metadata[self.K:2*self.K]=azimuth
    metadata[2*self.K:3*self.K]=elevation
    metadata[3*self.K:4*self.K]=self.log_fluxes
    metadata[-2]=freq_low
    metadata[-1]=N
    self.metadata=metadata
    hdul = fits.open('./influenceI.fits')
    infdata=hdul[0].data[0,:,:,:]
    infdata=infdata.astype(np.float32)
    hdul.close()
    observation={
      'infmap': infdata*INF_SCALE,
      'metadata': metadata*META_SCALE }
    # remember current action taken
    self.prev_clus_id=self.clus_id.copy()

    # create default fuzzy controller (to get the hint)
    self.ctrl.create_controller()
    self.hint=self.get_hint()

    return observation

  # return the average std of data, by making images
  def get_image_noise_(self,col='DATA'):
    fits_std=np.zeros(self.Nf)
    for ci in range(self.Nf):
      MS='L_SB'+str(ci)+'.MS'
      sb.run(generate_data.excon+' -m '+MS+' -p 20 -x 0 -c '+col+' -d '+str(self.Npix)+' > /dev/null',shell=True)
      hdu=fits.open(MS+'_I.fits')
      fitsdata=np.squeeze(hdu[0].data[0])
      hdu.close()
      fits_std[ci]=fitsdata.std()
    return np.sqrt(np.mean(fits_std**2))

  # return the average std of data
  def get_noise_(self,col='DATA'):
    fits_std=np.zeros(self.Nf)
    for ci in range(self.Nf):
      MS='L_SB'+str(ci)+'.MS'
      fits_std[ci]=self.get_noise_var_(MS,col)
    return np.sqrt(np.mean(fits_std**2))

  # extract noise info
  def get_noise_var_(self,msname,col='DATA'):
        tt=ctab.table(msname,readonly=True)
        t1=tt.query('ANTENNA1 != ANTENNA2',columns=col+',FLAG')
        data0=t1.getcol(col)
        flag=t1.getcol('FLAG')
        data=data0*(1-flag)
        tt.close()
        # set nans to 0
        data[np.isnan(data)]=0.
        # form IQUV
        sI=(data[:,:,0]+data[:,:,3])*0.5
        return sI.std()

  def print_clusters_(self,clus_id=None):
    # print only the clusters in clus_id
    ff=open(self.cluster,'w+')
    ff.write('## Cluster selection\n')
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
    ci=0
    with open(self.in_admm_rho,'r') as fh:
        for curline in fh:
          if (not curline.startswith('#')) and len(curline)>1:
             curline1=curline.split()
             # id hybrid rho
             self.rho[ci]=float(curline1[2])
             self.rho_id[ci]=int(curline1[0])
             # ignore hybrid parameter (==1)
             ci +=1

  def output_rho_(self,clus_id=None):
    # write rho to text file
    fh=open(self.out_admm_rho,'w+')
    fh.write('## format\n')
    fh.write('## cluster_id hybrid admm_rho_spectral admm_rho_spatial\n')
    if clus_id:
      for ci in clus_id:
        fh.write(str(self.rho_id[ci])+' '+str(1)+' '+str(self.rho[ci])+' 1.0\n')
    else:
      for ci in self.clus_id:
        fh.write(str(self.rho_id[ci])+' '+str(1)+' '+str(self.rho[ci])+' 1.0\n')
    fh.close()

  def get_hint(self):
    # Hint is just the default fuzzy config converted to an action
    hint_full=self.ctrl.update_action()
    # map from [0,1] to [-1,1]
    return 2.0*(hint_full-0.5)

  def calculate_reward_(self,Kselected):
    # reward ~ 1/(noise (variance) reduction) /(clusters calibrated)
    # penalty ~ maxiter (negative reward)
    data_var=self.std_data*self.std_data
    noise_var=self.std_residual*self.std_residual
    # AIC = -log(likelihood) + 2 deg_of_freedom
    # log(likelihood) ~ (normalized residual_sum_sqr)x baselines,
    # deg_of_freeedom ~ n_stations x n_directions
    # so AIC ~ n_stations^2 x normalized_residual + n_stations x n_directions
    # use -AIC as reward
    reward=-self.N*self.N*noise_var/(data_var+EPS)-Kselected*self.N
    # normalize reward (mean/variance found by the initial ~3000 reward values)
    reward=(reward-(-859))/3559.0

    # penalty: from fuzzy controller
    penalty=0

    return reward+penalty

  def render(self, mode='human'):
    print('%%%%%%%%%%%%%%%%%%%%%%')
    print('Render not implemented')
    print('%%%%%%%%%%%%%%%%%%%%%%')

  def close (self):
    pass

  def __del__(self):
    for ci in range(self.Nf):
       MS='L_SB'+str(ci)+'.MS'
       sb.run('rm -rf '+MS,shell=True)

#dem=DemixingEnv(K=6,Nf=3,Ninf=128,Npix=1024,Tdelta=10)
#obs=dem.reset()
#hint=dem.hint
#sb.run('mv influenceI.fits inf0.fits',shell=True)
#action=hint#2*(np.random.rand(20)-0.5)
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf1.fits',shell=True)
#action=2*(np.random.rand(20)-0.5)
#action[-1]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf2.fits',shell=True)
#action[2]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf3.fits',shell=True)
#action[3]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf4.fits',shell=True)
#action[4]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf5.fits',shell=True)
