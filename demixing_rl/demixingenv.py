import gym
from gym import spaces
import numpy as np
import subprocess as sb
from astropy.io import fits
import torch
import os,sys
# append script path
sys.path.append(os.path.relpath('../calibration'))

from calibration_tools import *
import generate_data
from generate_data import simulate_data

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

# action range [0,1], select source if >0.5
LOW=0.0
HIGH=1.

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
  def __init__(self, K=2, Nf=3, Ninf=128, Npix=1024, Tdelta=10):
    super(DemixingEnv, self).__init__()
    # Define action and observation space
    # action space dim=number of outlier clusters=vector (K-1)x1, K: number of directions
    self.K=K
    self.Ninf=Ninf
    self.Npix=Npix
    # actions: 0,1,..,K-2 : action (K-1)x1 each in [0,1]
    self.action_space = spaces.Box(low=np.zeros((self.K-1,1))*LOW,high=np.ones((self.K-1,1))*HIGH,dtype=np.float32)
    # observation (state space): residual and influence maps
    # metadata: separation,azimuth,elevation,frequency,
    self.observation_space = spaces.Dict({
       'infmap': spaces.Box(low=-np.inf,high=np.inf,shape=(Ninf,Ninf),dtype=np.float32),
       'metadata': spaces.Box(low=-np.inf,high=np.inf,shape=(3*self.K+1,1),dtype=np.float32)
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
    self.rho=np.ones(self.K,dtype=np.float32)

    # shell script and command names
    self.cmd_calc_influence='./doinfluence.sh > influence.out'
    
    # standard deviation of target map (raw data and residual)
    self.std_data=0
    self.std_residual=0
    self.metadata=np.zeros(3*self.K+1)

  def step(self, action):
    done=False # make sure to return True at some point
    # update state based on the action [-1,1] ->  rho = scale*(action)
    rho =action*(HIGH-LOW)/2+(HIGH+LOW)/2
    # find indices of selected directions
    indices=np.where(rho>0.5)
    self.clus_id=indices[0].tolist()
    self.clus_id.append(self.K-1)
    Kselected=len(self.clus_id)
    # create cluster file clusters based on clus_id indices
    self.print_clusters_()
    self.output_rho_()
    # run calibration, use --oversubscribe if not enough slots are available
    sb.run('mpirun -np 3 --oversubscribe '+generate_data.sagecal_mpi+' -f \'L_SB*.MS\'  -A 30 -P 2 -s sky.txt -c '+self.cluster+' -I DATA -O MODEL_DATA -p zsol -G '+self.out_admm_rho+' -n 4 -t '+str(self.Tdelta)+' -V',shell=True)

    # calculate influence
    sb.run(self.cmd_calc_influence,shell=True)
    hdul = fits.open('./influenceI.fits')
    infdata=hdul[0].data[0,:,:,:]
    hdul.close()
    observation={
      'infmap': infdata,
      'metadata': self.metadata }

    self.std_residual=self.make_images_(col='MODEL_DATA')

    # reward ~ 1/(noise (var) reduction) /(clusters calibrated)
    data_var=self.std_data*self.std_data
    noise_var=self.std_residual*self.std_residual
    # -AIC ~ -log(residual_sum_sqr) - deg_of_freedom
    # we use normalized residual_sum_sqr, and normalized_deg_freedom
    reward=math.log(data_var/(noise_var+EPS))-Kselected
    influence_std=infdata.std()/100 # arbitray scaling, because influence is alreade scaled in calculation
    # penalize by influence 
    reward-=influence_std*influence_std
    print('STD %f %f Inf %f K %d reward %f'%(data_var,noise_var,influence_std,Kselected,reward))
    info={}
    return observation, reward, done, info

  def reset(self):
    # run input simulations
    separation,azimuth,elevation,freq=simulate_data(Nf=self.Nf)
    # read full cluster
    self.Clus=readcluster(self.cluster_full)
    self.initialize_rho_()
    # select only the target
    self.print_clusters_()
    self.output_rho_()
    # run calibration, use --oversubscribe if not enough slots are available
    sb.run('mpirun -np 3 --oversubscribe '+generate_data.sagecal_mpi+' -f \'L_SB*.MS\'  -A 30 -P 2 -s sky.txt -c '+self.cluster+' -I DATA -O MODEL_DATA -p zsol -G '+self.out_admm_rho+' -n 4 -t '+str(self.Tdelta)+' -V',shell=True)

    # calculate influence (image at ./influenceI.fits)
    sb.run(self.cmd_calc_influence,shell=True)
    self.std_data=self.make_images_(col='DATA')
    self.std_residual=self.make_images_(col='MODEL_DATA')
    # concatenate metadata
    metadata=np.zeros(3*self.K+1)
    metadata[:self.K]=separation
    metadata[self.K:2*self.K]=azimuth
    metadata[2*self.K:3*self.K]=elevation
    metadata[-1]=freq
    self.metadata=metadata
    hdul = fits.open('./influenceI.fits')
    infdata=hdul[0].data[0,:,:,:]
    hdul.close()
    observation={
      'infmap': infdata,
      'metadata': metadata }
    return observation

  # also return the std of average image
  def make_images_(self,col='DATA'):
    for ci in range(self.Nf):
      MS='L_SB'+str(ci)+'.MS'
      sb.run(generate_data.excon+' -m '+MS+' -p 20 -x 0 -c '+col+' -d '+str(self.Npix)+' > /dev/null',shell=True)
      sb.run('bash ./calmean.sh \'L_SB*.MS_I*fits\' 1 && python calmean_.py && mv bar.fits '+col+'.fits',shell=True)
    hdu=fits.open(col+'.fits')
    fitsdata=np.squeeze(hdu[0].data[0])
    hdu.close()
    return fitsdata.std()


  def print_clusters_(self):
    # print only the clusters in clus_id
    ff=open(self.cluster,'w+')
    ff.write('## Cluster selection\n')
    for ci in self.Clus.keys():
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
             ci +=1

  def output_rho_(self):
    # write rho to text file
    fh=open(self.out_admm_rho,'w+')
    fh.write('## format\n')
    fh.write('## cluster_id hybrid admm_rho\n')
    ck=1
    for ci in self.clus_id:
      fh.write(str(ck)+' '+str(1)+' '+str(self.rho[ci])+'\n')
      ck+=1
    fh.close()

  def render(self, mode='human'):
    print('%%%%%%%%%%%%%%%%%%%%%%')
    print(self.rho)
    print('%%%%%%%%%%%%%%%%%%%%%%')

  def close (self):
    pass

#dem=DemixingEnv(K=6,Nf=3,Ninf=128,Npix=1024,Tdelta=10)
#obs=dem.reset()
#sb.run('mv influenceI.fits inf0.fits',shell=True)
#sb.run('mv MODEL_DATA.fits MODEL0.fits',shell=True)
#action=np.zeros(5)
#action[0]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf1.fits',shell=True)
#sb.run('mv MODEL_DATA.fits MODEL1.fits',shell=True)
#action[1]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf2.fits',shell=True)
#sb.run('mv MODEL_DATA.fits MODEL2.fits',shell=True)
#action[2]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf3.fits',shell=True)
#sb.run('mv MODEL_DATA.fits MODEL3.fits',shell=True)
#action[3]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf4.fits',shell=True)
#sb.run('mv MODEL_DATA.fits MODEL4.fits',shell=True)
#action[4]=1
#obs,reward,_,_=dem.step(action=action)
#sb.run('mv influenceI.fits inf5.fits',shell=True)
#sb.run('mv MODEL_DATA.fits MODEL5.fits',shell=True)
