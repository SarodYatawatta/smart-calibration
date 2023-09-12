import gym
from gym import spaces
import numpy as np
from astropy.io import fits
import torch
import os

from calibration_tools import *
from simulate import simulate_models

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

#LOW=0.00001
#HIGH=10000.0
# for starting off, use tighter bound
LOW=0.01
HIGH=1000.

EPS=0.01 # to make 1/(x+EPS) when x->0 not explode

class CalibEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, M=5):
    super(CalibEnv, self).__init__()
    # Define action and observation space
    # K: directions, possible actions: vector 2Kx1, 
    # K: spectral regularization, K: spatial regularization
    self.K=0 # will be set during reset()
    # M= maximum number of clusters, we need K<=M always
    self.M=M
    # sky model components, averaged per-direction,
    # for input to DQN, format each line: cluster_id, l, m, sI, sP=5
    # actions: 0:K: spectral, K:2K1: spatial in [-1,1], step() will rescale later
    self.action_space = spaces.Box(low=np.ones((2*self.M,1))*(-1.0),high=np.ones((2*self.M,1))*1.0,dtype=np.float32)
    # observation (state space): rho, 2x1 real vector 0,..inf
    # rho: lower bound 0, upper bound 10k, one dimension for each direction
    self.observation_space = spaces.Dict({
       'img': spaces.Box(low=-HIGH,high=HIGH,shape=(128,128),dtype=np.float32),
       'sky': spaces.Box(low=-HIGH,high=HIGH,shape=(self.M,5),dtype=np.float32)
       })

    self.f_low=None
    self.f_high=None
    self.ra0=None
    self.dec0=None
    self.Ts=10 # must match -t option in calibration

    # shell script and command names
    self.cmd_simul_data='./dosimul.sh > simul.out'
    self.cmd_calib_data='./docal.sh > calib.out'
    self.cmd_calc_influence='./doinfluence.sh '+str(self.f_low)+' '+str(self.f_high)+' '+str(self.ra0)+' '+str(self.dec0)+' '+str(self.Ts)+'> influence.out'

    # input/output file names
    # original data
    self.in_fits_data='./orig/data.fits'
    # residual 
    self.in_fits_res='./orig/res.fits'
    # influence map FITS file name
    self.in_fits_influence='./orig/influenceI.fits'
    # sky model text file
    self.in_text_sky='./sky_cluster_lmn.txt'
    # initial ADMM rho text file, determined by sky model fluxes
    self.in_admm_rho='./admm_rho0.txt'
    # input sky/cluster model
    self.in_sky_cluster='./skylmn.txt'
    # output ADMM rho
    self.out_admm_rho='./admm_rho_new.txt'

    # regularization factors for K directions, initialized to 1 here,
    # but will be re-initialized based on the simulation
    self.rho_spectral=np.ones(self.M,dtype=np.float32)
    self.rho_spatial=np.ones(self.M,dtype=np.float32)
    self.output_rho_()
    self.sky=None # sky model

  def initialize_rho_(self):
    # initialize from text file, format
    # id, hybrid, spectral_reg, spatial_reg
    ci=0
    with open(self.in_admm_rho,'r') as fh:
        for curline in fh:
          if (not curline.startswith('#')) and len(curline)>1:
             curline1=curline.split()
             # id hybrid rho
             self.rho_spectral[ci]=float(curline1[2])
             self.rho_spatial[ci]=float(curline1[3])
             ci +=1

  def output_rho_(self):
    # write rho to text file
    fh=open(self.out_admm_rho,'w+')
    fh.write('## format\n')
    fh.write('## cluster_id hybrid admm_rho_spectral admm_rho_spatial\n')
    rho_spectral=self.rho_spectral.squeeze()
    rho_spatial=self.rho_spatial.squeeze()
    for ci in range(self.K):
      fh.write(str(ci+1)+' '+str(1)+' '+str(rho_spectral[ci])+' '+str(rho_spatial[ci])+'\n')
    fh.close()

  def step(self, action):
    done=False # make sure to return True at some point
    # check given action fits the K value
    assert(len(action)==2*self.M)
    # update state based on the action [-1,1] ->  rho = scale*(action)
    rho=(action.squeeze())*(HIGH-LOW)/2+(HIGH+LOW)/2
    self.rho_spectral[:self.K] =rho[0:self.K]
    self.rho_spatial[:self.K] =rho[self.M:self.M+self.K]
    penalty=0
    # make sure rho stays within limits, if this happens, add a penalty
    for ci in range(self.K):
      if self.rho_spectral[ci]<LOW:
         self.rho_spectral[ci]=LOW
         penalty +=-0.1
      if self.rho_spectral[ci]>HIGH:
        self.rho_spectral[ci]=HIGH
        penalty +=-0.1
      if self.rho_spatial[ci]<LOW:
         self.rho_spatial[ci]=LOW
         penalty +=-0.1
      if self.rho_spatial[ci]>HIGH:
        self.rho_spatial[ci]=HIGH
        penalty +=-0.1

    # output rho
    self.output_rho_()
    # run calibration with current rho (observation)
    os.system(self.cmd_calib_data)
    # make influence map, find reward
    self.cmd_calc_influence='./doinfluence.sh '+str(self.f_low)+' '+str(self.f_high)+' '+str(self.ra0)+' '+str(self.dec0)+' '+str(self.Ts)+'> influence.out'
    os.system(self.cmd_calc_influence)
    # info : meta details {}
    hdul = fits.open('orig/influenceI.fits')
    data=hdul[0].data[0,:,:,:]
    hdul.close()
    hdud = fits.open('orig/data.fits')
    raw_data=hdud[0].data[0,0,:,:]
    sigma0=raw_data.std()
    hdud.close()
    hdur = fits.open('orig/res.fits')
    res_data=hdur[0].data[0,0,:,:]
    hdur.close()
    sigma1=res_data.std()
    observation={
         'img': data,
         'sky': self.sky }
    # reward: residual error, normalized by data power
    # good calibration sigma0>sigma1, so reward > 1
    reward=sigma0/sigma1+1./(data.std()+EPS)+penalty
    info={}
    return observation, reward, done, info

  def reset(self):
    # run input simulations
    # generate K, minimum 2, max M
    self.K=np.random.choice(np.arange(2,self.M+1))
    assert(self.K>1)
    assert(self.K<=self.M)
    M,freq_low,freq_high,ra0,dec0,time_slots=simulate_models(K=self.K)
    self.f_low=freq_low
    self.f_high=freq_high
    self.ra0=ra0
    self.dec0=dec0
    # make sure we have enough room to store sky model (and feed it to DQN)
    assert(self.M>=M)

    # rho <- rho0
    self.initialize_rho_()
    # output rho
    self.output_rho_()
    os.system(self.cmd_simul_data) 
    # if K<M, extra locations will be 0
    self.sky=read_skycluster(self.in_sky_cluster,self.M)
    # run calibration with current rho (observation)
    os.system(self.cmd_calib_data)
    # make influence map, find reward
    self.cmd_calc_influence='./doinfluence.sh '+str(self.f_low)+' '+str(self.f_high)+' '+str(self.ra0)+' '+str(self.dec0)+' '+str(self.Ts)+'> influence.out'
    os.system(self.cmd_calc_influence)
    # info : meta details {}
    hdul = fits.open('orig/influenceI.fits')
    #hdul.info()
    data=hdul[0].data[0,:,:,:]#+np.random.randn(1,128,128)
    observation={
            'img': data,
            'sky': self.sky } 
    return observation  # reward, done, info can't be included

  def render(self, mode='human'):
    print('%%%%%%%%%%%%%%%%%%%%%%')
    print(self.rho_spectral.data)
    print(self.rho_spatial.data)
    print('%%%%%%%%%%%%%%%%%%%%%%')

  def close (self):
    pass




#env=CalibEnv(M=5) # use M>=K
#obs=env.reset()
#print(obs['img'].shape)
#print(obs['sky'].shape)
#env.step(np.random.rand(env.M*2))
