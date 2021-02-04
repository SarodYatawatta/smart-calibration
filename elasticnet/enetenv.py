import gym
from gym import spaces
import numpy as np
from lbfgsnew import LBFGSNew
from autograd_tools import *
import time

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

#LOW=1e-9
#HIGH=1e9
# for starting off, use tighter bound
LOW=0.001
HIGH=0.100
class ENetEnv(gym.Env):
  """Elastic Net Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  # solve 
  # arg min_x || y - Ax ||^2 + \lambda1 (||x||_2)^2 + \lambda2 ||x||_1
  # A: tall matrix NxM, N>M
  def __init__(self, M=5, N=15):
    super(ENetEnv, self).__init__()
    # N=rows=datapoints, M=columns=parameters
    # Define action and observation space
    # possible actions: vector Kx1, K=2 to match rho
    self.K=2
    self.N=N
    self.M=M
    # actions: 0,1,..,K-1 : action K : scaled/shifted from actor output
    self.action_space = spaces.Box(low=np.zeros((self.K,1))*LOW,high=np.ones((self.K,1))*HIGH,dtype=np.float32)
    # observation (state space): two parts
    # 'A': design matrix -inf,...,inf : normalized by 1/||A||
    # 'eig': 1+eigenvalues (real), Mx1 real vector -inf,...,inf
    self.observation_space = spaces.Dict({
      'A': spaces.Box(low=np.zeros((self.N,self.M))*(-HIGH),high=np.ones((self.N,self.M))*HIGH,dtype=np.float32), 
      'eig': spaces.Box(low=np.ones((self.N,1))*(-HIGH),high=np.ones((self.N,1))*HIGH,dtype=np.float32)
       })

    self.SNR=0.1 # noise ratio
    # generate design matrix (A), noise-free data vector (y0), ground truth parameter vector (x0)
    #torch.manual_seed(19)
    # design matrix
    self.A=torch.randn(N,M,dtype=torch.float32,requires_grad=False,device=mydevice)
    self.A /=torch.norm(self.A) # normalize
    #torch.manual_seed(time.time())
    # how much sparsity (non-zero values)
    self.Mo=int(torch.randint(3,M,(1,)).data)
    # ground truth parameters
    z0=torch.randn(self.Mo,dtype=torch.float32,requires_grad=False,device=mydevice)
    self.x0=torch.zeros(M,dtype=torch.float32,requires_grad=False,device=mydevice)
    # sparse vector
    self.x0[np.random.randint(0,M,self.Mo)]=z0
    self.y0=torch.matmul(self.A,self.x0)
    # regularization factors for L2 and L1
    self.rho=LOW*torch.ones(self.K,dtype=torch.float32)
    # following used for rendering
    self.x=torch.zeros(M,dtype=torch.float32,requires_grad=False,device=mydevice)
    # following for evaluation (keeping noise the same)
    self.y=None

  def step(self, action, keepnoise=False):
    done=False # make sure to return True at some point
    # update state based on the action  rho = scale*(action)
    self.rho =action*(HIGH-LOW)/2+(HIGH+LOW)/2
    penalty=0
    # make sure rho stays within limits, if this happens, add a penalty
    for ci in range(self.K):
     if self.rho[ci]<LOW:
       self.rho[ci]=LOW
       penalty +=-0.1
     if self.rho[ci]>HIGH:
       self.rho[ci]=HIGH
       penalty +=-0.1

    # generate data (by adding noise to noise-free data)
    if not keepnoise:
      torch.manual_seed(time.time())
      n=torch.randn(self.N,dtype=torch.float32,requires_grad=False,device=mydevice)
      y=self.y0+self.SNR*torch.norm(self.y0)/torch.norm(n)*n
    else:
      y=self.y
    # parameters, initialized to zero
    x=torch.zeros(self.M,requires_grad=True,device=mydevice)

    def lossfunction(A,y,x,alpha=self.rho[0],beta=self.rho[1]):
       Ax=torch.matmul(A,x)
       err=y-Ax
       return torch.norm(err,2)**2+alpha*torch.norm(x,2)**2+beta*torch.norm(x,1)

    opt = LBFGSNew([x],history_size=7,max_iter=10,line_search_fn=True,batch_mode=False)

    # find solution x
    for nepoch in  range(0,20):
       def closure():
         if torch.is_grad_enabled():
           opt.zero_grad()
         loss=lossfunction(self.A,y,x,self.rho[0],self.rho[1])
         if loss.requires_grad:
           loss.backward()
           #print(loss.data.item())
         return loss

       opt.step(closure)


    # Jacobian of model = A
    jac=jacobian(torch.matmul(self.A,x),x).to(mydevice)

    # right hand term = -2 A^T
    df_dx=(lambda yi: gradient(lossfunction(self.A,yi,x,self.rho[0],self.rho[1]), x))
    # no need to pass one-hot vectors, because we calculate d( )/dy^T in one go
    e=torch.ones_like(y) # all ones
    ll=torch.autograd.functional.jacobian(df_dx,e)

    mm=torch.zeros_like(ll).to(mydevice)
    # copy ll because it is modified
    for i in range(self.N):
      ll2=ll[:,i].clone().detach()
      mm[:,i]=inv_hessian_mult(opt,ll2)


    # multiply by Jacobian of model
    B=torch.matmul(jac,mm).to('cpu')
    #print(B)
    # eigenvalues
    E,_=torch.eig(B)
    # 1+eigenvalues (only real part), sorted in ascending order
    EE=E[:,0]+1
    # remember this for rendering later
    self.x=x

    observation={'A': self.A.view(-1).cpu(),
         'eig': EE}
    # final error ||Ax-y||
    final_err=torch.norm(torch.matmul(self.A,x)-y,2).detach()
    # reward : penalize by adding -penalty
    # residual: normalize by data power, eigenvalues = normalize by min/max
    reward=torch.norm(y,2)/final_err+torch.min(EE)/torch.max(EE)+penalty
    #reward.clamp_(-1,1) # clip to [-1,1] - only useful for multiple environments, not here

    # info : meta details {}
    info={}    
    return observation, reward, done, info

  def reset(self):
    #torch.manual_seed(19)
    # design matrix
    self.A=torch.randn(self.N,self.M,dtype=torch.float32,requires_grad=False,device=mydevice)
    self.A /=torch.norm(self.A) # normalize
    #torch.manual_seed(time.time())
    # ground truth parameters
    self.Mo=int(torch.randint(3,self.M,(1,)).data)
    # ground truth parameters
    z0=torch.randn(self.Mo,dtype=torch.float32,requires_grad=False,device=mydevice)
    self.x0=torch.zeros(self.M,dtype=torch.float32,requires_grad=False,device=mydevice)
    # sparse vector
    self.x0[np.random.randint(0,self.M,self.Mo)]=z0
    self.y0=torch.matmul(self.A,self.x0)

    # rho <- rho0
    self.rho=LOW*torch.ones(self.K,dtype=torch.float32)
    observation={'A': self.A.view(-1).cpu(),
            'eig': torch.zeros(self.N,dtype=torch.float32)} 
    return observation  # reward, done, info can't be included

  def render(self, mode='human', showerr=False):
    if not showerr:
      print('%%%%%%%%%%%%%%%%%%%%%%')
      print('%f %f'%(self.rho[0],self.rho[1])) 
      for i in range(self.M):
        print('%d %f %f'%(i,self.x0[i],self.x[i]))
      print('%%%%%%%%%%%%%%%%%%%%%%')
      print('%e %e %f'%(self.rho[0],self.rho[1],torch.norm(self.x0-self.x)))
    else:
      print('%e %e %f'%(self.rho[0],self.rho[1],torch.norm(self.x0-self.x)))

  # find initial solution with initial rho
  def initsol(self):
    # generate data (by adding noise to noise-free data)
    torch.manual_seed(time.time())
    # one observation
    n=torch.randn(self.N,dtype=torch.float32,requires_grad=False,device=mydevice)
    self.y=self.y0+self.SNR*torch.norm(self.y0)/torch.norm(n)*n
    # parameters, initialized to zero
    x=torch.zeros(self.M,requires_grad=True,device=mydevice)

    def lossfunction(A,y,x,alpha=self.rho[0],beta=self.rho[1]):
       Ax=torch.matmul(A,x)
       err=y-Ax
       return torch.norm(err,2)**2+alpha*torch.norm(x,2)**2+beta*torch.norm(x,1)

    opt = LBFGSNew([x],history_size=7,max_iter=10,line_search_fn=True,batch_mode=False)

    # find solution x
    for nepoch in  range(0,20):
       def closure():
         if torch.is_grad_enabled():
           opt.zero_grad()
         loss=lossfunction(self.A,self.y,x,self.rho[0],self.rho[1])
         if loss.requires_grad:
           loss.backward()
           #print(loss.data.item())
         return loss

       opt.step(closure)

    self.x=x

  def close (self):
    pass
