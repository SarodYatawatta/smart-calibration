import numpy as np
import torch
import torch.nn as nn
from demix_sac import DemixingAgent
from demixingenv import DemixingEnv
import pickle

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.tsk import TSK
from regressor_net import RegressorNet

# (try to) use a GPU for computation
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# directions, including target
K=6
# influence map size (not needed)
Ninf=128
# metadata size
M=3*K+2
# number of samples to collect (buffer size)
n_samples=3

n_iter=10

env = DemixingEnv(K=K,Nf=3,Ninf=128,Npix=1024,Tdelta=10,provide_hint=True, provide_influence=False)

# Multilayer perceptron
net=RegressorNet(n_input=M,n_output=K-1,n_hidden=32,name='test')
net.load_checkpoint()
net.eval()
net.to(mydevice)

# Define TSK model parameters
n_rule = 3  # Num. of rules per each input
lr = 0.01  # learning rate
consbn = True
order = 1 # 0: y = a, 1: y= a x + b
n_class=K-1

# --------- Define antecedent ------------
gmf = nn.Sequential(
        AntecedentGMF(in_dim=M, n_rule=n_rule, high_dim=True, init_center=None),
        nn.LayerNorm(n_rule),
        nn.ReLU()
    )# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = nn.Sequential(
        TSK(in_dim=M, out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None),
        nn.Tanh()
        )
model.load_state_dict(torch.load('tsk.model'))
model.eval()
model.to(mydevice)

rewards=np.zeros((n_iter,3))
for ci in range(n_iter):
    observation = env.reset()
    x=observation['metadata'].copy()
    x=x[None,]
    xt=torch.tensor(x).to(mydevice)
    y=net(xt)
    action=np.zeros(K)
    action[:-1]=y.detach().cpu().numpy()
    observation_, reward, done, hint, info = env.step(action)
    xt=torch.tensor(x).to(mydevice)
    y=model(xt)
    action1=np.zeros(K)
    action1[:-1]=y.detach().cpu().numpy()
    observation_, reward1, done, hint, info = env.step(action1)
    observation_, reward2, done, hint, info = env.step(hint)
    print(action)
    print(reward)
    print(action1)
    print(reward1)
    print(hint)
    print(reward2)
    rewards[ci,0]=reward
    rewards[ci,1]=reward1
    rewards[ci,2]=reward2

with open('rewards.pkl','wb') as f:
    pickle.dump(rewards,f)
