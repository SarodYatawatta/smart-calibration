import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK


# directions, including target
K=6
# influence map size (not needed)
Ninf=128
# metadata size
M=3*K+2
# number of samples to collect (buffer size)
n_samples=3
# cadence to save buffer
n_cadence=2

# Define TSK model parameters
n_rule = 3  # Num. of rules per each input
lr = 0.01  # learning rate
consbn = True
order = 1
n_class=K-1

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

parameter_dict={}
for n,p in model.named_parameters():
    parameter_dict[n]=p.detach().numpy()
    print(n)
    print(p.shape)

weights=parameter_dict['0.antecedent.1.weight']
bias=parameter_dict['0.antecedent.1.bias']
import matplotlib.pyplot as plt
n_inputs=6
# each column : seperation,az,el (xK), log(freq), n_stations
ylabels=['CasA','CygA','HerA','TauA','VirA','Target']

META_SCALE=1e3
xmean=np.zeros(M,dtype=np.float32)
xmean[0:5]=64
xmean[12:18]=30
xmean[18]=20
xmean[19]=50
x=(np.arange(-0.3,0.3,0.001))*META_SCALE

fig,axs=plt.subplots(n_inputs,sharex=True)
fig.suptitle('Separation')

for which_input in range(6):
   centers=parameter_dict['0.antecedent.0.center'][which_input]
   sigmas=parameter_dict['0.antecedent.0.sigma'][which_input]
   for rule in range(n_rule):
      y=np.exp(-((x/META_SCALE)-centers[rule])**2/(2*sigmas[rule]**2))
      axs[which_input%6].plot(x+xmean[which_input],y)
      axs[which_input%6].set_ylabel(ylabels[which_input%6])
      axs[which_input%6].grid(1)

axs[-1].set_xlabel('Degrees')
plt.savefig('gmf_separation.png')
fig.clf()

fig,axs=plt.subplots(n_inputs,sharex=True)
fig.suptitle('Azimuth')

for which_input in range(6,12):
   centers=parameter_dict['0.antecedent.0.center'][which_input]
   sigmas=parameter_dict['0.antecedent.0.sigma'][which_input]
   for rule in range(n_rule):
      y=np.exp(-((x/META_SCALE)-centers[rule])**2/(2*sigmas[rule]**2))
      axs[which_input%6].plot(x+xmean[which_input],y)
      axs[which_input%6].set_ylabel(ylabels[which_input%6])
      axs[which_input%6].grid(1)

axs[-1].set_xlabel('Degrees')
plt.savefig('gmf_azimuth.png')

fig.clf()

fig,axs=plt.subplots(n_inputs,sharex=True)
fig.suptitle('Elevation')

for which_input in range(12,18):
   centers=parameter_dict['0.antecedent.0.center'][which_input]
   sigmas=parameter_dict['0.antecedent.0.sigma'][which_input]
   for rule in range(n_rule):
      y=np.exp(-((x/META_SCALE)-centers[rule])**2/(2*sigmas[rule]**2))
      axs[which_input%6].plot(x+xmean[which_input],y)
      axs[which_input%6].set_ylabel(ylabels[which_input%6])
      axs[which_input%6].grid(1)

axs[-1].set_xlabel('Degrees')
plt.savefig('gmf_elevation.png')

fig.clf()

fig,axs=plt.subplots(2,sharex=False)
fig.suptitle('Frequency and Stations')

ylabels=['log(Frequency)','Stations']
for which_input in range(18,20):
   centers=parameter_dict['0.antecedent.0.center'][which_input]
   sigmas=parameter_dict['0.antecedent.0.sigma'][which_input]
   for rule in range(n_rule):
      y=np.exp(-((x/META_SCALE)-centers[rule])**2/(2*sigmas[rule]**2))
      if which_input==18:
         axs[which_input%2].plot(x+xmean[which_input],y)
      else:
         axs[which_input%2].plot(x+xmean[which_input],y)
      axs[which_input%2].set_xlabel(ylabels[which_input%2])
      axs[which_input%2].grid(1)

plt.savefig('gmf_freq_stat.png')

fig.clf()
