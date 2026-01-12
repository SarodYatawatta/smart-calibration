import numpy as np
import torch
import torch.optim

from training_buffer import TrainingBuffer

# directions, including target
K=6
# influence map size (not needed)
Ninf=128
# metadata size
M=3*K+2
# number of samples to collect (buffer size)
n_samples=3

# we do not include target as an output, n_output=K-1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=K-1)
buffer.load_checkpoint()

META_SCALE=1e-3
# each column : seperation,az,el (xK), log(freq), n_stations
ylabels=['CasA','CygA','HerA','TauA','VirA','Target']
# sources: CasA, CygA, HerA, TauA, VirA, target
X=buffer.x_[:buffer.mem_cntr]
print(X.shape)
y=buffer.y_[:buffer.mem_cntr]
print(y.shape)

import matplotlib.pyplot as plt
n_directions=K
fig,axs=plt.subplots(n_directions)
# separation, azimuth, elevation
fig.suptitle('Separation/deg')
#fig.suptitle('Azimuth/deg')
#fig.suptitle('Elevation/deg')
for dir_id in range(n_directions):
    x=X[:,0*K+dir_id]/META_SCALE # 0*K, 1*K or 2*K
    axs[dir_id].plot(x,'.')
    axs[dir_id].set_ylabel(ylabels[dir_id])
axs[-1].set_xlabel('Simulation number')

plt.savefig('foo.png')

import pickle
# plot rewards
with open('rewards.pkl','rb') as f:
    rewards=pickle.load(f)

# scale back
rewards=rewards*3559.0+859
fig=plt.figure(2)
plt.plot(rewards)
plt.legend(['Multilayer perceptron','Fuzzy network','Data driven'])
plt.xlabel('Trial')
plt.ylabel('Reward')
plt.savefig('bar.png')
