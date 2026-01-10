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
    print(p)

weights=parameter_dict['0.antecedent.1.weight']
bias=parameter_dict['0.antecedent.1.bias']
import matplotlib.pyplot as plt
n_inputs=20
fig,axs=plt.subplots(n_inputs)
x=np.arange(-3,3,0.1)
for which_input in range(n_inputs):
   centers=parameter_dict['0.antecedent.0.center'][which_input]
   sigmas=parameter_dict['0.antecedent.0.sigma'][which_input]
   for rule in range(n_rule):
      y=weights[rule]*np.exp(-(x-centers[rule])**2/(2*sigmas[rule]**2))#+bias[rule]
      axs[which_input].plot(x,y)

plt.savefig('foo.png')
