import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK

from training_buffer import TrainingBuffer
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
# cadence to save buffer
n_cadence=2

# Define TSK model parameters
n_rule = 3  # Num. of rules per each input
lr = 0.01  # learning rate
consbn = True
order = 1
n_class=K-1

# we do not include target as an output, n_output=K-1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=K-1)
buffer.load_checkpoint()

X=buffer.x_[:buffer.mem_cntr]
y=buffer.y_[:buffer.mem_cntr]

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

import matplotlib.pyplot as plt
import torch.optim as optim
from autograd_tools import influence_matrix

If=np.zeros((n_class,M))
model.to(mydevice)
for ci in range(100):
   xt=torch.tensor(X[ci]).to(mydevice)
   xt=xt[None,]
   yt=torch.tensor(y[ci]).to(mydevice)
   yt=yt[None,]
   If+=influence_matrix(model,xt,yt,opt=None,override_input=True).detach().cpu().numpy()
If=If/100
print(If)
