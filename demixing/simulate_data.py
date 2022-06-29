import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
# append script path
sys.path.append('/home/sarod/work/ttorch/smart-calibration/calibration')

from generate_data import generate_training_data
from transformer_models import *

#########################################################
# Simulate data and (approx) ground truth labels
#########################################################


# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# Influence map size
Ninf=64
# extra info (separation,azimuth,elevation), log(||J||,||C||,|Inf|), LLR, log(freq)
Nextra=8
Ninput=Ninf*Ninf+Nextra
# Directions (including target) == heads
K=6
# hidden dimension per head (must be a multiple of heads)
Nmodel=66

n_heads=K
input_dims=Ninput*n_heads
model_dims=Nmodel*n_heads

R=ReplayBuffer(3200,(input_dims,),(K-1,))

load_model=False
save_model=False
# save model after this many iterations
save_cadence=4

if load_model:
    R.load_checkpoint()

for epoch in range(30):
  x,y=generate_training_data(Ninf=Ninf)
  R.store_data(x,y)

  if save_model and epoch>0 and epoch%save_cadence==0:
      R.save_checkpoint()

if save_model:
    R.save_checkpoint()
