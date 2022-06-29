import numpy as np
from transformer_models import *

#########################################################
# Merge two sets of simulated data into one set
#########################################################

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
R1=ReplayBuffer(2400,(input_dims,),(K-1,))
R.load_checkpoint()
R1.load_checkpoint(filename='./simul_data.buffer')

r1_filled=R1.mem_cntr
r_filled=R.mem_cntr
R1.x[r1_filled:r1_filled+r_filled]=R.x[:r_filled]
R1.y[r1_filled:r1_filled+r_filled]=R.y[:r_filled]
R1.mem_cntr+=r_filled
R1.save_checkpoint(filename='./combined.buffer')
