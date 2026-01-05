import numpy as np
from demixingenv import DemixingEnv
from training_buffer import TrainingBuffer

# directions, including target
K=6
# influence map size (not needed)
Ninf=128
# metadata size
M=3*K+2
# number of samples to collect (buffer size)
n_samples=3000
# cadence to save buffer
n_cadence=100

#metadata: separation,az,el ( x K),log(freq),stations
#metadata scaling factor
META_SCALE=1e-3
env = DemixingEnv(K=K,Nf=3,Ninf=128,Npix=1024,Tdelta=10,provide_hint=True, provide_influence=False)

# we do not include target as an output, n_output=K-1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=K-1)
for ci in range(n_samples):
   observation = env.reset()
   hint=env.get_hint()
   x=observation['metadata']
   y=hint[:-1]
   buffer.store_observation(x,y)
   if ci>0 and ci%n_cadence==0:
     buffer.save_checkpoint()

buffer.save_checkpoint()
