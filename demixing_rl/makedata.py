import numpy as np
from demixingfullenv import DemixingFullEnv
from training_buffer import TrainingBuffer

# directions, including target
K=6
# metadata size
M=3*K+2
# number of samples to collect (buffer size)
n_samples=3000
# number of iterations to make data
n_iter=40

#metadata: separation,az,el ( x K),log(freq),stations
#metadata scaling factor
META_SCALE=1e-3
env = DemixingFullEnv(K=K,Nf=3,Npix=1024,Tdelta=10,provide_hint=True)

# we do not include target as an output, n_output=K-1+K-1+1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=2*(K-1)+1)
# load already exisiting file (note: size of buffer will be 
# determined by the loaded file)
#buffer.load_checkpoint()

n_filled=0
for ci in range(n_iter):
   observation = env.reset()
   hint=env.get_hint()
   # exclude all zero hints
   if hint.any():
     x=observation['metadata']
     y=hint
     n_filled +=1
     buffer.store_observation(x,y)
     # always save buffer after each iteration
     buffer.save_checkpoint()
   print(f'Iteration {ci}/{n_iter} filled {n_filled}')

buffer.save_checkpoint()
