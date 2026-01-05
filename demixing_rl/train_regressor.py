import numpy as np
import torch
import torch.optim
from regressor_net import RegressorNet
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

batch_size=2
n_iter=2000

# we do not include target as an output, n_output=K-1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=K-1)
buffer.load_checkpoint()

net=RegressorNet(n_input=M,n_output=K-1,n_hidden=32,name='test')
net.to(mydevice)

optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)

def loss_function(x,y):
    loss=(x-y).norm()**2
    return loss


for ci in range(n_iter):
   x,y=buffer.sample_buffer(batch_size)
   xt=torch.tensor(x).to(mydevice)
   yt=torch.tensor(y).to(mydevice)
   def closure():
    if torch.is_grad_enabled():
       optimizer.zero_grad()
    yout=net(xt)
    loss=loss_function(yt,yout)
    if loss.requires_grad:
      loss.backward()
    print(loss.data.item())
    return loss

   optimizer.step(closure)

net.save_checkpoint()
