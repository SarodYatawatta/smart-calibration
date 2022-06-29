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
# Train a model using simulated data
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

# num_layers below indicate how many attention blocks are stacked
net=TransformerEncoder(num_layers=1,input_dim=input_dims, model_dim=model_dims, num_heads=n_heads, num_classes=K-1, dropout=0.6).to(mydevice)
R=ReplayBuffer(4000,(input_dims,),(K-1,))

criterion=nn.BCELoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)
from lbfgsnew import LBFGSNew
#optimizer = LBFGSNew(net.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)

batch_size=64 # started with 20, dropout=0.1

load_model=True
save_model=True
# save model after this many iterations
save_cadence=4000

if load_model:
    checkpoint=torch.load('./net.model',map_location=mydevice)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.train()
R.load_checkpoint('./simul_data.buffer')

for epoch in range(32000):
  inputs,labels=R.sample_minibatch(batch_size)
  inputs,labels=torch.from_numpy(inputs).to(mydevice),torch.from_numpy(labels).to(mydevice)

  def closure():
    if torch.is_grad_enabled():
      optimizer.zero_grad()
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    if loss.requires_grad:
        loss.backward()
        print('%d %f'%(epoch,loss.data.item()))
    return loss

  optimizer.step(closure)

  if save_model and epoch>0 and epoch%save_cadence==0:
      torch.save({
        'model_state_dict':net.state_dict(),
        },'./net.model')

if save_model:
    torch.save({
        'model_state_dict':net.state_dict(),
        },'./net.model')


R=ReplayBuffer(4000,(input_dims,),(K-1,))
R.load_checkpoint(filename='./test_data.buffer')
batch_size=4

inputs,labels=R.sample_minibatch(batch_size)
inputs,labels=torch.from_numpy(inputs).to(mydevice),torch.from_numpy(labels).to(mydevice)

net.eval()
outputs=net(inputs)

print(labels)
print(outputs)
