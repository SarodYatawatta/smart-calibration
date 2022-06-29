import numpy as np
import torch
import torchvision
import sys
# append script path
sys.path.append('/home/sarod/work/ttorch/smart-calibration/calibration')
sys.path.append('/home/sarod/work/ttorch/smart-calibration/elasticnet')


#########################################################
# Evaluate performance metrics: model influence function
#########################################################

from autograd_tools import influence_matrix, inverse_hessian_vec_prod
from generate_data import generate_training_data
from transformer_models import *

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# Calculate influence function of the trained model

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
net=TransformerEncoder(num_layers=1,input_dim=input_dims, model_dim=model_dims, num_heads=n_heads, num_classes=K-1, dropout=0.001).to(mydevice)
R=ReplayBuffer(2400,(input_dims,),(K-1,))
R.load_checkpoint(filename='./combined.buffer')
#R.load_checkpoint(filename='./simul_data.buffer')

checkpoint=torch.load('./net.model',map_location=mydevice)
net.load_state_dict(checkpoint['model_state_dict'])

# train LBFGS to get inv. Hessian
from lbfgsnew import LBFGSNew
optimizer = LBFGSNew(net.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
criterion=nn.BCELoss()
batch_size=4
for epoch in range(30):
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


# Reload model
checkpoint=torch.load('./net.model',map_location=mydevice)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

batch_size=4

inputs,labels=R.sample_minibatch(batch_size)
inputs,labels=torch.from_numpy(inputs).to(mydevice),torch.from_numpy(labels).to(mydevice)

outputs=net(inputs)

print(labels)
print(outputs)

#attmaps=net.get_attention_maps(inputs)[0]
#ci=0
#for att in attmaps:
#  print(att) # last column is for target
#  ci+=1

x=inputs[0]
x=x[None,]
y=net(x)

Infl=influence_matrix(net,x,y,optimizer)
#Infl0=influence_matrix(net,x,y)

from scipy.io import savemat
mydict={'lbfgs':Infl.cpu().numpy()}
#mydict={'lbfgs':Infl.cpu().numpy(), 'taylor':Infl0.cpu().numpy()}
savemat('aa.mat',mydict)
print(Infl.shape)
# rows: each of K-1 classes
nrows=Infl.shape[0]
for ci in range(nrows): 
  Z=Infl[ci].view(Ninput,-1)
  # go over directions
  for ck in range(K):
    z=Z[:,ck]
    # unpack to separate components
    infmap=z[:Ninf*Ninf]
    metadata=z[Ninf*Ninf:]
    print(infmap)
    torchvision.utils.save_image(infmap.view((Ninf,Ninf)),'If_'+str(ci)+'_'+str(ck)+'.png',normalize=True) # without normalization all values close to zero
    print(metadata.data)

  #Z=Infl0[ci].view(Ninput,-1)
  ## go over directions
  #for ck in range(K):
  #  z=Z[:,ck]
  #  # unpack to separate components
  #  infmap=z[:Ninf*Ninf]
  #  metadata=z[Ninf*Ninf:]
  #  print(infmap)
  #  torchvision.utils.save_image(infmap.view((Ninf,Ninf)),'Iftay_'+str(ci)+'_'+str(ck)+'.png',normalize=True) # without normalization all values close to zero
  #  print(metadata.data)
