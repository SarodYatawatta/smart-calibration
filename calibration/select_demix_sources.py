import numpy as np
import torch
from generate_data import generate_training_data
from transformer_models import *
import torch.optim as optim
from torch.autograd import Variable

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# Influence map size
Ninf=64
# extra info (separation,azimuth,elevation), ||J||,||C||,|Inf|, freq
Nextra=7
Ninput=Ninf*Ninf+Nextra
# Directions (including target) == heads
K=6
# hidden dimension per head (must be a multiple of heads)
Nmodel=36

n_heads=K
input_dims=Ninput*n_heads
model_dims=Nmodel*n_heads

net=TransformerEncoder(num_layers=1,input_dim=input_dims, model_dim=model_dims, num_heads=n_heads, num_classes=K-1, dropout=0.1).to(mydevice)
R=ReplayBuffer(10,(input_dims,),(K-1,))

criterion=nn.BCELoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)

batch_size=2

load_model=True
save_model=True
if load_model:
    checkpoint=torch.load('./net.model',map_location=mydevice)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.train()
    R.load_checkpoint()

for epoch in range(10):
  x,y=generate_training_data(Ninf=Ninf)
  #x,y=np.random.randn((input_dims)),np.random.randn((K-1))
  R.store_data(x,y)

  if R.mem_cntr < batch_size:
      continue

  inputs,labels=R.sample_minibatch(batch_size)
  inputs,labels=Variable(torch.from_numpy(inputs)).to(mydevice),Variable(torch.from_numpy(labels)).to(mydevice)

  optimizer.zero_grad()
  outputs=net(inputs)
  loss=criterion(outputs,labels)
  loss.backward()
  optimizer.step()
  print(loss.data.item())


if save_model:
    torch.save({
        'model_state_dict':net.state_dict(),
        },'./net.model')
    R.save_checkpoint()
