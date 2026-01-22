import numpy as np
import torch
import torch.optim

from sklearn.model_selection import train_test_split
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

batch_size=256
n_iter=20000
lr=0.01

# we do not include target as an output, n_output=K-1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=K-1)
buffer.load_checkpoint()

X=buffer.x_[:buffer.mem_cntr]
y=buffer.y_[:buffer.mem_cntr]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


net=RegressorNet(n_input=M,n_output=K-1,n_hidden=32,name='test')
net.to(mydevice)

optimizer=torch.optim.Adam(net.parameters(),lr=lr)

def loss_function(x,y):
    loss=(x-y).norm()**2
    return loss


def test_loss():
   loss=0
   for batch in range(x_test.shape[0]):
      x,y=x_test[batch],y_test[batch]
      xt=torch.tensor(x[None,],requires_grad=False).to(mydevice)
      yt=torch.tensor(y).to(mydevice)
      yout=net(xt)
      loss+=loss_function(yt,yout)

   loss /= x_test.shape[0]

   return loss.data.item()


for ci in range(n_iter):
   batch=np.random.choice(x_train.shape[0],batch_size,replace=False)
   x=x_train[batch]
   y=y_train[batch]
   xt=torch.tensor(x).to(mydevice)
   yt=torch.tensor(y).to(mydevice)

   def closure():
    if torch.is_grad_enabled():
       optimizer.zero_grad()
    yout=net(xt)
    loss=loss_function(yt,yout)
    if loss.requires_grad:
      loss.backward()
    return loss

   optimizer.step(closure)
   error=loss_function(yt,net(xt))/batch_size
   print(f'{ci} {error.data.item()} {test_loss()}')


net.save_checkpoint()
