import numpy as np
import torch
import torch.nn as nn
import torch.optim

from sklearn.model_selection import train_test_split

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
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

# we do not include target as an output, n_output=K-1
buffer=TrainingBuffer(n_samples,n_input=M,n_output=K-1)
buffer.load_checkpoint()

X=buffer.x_[:buffer.mem_cntr]
y=buffer.y_[:buffer.mem_cntr]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define TSK model parameters
n_rule = 3  # Num. of rules per each input
lr = 0.01  # learning rate
consbn = True
order = 1 # 0: y = a, 1: y= a x + b
n_class=K-1

# --------- Define antecedent ------------
init_center = antecedent_init_center(X, y, n_rule=n_rule)
gmf = nn.Sequential(
        AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        nn.LayerNorm(n_rule),
        nn.ReLU()
    )# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = nn.Sequential(
        TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None),
        nn.Tanh()
        )
model.to(mydevice)

optimizer=torch.optim.Adam(model.parameters(),lr=lr)

def loss_function(x,y):
    loss=(x-y).norm()**2
    return loss

n_iter=8000
batch_size=256

for ci in range(n_iter):
   #x,y=buffer.sample_buffer(batch_size)
   batch=np.random.choice(x_train.shape[0],batch_size,replace=False)
   x=x_train[batch]
   y=y_train[batch]
   xt=torch.tensor(x).to(mydevice)
   yt=torch.tensor(y).to(mydevice)
   def closure():
    if torch.is_grad_enabled():
       optimizer.zero_grad()
    yout=model(xt)
    loss=loss_function(yt,yout)
    if loss.requires_grad:
      loss.backward()
    print(loss.data.item()/batch_size)
    return loss

   optimizer.step(closure)

torch.save(model.state_dict(),'tsk.model')
model.load_state_dict(torch.load('tsk.model'))

batch=np.random.choice(x_test.shape[0],2,replace=False)
x,y=x_test[batch],y_test[batch]
xt=torch.tensor(x,requires_grad=True).to(mydevice)
yout=model(xt)
yout=yout.detach().cpu().numpy()
print(x)
print(y)
print(yout)

for n,p in model.named_parameters():
    print(n)
    print(p.shape)
