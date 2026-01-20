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

META_SCALE=1e3
xmean=np.zeros(M,dtype=np.float32)
xmean[0:5]=64
xmean[12:18]=30
xmean[18]=20
xmean[19]=50
xmean/=META_SCALE
X=X-xmean

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Define TSK model parameters
n_rule = 3  # Num. of rules per each input
lr = 0.01  # learning rate
consbn = True
order = 1 # 0: y = a, 1: y= a x + b
n_class=K-1

# --------- Define antecedent ------------
init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)
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

# keep parameters dict
parameter_dict={}
for n,p in model.named_parameters():
    parameter_dict[n]=p

optimizer=torch.optim.Adam(model.parameters(),lr=lr)

def loss_function(x,y):
    loss=(x-y).norm()**2
    return loss

def center_difference_loss():
    #total_loss=0
    inv_total_loss=0
    # go through each input
    for which_input in range(M):
        # get antecedent center
        centers=parameter_dict['0.antecedent.0.center'][which_input]
        #dist_norm=0
        inv_dist_norm=0
        for rule in range(n_rule):
            for rule1 in range(rule+1,n_rule):
               dist=centers[rule]-centers[rule1]
               #dist_norm+=dist**2
               inv_dist_norm+=1/(dist**2+1e-5)
        #total_loss+=dist_norm
        inv_total_loss+=inv_dist_norm

    return inv_total_loss/(M*n_rule*(n_rule-1)/2)
        
def sigma_loss():
    total_loss=0
    # go through each input
    for which_input in range(M):
        # get antecedent sigma 
        sigmas=parameter_dict['0.antecedent.0.sigma'][which_input]
        for rule in range(n_rule):
            dist=sigmas[rule]**2
            total_loss+=dist

    return total_loss/(M*n_rule)
 
def test_loss():
   loss=0
   for batch in range(x_test.shape[0]):
      x,y=x_test[batch],y_test[batch]
      xt=torch.tensor(x[None,],requires_grad=False).to(mydevice)
      yt=torch.tensor(y).to(mydevice)
      yout=model(xt)
      loss+=loss_function(yt,yout)

   loss /= x_test.shape[0]

   return loss.data.item()

n_iter=20000
batch_size=256

# center difference reg
g1=1e-4
# sigma reg
g2=1e-4

for ci in range(n_iter):
   batch=np.random.choice(x_train.shape[0],batch_size,replace=False)
   x=x_train[batch]
   y=y_train[batch]
   xt=torch.tensor(x).to(mydevice)
   yt=torch.tensor(y).to(mydevice)
   def closure():
     if torch.is_grad_enabled():
       optimizer.zero_grad()
     yout=model(xt)
     center_diff=center_difference_loss()
     sigma_var=sigma_loss()
     loss=loss_function(yt,yout)/batch_size+g1*center_diff+g2*sigma_var
     if loss.requires_grad:
       loss.backward()
     return loss

   optimizer.step(closure)
   error=loss_function(yt,model(xt))/batch_size
   center_diff=center_difference_loss()
   sigma_var=sigma_loss()
   print(f'{ci} {error.data.item()} {center_diff.data.item()} {sigma_var.data.item()} {test_loss()}')

torch.save(model.state_dict(),'tsk.model')
