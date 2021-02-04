import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd.functional
import time
# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

################# jacobian calculation
def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y).to(mydevice)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

# Jacobian \partial(y)/\partial(x)^T
def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0]).to(mydevice)
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y).to(mydevice)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac


################# return inv(Hessian)  q
# where : q vector is multiplied by (approx) inverse Hessian
def inv_hessian_mult(opt,q):
 # opt: LBFGS optimizer struct (at convergence)
 # q: vector to be multiplied : Note this will be modified
 # return : inv(H) q, 
 # inv(H): inverse Hessian approximation using LBFGS update
 pp=opt.state_dict()
 # get curvature pairs, y=delta(grad)
 dirs=pp['state'][0]['old_dirs']
 # s=delta(theta)
 stps=pp['state'][0]['old_stps']
 if dirs==None or stps==None:
   # error, return q
   return q
 N=len(dirs)
 #ro=(pp['state'][0]['ro'])
 #al=(pp['state'][0]['al'])
 ro=torch.zeros(N).to(mydevice)
 al=torch.zeros(N).to(mydevice)
 # last or the (N-1)-th pair are the latest
 ys=dirs[-1].dot(stps[-1])
 yy=dirs[-1].dot(dirs[-1])
 for i in range(N):
  ro[i]=1.0/dirs[i].dot(stps[i])
 for i in range(N-1,-1,-1):
  al[i]=stps[i].dot(q)*ro[i]
  q.add_(dirs[i],alpha=-al[i])
 r=torch.mul(q,ys/yy)
 for i in range(N):
  be_i=dirs[i].dot(r)*ro[i]
  r.add_(stps[i],alpha=al[i]-be_i)

 return r


