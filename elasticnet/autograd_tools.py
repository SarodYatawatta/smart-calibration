import numpy as np
import torch
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
# using BFGS approximation for the inverse Hessian
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


################################################
# return gradient vector
# also set gradient to None (to prevent memory leaks)
def gather_flat_grad(model):
   views=[]
   for s in model.parameters():
      if s.grad is not None:
         views.append(s.grad.data.contiguous().view(-1))
         s.grad=None
   return torch.cat(views,0)

# return parameter vector
def gather_flat_parameters(parameters):
   views=[]
   for s in parameters:
     views.append(s.contiguous().view(-1))
   return torch.cat(views,0)

################################################
# Return influcence function size: output x input
# model: nn.Module, with forward() and backward() methods
# Output y = model(x)
# model parameters are theta, stored internally
# opt: LBFGS optimizer, re-used for inverse Hessian
# override_input: if True, use provided input (instead of all ones)
def influence_matrix(model,xinput,youtput,opt=None,override_input=False):
 # xinput: x (input)
 # youtput: y (output)
 
 # pass all ones input via the model
 if override_input:
     x=xinput.detach().clone().requires_grad_(True).to(mydevice)
 else:
     x=torch.ones(xinput.shape,requires_grad=True).to(mydevice)
 # vectorize
 N=x.view(-1).shape[0]
 M=youtput.view(-1).shape[0]

 # labels: vector of 1
 if override_input:
     labels=youtput
 else:
     labels=torch.ones_like(youtput).to(mydevice)

 criterion=torch.nn.MSELoss()
 def l2loss(ytrue,xin):
    ypred=model(xin) 
    return criterion(ypred.view(-1),ytrue.view(-1))


 # storage MxN for Influence function
 If=torch.zeros(M,N).to(mydevice)

 # outer loop over N (because we have to calculate inv(Hessian)
 for ci in range(N):
   # right hand side
   # d( loss() )/d(x^T)
   g=gradient(l2loss(labels,x),x)
   g=g.view(-1)
   # find d( d( loss() )/dx^T ) / d(theta) by backward()
   g[ci].backward()
   ddf_dxdtheta=gather_flat_grad(model)
   # pass this through inverse(Hessian)
   if opt:
     iddf=inv_hessian_mult(opt,ddf_dxdtheta)
   else:
     #iddf=ddf_dxdtheta
     iddf=inverse_hessian_vec_prod(model, criterion, x, labels, ddf_dxdtheta, maxiter=10)

   # inner loop over M
   for cj in range(M):
     y=model(x)
     y=y.view(-1)
     # Jacobian d(model) / dtheta^T
     y[cj].backward()
     jvec=gather_flat_grad(model)

     # find dot product
     If[cj,ci]=torch.dot(iddf,jvec)

 return If



################################################
# Product of Hessian x vector, Perlmutter trick
# taken from https://discuss.pytorch.org/t/efficient-o-n-hessian-vector-product-with-pearlmutter-trick/59037/3
# model: forward model to get parameters, prediction
# criterion: loss function, use MSE loss here
# v: vector to be multiplied by the Hessian
def hessian_vec_prod(model, criterion, inputs, outputs, v):
    # Zero the gradients
    model.zero_grad()
    # Forward-pass
    prediction = model(inputs)
    L = criterion(prediction, outputs)
    # Compute gradient
    y = torch.autograd.grad(L, model.parameters(), create_graph = True, retain_graph = True)
    # Apply R-operator on gradient
    vp = right_op(gather_flat_parameters(y), model.parameters(), v)
    return vp

def right_op(y, x, v):
    # Adapted from: https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
    w = torch.zeros(y.size(), requires_grad = True).to(mydevice)
    g = torch.autograd.grad(y, x, grad_outputs = w, create_graph = True, allow_unused = True)
    r = torch.autograd.grad(gather_flat_parameters(g), w, grad_outputs = v, create_graph = False, allow_unused = True)
    return gather_flat_parameters(r)


# solve H x = v to find x = inv(H) v
# using Taylor expansion, see Koh&Liang 2017, sec 3
# Hiv_0 = v
# Hiv_{j+1} = v+ (I-H) Hiv_j = v + Hiv_j - H (Hiv_j)
def inverse_hessian_vec_prod(model, criterion, inputs, outputs, v, maxiter=10):
   # initial value
   x=v
   # always normalize for convergence, otherwise goes to nan/inf
   x/=torch.norm(x)
   for ci in range(maxiter):
     # find H x
     q = hessian_vec_prod(model,criterion,inputs,outputs,x)
     x = v + x - q
     x/=torch.norm(x)

   return x
