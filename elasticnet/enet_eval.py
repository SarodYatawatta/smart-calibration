import gymnasium as gym
import numpy as np
from enetenv import ENetEnv
# Uncomment which agent to use here TD3,SAC or DDPG
#from enet_ddpg import Agent
#from enet_td3 import Agent
from enet_sac import Agent

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
import scipy.linalg
from scipy.optimize import minimize


# Class for using grid search
class SKEnet(BaseEstimator,RegressorMixin):
  """
   Methods inherited
   BaseEstimator: set_param(), get_param()
   RegressorMixin: score()
  """
  def __init__(self, lambda1, lambda2):
     """
     """
     super().__init__()
     self.lambda1=lambda1
     self.lambda2=lambda2
     self.coeff_=None # solution

  def fit(self,X,Y):
     """
     X=A, Y=y
     solve
     argmin_theta ||A theta -y||^2 + lambda1 ||theta||_1 + lambda2 ||theta||_2^2
     """
     # N: datapoints, M:parameters
     N,M=X.shape
     N1,_=Y.shape 
     assert(N==N1)

     # loss function
     def lossfunction(x,A,y):
       x=np.reshape(x,(M,1))
       loss=self.lambda2*(np.linalg.norm(x,2)**2)+self.lambda1*(np.linalg.norm(x,1))
       err1=np.matmul(A,x)-y
       loss+=np.linalg.norm(err1,2)**2 
       return loss

     # initialize solution to zero
     theta0=np.zeros(M)
     result=minimize(lossfunction,theta0,args=(X,Y),method='L-BFGS-B',)#options={'disp':True, 'ftol':1e-9,'gtol':1e-9})
     # copy solution coeff_=theta
     self.coeff_=np.reshape(result.x,(M,1))

     return self

  def predict(self,X):
     """
     X=A
     output = A theta
     """
     return np.matmul(X,self.coeff_) 


# load a pre-trained model and evaluate with new observations
if __name__ == '__main__':
    M=20
    N=20
    env = ENetEnv(M,N)
    # actions: 2,
    agent = Agent(gamma=0.99, batch_size=64, n_actions=2,
                  max_mem_size=1000, input_dims=[N+N*M], lr_a=0.0001, lr_c=0.0001) # most settings are not relevant because we only evaluate 
    n_games = 2
    
    # load from previous sessions
    agent.load_models_for_eval()

    # for grid search
    sk=SKEnet(lambda1=0.0001,lambda2=0.0001)
    # grid settings
    parameters={'lambda1':[0.001, 0.005, 0.01, 0.05, 0.1], 'lambda2':[0.001, 0.005, 0.01, 0.05, 0.1]}
    clf=GridSearchCV(sk,parameters, cv=2, verbose=0, scoring='neg_mean_squared_error')
   
    for i in range(n_games):
        done = False
        observation = env.reset()
        env.initsol()
        loop=0
        while (not done) and loop<4: # limit number of loops as well
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action,keepnoise=True)
            observation = observation_
            loop+=1

        # Use grid search for comparison
        A=env.A.cpu().numpy()
        y=np.reshape(env.y.cpu().numpy(),(N,1))
        clf.fit(A,y)
        best=clf.best_params_
        print('%d RL %f,%f GR %f,%f'%(i,env.rho[0],env.rho[1],best['lambda1'],best['lambda2']))
        bs=SKEnet(lambda1=best['lambda1'],lambda2=best['lambda2'])
        bs.fit(A,y)

        # compare results
        # ground truth
        x0=env.x0.cpu().numpy()
        # RL solution
        x=env.x.detach().cpu().numpy()
        # Grid search solution
        g=np.squeeze(bs.coeff_)
        print('RL %f GR %f'%(np.linalg.norm(x0-x,1)/np.linalg.norm(x0,1),np.linalg.norm(x0-g,1)/np.linalg.norm(x0,1)))
