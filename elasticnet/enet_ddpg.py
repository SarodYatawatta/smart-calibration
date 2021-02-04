import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from lbfgsnew import LBFGSNew # custom optimizer
import pickle # for saving replaymemory

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and T.cuda.is_available():
  mydevice=T.device('cuda')
else:
  mydevice=T.device('cpu')

# initialize all layer weights, based on the fan in
def init_layer(layer,sc=None):
  sc = sc or 1./np.sqrt(layer.weight.data.size()[0])
  T.nn.init.uniform_(layer.weight.data, -sc, sc)
  T.nn.init.uniform_(layer.bias.data, -sc, sc)

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.filename='replaymem_ddpg.model' # for saving object

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.state_memory[index] = T.cat((state['eig'],state['A']))
        self.new_state_memory[index] = T.cat((state_['eig'],state_['A']))

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)
        
    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
          temp=pickle.load(f)
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.state_memory=temp.state_memory
          self.new_state_memory=temp.new_state_memory
          self.action_memory=temp.action_memory
          self.reward_memory=temp.reward_memory
          self.terminal_memory=temp.terminal_memory
 
# input: state,action output: q-value
class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims # state dims
        self.n_actions = n_actions # action dims
        self.fc11 = nn.Linear(*self.input_dims, 512)
        self.fc12 = nn.Linear(512, 256) # before concat with actions

        self.fc21 = nn.Linear(n_actions, 128)
        self.fc22 = nn.Linear(128, 64) # before concat with state

        self.fc3 = nn.Linear(256+64, 1)

        self.bn11 = nn.LayerNorm(512)
        self.bn12 = nn.LayerNorm(256)
        self.bn21 = nn.LayerNorm(128)
        self.bn22 = nn.LayerNorm(64)

        init_layer(self.fc11)
        init_layer(self.fc12)
        init_layer(self.fc21)
        init_layer(self.fc21)
        init_layer(self.fc3,0.003) # last layer 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = LBFGSNew(self.parameters(), history_size=7, max_iter=1, line_search_fn=True,batch_mode=True)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_ddpg_critic.model')

        self.to(self.device)

    def forward(self, state, action):
        x=F.elu(self.bn11(self.fc11(state)))
        x=F.elu(self.bn12(self.fc12(x)))

        y=F.elu(self.bn21(self.fc21(action)))
        y=F.elu(self.bn22(self.fc22(y)))

        # concat state,actions
        z = T.cat((x,y),dim=1)

        qval=self.fc3(z)

        return qval

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))

# input: state output: action
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.n_actions)
        self.bn1 = nn.LayerNorm(512)
        self.bn2 = nn.LayerNorm(256)

        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3,0.003) # last layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = LBFGSNew(self.parameters(), history_size=7, max_iter=1, line_search_fn=True,batch_mode=True)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_ddpg_actor.model')

        self.to(self.device)

    def forward(self, x):
        x=F.elu(self.bn1(self.fc1(x)))
        x=F.elu(self.bn2(self.fc2(x)))
        actions=T.tanh(self.fc3(x)) # in [-1,1], shift, scale up as needed (in the environment)

        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


class Agent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001):
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size

        self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions) 
    
        # current net
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, 
                name='a_eval')
        self.critic=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_eval')
        # target net
        self.target_actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, 
                name='a_target')
        self.target_critic=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_target')
        # noise with memory
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def store_transition(self, state, action, reward, state_, terminal):
        self.replaymem.store_transition(state,action,reward,state_,terminal)

    def choose_action(self, observation):
        self.actor.eval() # to disable batchnorm
        state = T.cat((observation['eig'],observation['A'])).to(mydevice)
        mu = self.actor.forward(state).to(mydevice)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(mydevice)
        return mu_prime.cpu().detach().numpy()

    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return

        
        state, action, reward, new_state, done = \
                                self.replaymem.sample_buffer(self.batch_size)

        state_batch = T.tensor(state).to(mydevice)
        new_state_batch = T.tensor(new_state).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice)
        terminal_batch = T.tensor(done).to(mydevice)

        self.target_actor.eval()
        self.target_critic.eval()
        target_actions = self.target_actor.forward(new_state_batch)
        critic_value_ = self.target_critic.forward(new_state_batch, target_actions)

        target = []
        for j in range(self.batch_size):
            target.append(reward_batch[j] + self.gamma*critic_value_[j]*(1-terminal_batch[j]))
        target = T.tensor(target).to(mydevice)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        def closure():
          if T.is_grad_enabled():
            self.critic.optimizer.zero_grad()
          critic_value = self.critic.forward(state_batch, action_batch)
          bellman_error=(critic_value-target)# dont clip .clamp(-1,1)
          critic_loss=T.norm(bellman_error,2)**2
          if critic_loss.requires_grad:
            critic_loss.backward(retain_graph=True)
          return critic_loss
        self.critic.optimizer.step(closure)
        self.critic.eval()

        self.actor.train()
        def closure1():
          if T.is_grad_enabled():
            self.actor.optimizer.zero_grad()
          mu = self.actor.forward(state_batch)
          actor_loss = -self.critic.forward(state_batch, mu)
          actor_loss = T.mean(actor_loss)
          if actor_loss.requires_grad:
            actor_loss.backward(retain_graph=True)
          return actor_loss
        self.actor.optimizer.step(closure1)
        self.actor.eval()

        self.update_network_parameters()


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.replaymem.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.target_actor.eval()
        self.critic.train()
        self.target_critic.eval()

    def load_models_for_eval(self):
        self.actor.load_checkpoint_for_eval()
        self.critic.load_checkpoint_for_eval()
        self.actor.eval()
        self.critic.eval()



#a=Agent(gamma=0.99, batch_size=32, n_actions=2,  
#                  max_mem_size=1000, input_dims=[11], lr_a=0.001)
