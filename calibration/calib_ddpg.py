import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from lbfgsnew import LBFGSNew # custom optimizer
import pickle # for saving replaymemory
import os # for saving networks 

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
    def __init__(self, max_size, input_shape, K, M, n_actions):
        self.mem_size = max_size
        self.K=K # how many clusters
        self.M=M # how many skymodel components (each 5 values) 
        self.mem_cntr = 0
        self.state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_memory_sky= np.zeros((self.mem_size, self.M, 5), dtype=np.float32)
        self.new_state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory_sky = np.zeros((self.mem_size, self.M, 5), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.filename='replaymem_ddpg.model' # for saving object

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.state_memory_img[index] = state['img']
        self.state_memory_sky[index] = state['sky']
        self.new_state_memory_img[index] = state_['img']
        self.new_state_memory_sky[index] = state_['sky']

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = {'img': self.state_memory_img[batch],
                  'sky': self.state_memory_sky[batch]}
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = {'img': self.new_state_memory_img[batch],
                  'sky': self.new_state_memory_sky[batch]}
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
          self.state_memory_img=temp.state_memory_img
          self.state_memory_sky=temp.state_memory_sky
          self.new_state_memory_img=temp.new_state_memory_img
          self.new_state_memory_sky=temp.new_state_memory_sky
          self.action_memory=temp.action_memory
          self.reward_memory=temp.reward_memory
          self.terminal_memory=temp.terminal_memory

# input: state,action output: q-value
class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, M):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        # width and height of image (note dim[0]=channels=1)
        w=input_dims[1]
        h=input_dims[2]
        self.n_actions = n_actions
        # input 1 chan: grayscale image
        self.conv1=nn.Conv2d(1,16,kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,32,kernel_size=5, stride=2)
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,32,kernel_size=5,stride=2)
        self.bn3=nn.BatchNorm2d(32)

        # network to pass K values (action) and 5xM values forward
        self.fc1=nn.Linear(n_actions+5*M,128)
        self.fc2=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from sky+action network
        self.head=nn.Linear(linear_input_size,1)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.head,0.003)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = LBFGSNew(self.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_ddpg_critic.model')

        self.to(self.device)

    def forward(self, x, y, z): # x is image, y is the rho (action) z: sky tensor
        x=F.relu(self.bn1(self.conv1(x))) # image
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        z=T.flatten(z,start_dim=1) # sky
        y=F.relu(self.fc1(T.cat((y,z),1))) # action, sky
        y=F.relu(self.fc2(y))

        qval=self.head(T.cat((x,y),1))

        return qval 

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# input: state output: action
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, M):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        # width and height of image (note dim[0]=channels=1)
        w=input_dims[1]
        h=input_dims[2]
        self.n_actions = n_actions
        # input 1 chan: grayscale image
        self.conv1=nn.Conv2d(1,16,kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,32,kernel_size=5, stride=2)
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,32,kernel_size=5,stride=2)
        self.bn3=nn.BatchNorm2d(32)

        # network to pass  5xM values (sky) forward
        self.fc11=nn.Linear(5*M,128)
        self.fc12=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from sky network
        self.fc21=nn.Linear(linear_input_size,128)
        self.fc22=nn.Linear(128,n_actions)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc11)
        init_layer(self.fc12)
        init_layer(self.fc21)
        init_layer(self.fc22,0.003) # last layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = LBFGSNew(self.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_ddpg_actor.model')

        self.to(self.device)

    def forward(self, x, z): # x is image, z: sky tensor
        x=F.elu(self.bn1(self.conv1(x)))
        x=F.elu(self.bn2(self.conv2(x)))
        x=F.elu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        z=T.flatten(z,start_dim=1) # sky
        z=F.relu(self.fc11(z)) # sky
        z=F.relu(self.fc12(z))
        x=F.elu(self.fc21(T.cat((x,z),1)))
        actions=T.tanh(self.fc22(x)) # in [-1,1], scale up as needed

        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


class Agent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001, K=2, M=3):
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.iter_cntr = 0
        self.replace_target = 100 # how often to update target net
        self.replaymem=ReplayBuffer(max_mem_size, input_dims, K, M, n_actions)
    
        # current net
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=M, name='a_eval')
        self.critic=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_eval')
        # target net
        self.target_actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=M, name='a_target')
        self.target_critic=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_target')
        # noise with memory
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_state_dict = self.critic.state_dict()
        target_critic_dict = self.target_critic.state_dict()
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        actor_state_dict = self.actor.state_dict()
        target_actor_dict = self.target_actor.state_dict()
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def store_transition(self, state, action, reward, state_, terminal):
        self.replaymem.store_transition(state,action,reward,state_,terminal)

    def choose_action(self, observation):
        self.actor.eval() # to disable batchnorm
        state = T.tensor([observation['img']],dtype=T.float32).to(mydevice)
        state_sky = T.tensor([observation['sky']],dtype=T.float32).to(mydevice)
        mu = self.actor.forward(state,state_sky).to(mydevice)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(mydevice)
        return mu_prime.cpu().detach().numpy()


    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return

        
        state, action, reward, new_state, done = \
                                self.replaymem.sample_buffer(self.batch_size)
 
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(state['img']).to(mydevice)
        state_batch_sky = T.tensor(state['sky']).to(mydevice)
        new_state_batch = T.tensor(new_state['img']).to(mydevice)
        new_state_batch_sky = T.tensor(new_state['sky']).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice)
        terminal_batch = T.tensor(done).to(mydevice)


        self.target_actor.eval()
        self.target_critic.eval()
        target_actions = self.target_actor.forward(new_state_batch,new_state_batch_sky)
        critic_value_ = self.target_critic.forward(new_state_batch, target_actions, new_state_batch_sky)

        target = []
        for j in range(self.batch_size):
            target.append(reward_batch[j] + self.gamma*critic_value_[j]*(1-terminal_batch[j]))
        target = T.tensor(target).to(mydevice)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        def closure():
          if T.is_grad_enabled():
            self.critic.optimizer.zero_grad()
          critic_value = self.critic.forward(state_batch, action_batch, state_batch_sky)
          bellman_error=(critic_value-target).clamp(-1,1)
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
          mu = self.actor.forward(state_batch, state_batch_sky)
          actor_loss = -self.critic.forward(state_batch, mu, state_batch_sky)
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
        self.target_actor.train()
        self.critic.train()
        self.target_critic.eval()

    def load_models_for_eval(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.actor.eval()
        self.critic.eval()

    def print(self):
        print(self.actor)
        print(self.critic)

#a=Agent(gamma=0.99, batch_size=32, n_actions=2, M=4,
#             max_mem_size=1000, input_dims=[1,128,128], lr_a=0.001, lr_c=0.001)

