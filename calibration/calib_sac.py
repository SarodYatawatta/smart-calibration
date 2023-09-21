import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
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

EPS=1e-6

# initialize all layer weights, based on the fan in
def init_layer(layer,sc=None):
  sc = sc or 1./np.sqrt(layer.weight.data.size()[0])
  T.nn.init.uniform_(layer.weight.data, -sc, sc)
  T.nn.init.uniform_(layer.bias.data, -sc, sc)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, M, n_actions):
        self.mem_size = max_size
        self.M=M # how many (maximum) clusters + 1, also equal to
        # how many skymodel components (each 5 values), and one more row
        # note: n_actions = 2 K (spectral and spatial)
        self.mem_cntr = 0
        self.state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_memory_sky= np.zeros((self.mem_size, self.M+1, 5), dtype=np.float32)
        self.new_state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory_sky = np.zeros((self.mem_size, self.M+1, 5), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.hint_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.filename='replaymem_sac.model' # for saving object

    def store_transition(self, state, action, reward, state_, done, hint):
        index = self.mem_cntr % self.mem_size
        self.action_memory[index] = action
        self.hint_memory[index] = hint
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
        hint = self.hint_memory[batch]

        return states, actions, rewards, states_, terminal, hint

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
          self.hint_memory=temp.hint_memory

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

        # network to pass 2K values (n_actions) and 5x(M+1) values forward
        self.fc1=nn.Linear(n_actions+5*(M+1),128)
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
        self.checkpoint_file = os.path.join('./', name+'_sac_critic.model')

        self.to(self.device)

    def forward(self, x, y, z): # x is image, y is the rho (action) z: sky tensor
        x=F.relu(self.bn1(self.conv1(x))) # image
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        y=T.flatten(y,start_dim=1) # action
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
    def __init__(self, lr, input_dims, n_actions, max_action, name, M):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_action = max_action
        self.reparam_noise=EPS
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

        # network to pass  5x(M+1) values (sky) forward
        self.fc11=nn.Linear(5*(M+1),128)
        self.fc12=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from sky network
        self.fc21=nn.Linear(linear_input_size,128)
        self.fc22mu=nn.Linear(128,n_actions)
        self.fc22sigma=nn.Linear(128,n_actions)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc11)
        init_layer(self.fc12)
        init_layer(self.fc21)
        init_layer(self.fc22mu,0.003) # last layer
        init_layer(self.fc22sigma,0.003) # last layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = LBFGSNew(self.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_sac_actor.model')

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
        mu=self.fc22mu(x)
        sigma=self.fc22sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu,sigma

    def sample_normal(self, x, z, reparameterize=True):
        mu, sigma = self.forward(x, z)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        # add batch dimension if missing
        if actions.dim()==1:
         actions.unsqueeze_(0)

        # scale actions, but the env will re-scale it again
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        # mapping Gaussian to bounded distribution in (-1,1)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


class Agent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001, M=3, reward_scale=2, alpha=0.1, hint_threshold=0.1, admm_rho=1.0, name_prefix='', use_hint=False):
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        # make sure we have enough space to accomodate actions
        assert(2*M>=n_actions)
        # actions are always in [-1,1]
        self.max_action=1

        self.replaymem=ReplayBuffer(max_mem_size, input_dims, M, n_actions)
    
        # online nets
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=M,
                max_action=self.max_action, name='a_eval')
        self.critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_eval_1')
        self.critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_eval_2')

        self.target_critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_target_1')
        self.target_critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_target_2')

        # temperature
        self.alpha=T.tensor(alpha,requires_grad=False,device=mydevice)
        # reward scale ~ 1/alpha where alpha*entropy(pi(.|.)) is used for regularization of future reward
        self.scale= reward_scale

        self.use_hint=use_hint
        self.learn_counter=0
        if self.use_hint:
           self.zero_tensor=T.tensor(0.).to(mydevice)
           self.hint_threshold=hint_threshold
           self.rho=T.tensor(0.0,requires_grad=False,device=mydevice)
           self.admm_rho=admm_rho

        # initialize targets (hard copy)
        self.update_network_parameters(self.target_critic_1, self.critic_1, tau=1.)
        self.update_network_parameters(self.target_critic_2, self.critic_2, tau=1.)

    def update_network_parameters(self, target_model, origin_model, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau* local_param.data + (1-tau) * target_param.data)

    def store_transition(self, state, action, reward, state_, terminal, hint):
        self.replaymem.store_transition(state,action,reward,state_,terminal, hint)

    def choose_action(self, observation):
        self.actor.eval() # to disable batchnorm

        state = T.tensor(observation['img'].astype(np.float32),dtype=T.float32).to(mydevice)
        state=state[None,]
        state_sky = T.tensor(observation['sky'].astype(np.float32),dtype=T.float32).to(mydevice)
        state_sky=state_sky[None,]
        actions,_ = self.actor.sample_normal(state,state_sky,reparameterize=False)

        self.actor.train() # to enable batchnorm

        return actions.cpu().detach().numpy()[0]


    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done, hint = \
                                self.replaymem.sample_buffer(self.batch_size)
 
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(state['img']).to(mydevice)
        state_batch_sky = T.tensor(state['sky']).to(mydevice)
        new_state_batch = T.tensor(new_state['img']).to(mydevice)
        new_state_batch_sky = T.tensor(new_state['sky']).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice).unsqueeze(1)
        terminal_batch = T.tensor(done).to(mydevice).unsqueeze(1)
        hint_batch = T.tensor(hint).to(mydevice)

        with T.no_grad():
            new_actions, new_log_probs = self.actor.sample_normal(new_state_batch, new_state_batch_sky, reparameterize=False)
            q1_new_policy = self.target_critic_1.forward(new_state_batch, new_state_batch_sky, new_actions)
            q2_new_policy = self.target_critic_2.forward(new_state_batch, new_state_batch_sky, new_actions)
            min_next_target=T.min(q1_new_policy,q2_new_policy)-self.alpha*new_log_probs
            min_next_target[terminal_batch]=0.0
            new_q_value=reward_batch+self.gamma*min_next_target

        q1_new_policy = self.critic_1.forward(state_batch, state_batch_sky, action_batch)
        q2_new_policy = self.critic_2.forward(state_batch, state_batch_sky, action_batch)
        critic_1_loss = F.mse_loss(q1_new_policy, new_q_value)
        critic_2_loss = F.mse_loss(q2_new_policy, new_q_value)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state_batch, state_batch_sky, reparameterize=True)

        q1_new_policy = self.critic_1.forward(state_batch, state_batch_sky, actions)
        q2_new_policy = self.critic_2.forward(state_batch, state_batch_sky, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        # local function to calculate KLD
        def kld_loss(action,hint):
            # map from [-1,1] to [0,1], add epsilon to avoid 0
            action_m=T.clamp(0.5*action+0.5+self.actor.reparam_noise,min=EPS,max=1.0)
            hint_m=T.clamp(0.5*hint+0.5+self.actor.reparam_noise,min=EPS,max=1.0)
            # KLD : hint * log(hint/action) = hint * (log hint - log action)
            return hint_m*(T.log(hint_m)-T.log(action_m))

        if not self.use_hint:
          actor_loss = (self.alpha*log_probs - critic_value).mean()

          self.actor.optimizer.zero_grad()
          actor_loss.backward()
          self.actor.optimizer.step()
        else:
          gfun=(T.max(self.zero_tensor,((kld_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
          actor_loss = (self.alpha*log_probs - critic_value).mean()+0.5*self.admm_rho*gfun*gfun+self.rho*gfun
          self.actor.optimizer.zero_grad()
          actor_loss.backward()
          self.actor.optimizer.step()
          print(f'AC {actor_loss.data.item()} {gfun.data.item()}')

        if self.learn_counter%10==0:
           if self.use_hint:
               with T.no_grad():
                  gfun=(T.max(self.zero_tensor,((kld_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
                  self.rho+=self.admm_rho*gfun


        self.learn_counter+=1

        self.update_network_parameters(self.target_critic_1, self.critic_1)
        self.update_network_parameters(self.target_critic_2, self.critic_2)


    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.replaymem.save_checkpoint()


    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.update_network_parameters(self.target_critic_1, self.critic_1, tau=1.)
        self.update_network_parameters(self.target_critic_2, self.critic_2, tau=1.)

    def load_models_for_eval(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def print(self):
        print(self.actor)
        print(self.critic_1)

#a=Agent(gamma=0.99, batch_size=32, n_actions=2, M=4,
#             max_mem_size=1000, input_dims=[1,128,128], lr_a=0.001, lr_c=0.001)

