import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import pickle # for saving replaymemory
import os # for saving networks 


# output of first linear layer, increase this if metadata size is larger
INNER_DIM=32

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

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, M, use_influence=False, filename='replaymem_sac.model'):
        self.mem_size = max_size
        self.M=M # metadata
        self.mem_cntr = 0
        self.use_influence=use_influence
        if self.use_influence:
           self.state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_memory_sky= np.zeros((self.mem_size, self.M), dtype=np.float32)
        if self.use_influence:
           self.new_state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory_sky = np.zeros((self.mem_size, self.M), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.hint_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.filename=filename # for saving object

    def store_transition(self, state, action, reward, state_, done, hint):
        index = self.mem_cntr % self.mem_size
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        if self.use_influence:
           self.state_memory_img[index] = state['infmap']
        self.state_memory_sky[index] = state['metadata']
        if self.use_influence:
           self.new_state_memory_img[index] = state_['infmap']
        self.new_state_memory_sky[index] = state_['metadata']
        self.hint_memory[index]=hint

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = {'infmap': self.state_memory_img[batch] if self.use_influence else 0,
                  'metadata': self.state_memory_sky[batch]}
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = {'infmap': self.new_state_memory_img[batch] if self.use_influence else 0,
                  'metadata': self.new_state_memory_sky[batch]}
        terminal = self.terminal_memory[batch]
        hint=self.hint_memory[batch]

        return states, actions, rewards, states_, terminal, hint

    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)

    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
          temp=pickle.load(f)
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          if self.use_influence:
             self.state_memory_img=temp.state_memory_img
          self.state_memory_sky=temp.state_memory_sky
          if self.use_influence:
             self.new_state_memory_img=temp.new_state_memory_img
          self.new_state_memory_sky=temp.new_state_memory_sky
          self.action_memory=temp.action_memory
          self.reward_memory=temp.reward_memory
          self.terminal_memory=temp.terminal_memory
          self.hint_memory=temp.hint_memory

# input: state, output: action space \in R^|action|
class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, M, use_influence=False):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.use_influence= use_influence
        # width and height of image (note dim[0]=channels=1)
        w=input_dims[1]
        h=input_dims[2]
        self.n_actions = n_actions
        # input 1 chan: grayscale image
        if self.use_influence:
           self.conv1=nn.Conv2d(1,16,kernel_size=5, stride=2)
           self.bn1=nn.BatchNorm2d(16)
           self.conv2=nn.Conv2d(16,32,kernel_size=5, stride=2)
           self.bn2=nn.BatchNorm2d(32)
           self.conv3=nn.Conv2d(32,32,kernel_size=5,stride=2)
           self.bn3=nn.BatchNorm2d(32)

        # network to pass M values forward
        self.fc1=nn.Linear(M+n_actions,128)
        self.fc2=nn.Linear(128,INNER_DIM)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        if self.use_influence:
           linear_input_size=convw*convh*32+INNER_DIM # +INNER_DIM from metadata network
        else:
           linear_input_size=INNER_DIM

        self.head=nn.Linear(linear_input_size,1)

        if self.use_influence:
           init_layer(self.conv1)
           init_layer(self.conv2)
           init_layer(self.conv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.head,0.003)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_sac_critic.model')

        self.to(self.device)

    def forward(self, x, z, action): # x is image, z is the sky tensor
        if self.use_influence:
           x=F.relu(self.bn1(self.conv1(x))) # image
           x=F.relu(self.bn2(self.conv2(x)))
           x=F.relu(self.bn3(self.conv3(x)))
           x=T.flatten(x,start_dim=1)

        z=F.relu(self.fc1(T.cat((z,action),1))) # action, sky
        z=F.relu(self.fc2(z))

        if self.use_influence:
           qval=self.head(T.cat((x,z),1))
        else:
           qval=self.head(z)

        return qval 

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=False))

# input: state output: [0,1]^|action|
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, max_action, name, M, use_influence=False):
        super(ActorNetwork, self).__init__()
        self.use_influence=use_influence
        self.input_dims = input_dims
        self.max_action = max_action
        self.reparam_noise=1e-6
        # width and height of image (note dim[0]=channels=1)
        w=input_dims[1]
        h=input_dims[2]
        self.n_actions = n_actions
        # input 1 chan: grayscale image
        if self.use_influence:
           self.conv1=nn.Conv2d(1,16,kernel_size=5, stride=2)
           self.bn1=nn.BatchNorm2d(16)
           self.conv2=nn.Conv2d(16,32,kernel_size=5, stride=2)
           self.bn2=nn.BatchNorm2d(32)
           self.conv3=nn.Conv2d(32,32,kernel_size=5,stride=2)
           self.bn3=nn.BatchNorm2d(32)

        # network to pass  M values (metadata) forward
        self.fc11=nn.Linear(M,128)
        self.fc12=nn.Linear(128,INNER_DIM)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        if self.use_influence:
           linear_input_size=convw*convh*32+INNER_DIM # +INNER_DIM from sky network
        else:
           linear_input_size=INNER_DIM

        self.fc21=nn.Linear(linear_input_size,128)
        self.fc22mu=nn.Linear(128,n_actions)
        self.fc22logsigma=nn.Linear(128,n_actions)

        if self.use_influence:
           init_layer(self.conv1)
           init_layer(self.conv2)
           init_layer(self.conv3)
        init_layer(self.fc11)
        init_layer(self.fc12)
        init_layer(self.fc21)
        init_layer(self.fc22mu,0.003) # last layer
        init_layer(self.fc22logsigma,0.003) # last layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_sac_actor.model')

        self.to(self.device)

    def forward(self, x, z): # x is image, z: metadata tensor
        if self.use_influence:
           x=F.elu(self.bn1(self.conv1(x)))
           x=F.elu(self.bn2(self.conv2(x)))
           x=F.elu(self.bn3(self.conv3(x)))
           x=T.flatten(x,start_dim=1)

        z=T.flatten(z,start_dim=1) # metadata
        z=F.relu(self.fc11(z)) # metadata
        z=F.relu(self.fc12(z))
        if self.use_influence:
           x=F.elu(self.fc21(T.cat((x,z),1)))
        else:
           x=F.elu(self.fc21(z))

        mu=self.fc22mu(x)
        logsigma=self.fc22logsigma(x)
        logsigma = T.clamp(logsigma, min=-20, max=2)
        return mu,logsigma

    def sample_normal(self, x, z, reparameterize=True):
        mu, logsigma = self.forward(x,z)
        sigma=logsigma.exp()
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        # add batch dimension if missing
        if actions.dim()==1:
         actions.unsqueeze_(0)

        actions_t=T.tanh(actions)
        # scale actions
        action = actions_t*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(self.max_action*(1-actions_t.pow(2))+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=False))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,weights_only=False,map_location=T.device('cpu')))

class DemixingAgent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001, n_meta=30, reward_scale=2, alpha=0.1, hint_threshold=0.1, admm_rho=1.0, name_prefix='', use_hint=False, use_influence=False):
        self.use_influence=use_influence
        # Note: n_meta is metadata size
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        # actions are always in [-1,1]
        self.max_action=1
        self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions, n_meta, use_influence=self.use_influence)
    
        # online nets
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=n_meta,
                max_action=self.max_action, name=name_prefix+'a_eval', use_influence=self.use_influence)
        self.critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=n_meta, name=name_prefix+'q_eval_1', use_influence=self.use_influence)
        self.critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=n_meta, name=name_prefix+'q_eval_2', use_influence=self.use_influence)

        self.target_critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=n_meta, name=name_prefix+'q_target_1', use_influence=self.use_influence)
        self.target_critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=n_meta, name=name_prefix+'q_target_2', use_influence=self.use_influence)
        # temperature
        self.alpha=T.tensor(alpha,requires_grad=False,device=mydevice)
        # reward scale
        self.scale= reward_scale

        self.zero_tensor=T.tensor(0.).to(mydevice)
        self.learn_alpha=False
        if self.learn_alpha:
          # -number of bits to represent the actions
          self.target_entropy=-np.sum(n_actions)
          self.alpha_lr=1e-4

        self.use_hint=use_hint
        if self.use_hint:
           self.hint_threshold=hint_threshold
           self.rho=T.tensor(0.0,requires_grad=False,device=mydevice)
           self.admm_rho=admm_rho

        # initialize targets (hard copy)
        self.update_network_parameters(self.target_critic_1, self.critic_1, tau=1.)
        self.update_network_parameters(self.target_critic_2, self.critic_2, tau=1.)

        self.learn_counter=0

    def update_network_parameters(self, target_model, origin_model, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau* local_param.data + (1-tau) * target_param.data)

    def store_transition(self, state, action, reward, state_, terminal, hint):
        self.replaymem.store_transition(state,action,reward,state_,terminal,hint)

    def choose_action(self, observation, evaluation_episode=False):
        if self.use_influence:
           state = T.tensor(observation['infmap'].astype(np.float32),dtype=T.float32).to(mydevice)
           state = state[None,]
        else:
           state=0

        state_sky = T.tensor(observation['metadata'].astype(np.float32),dtype=T.float32).to(mydevice)
        state_sky = state_sky[None,]
        self.actor.eval() # to disable batchnorm
        action_probabilities,_ = self.actor.sample_normal(state,state_sky,reparameterize=False)
        self.actor.train() # to enable batchnorm

        return action_probabilities.cpu().detach().numpy()[0]

    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done, hint = \
                                self.replaymem.sample_buffer(self.batch_size)
 
        if self.use_influence:
           state_batch = T.tensor(state['infmap']).to(mydevice)
        else:
           state_batch=0
        state_batch_sky = T.tensor(state['metadata']).to(mydevice)
        if self.use_influence:
           new_state_batch = T.tensor(new_state['infmap']).to(mydevice)
        else:
           new_state_batch=0
        new_state_batch_sky = T.tensor(new_state['metadata']).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice).unsqueeze(1)
        terminal_batch = T.tensor(done).to(mydevice).unsqueeze(1)
        hint_batch= T.tensor(hint).to(mydevice)

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
            action_m=0.5*action+0.5+self.actor.reparam_noise
            hint_m=0.5*hint+0.5+self.actor.reparam_noise
            # KLD : hint * log(hint/action) = hint * (log hint - log action)
            return hint_m*(T.log(hint_m)-T.log(action_m))

        if not self.use_hint:
          actor_loss = (self.alpha*log_probs - critic_value).mean()

          self.actor.optimizer.zero_grad()
          actor_loss.backward()
          self.actor.optimizer.step()
        else:
             #gfun=(T.max(self.zero_tensor,((F.mse_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
             gfun=(T.max(self.zero_tensor,((kld_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
             actor_loss = (self.alpha*log_probs - critic_value).mean()+0.5*self.admm_rho*gfun*gfun+self.rho*gfun
             self.actor.optimizer.zero_grad()
             actor_loss.backward()
             self.actor.optimizer.step()
             print(f'AC {actor_loss.data.item()} {gfun.data.item()}')


        if self.learn_counter%10==0:
          if self.learn_alpha or self.use_hint:
             with T.no_grad():
               actions, log_probs = self.actor.sample_normal(state_batch, state_batch_sky, reparameterize=False)
               if self.learn_alpha:
                  self.alpha=T.max(self.zero_tensor,self.alpha+self.alpha_lr*((self.target_entropy-(-log_probs)).mean()))

               if self.use_hint:
                      #gfun=(T.max(self.zero_tensor,((F.mse_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
                      gfun=(T.max(self.zero_tensor,((kld_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
                      self.rho+=self.admm_rho*gfun

        if self.learn_counter%100==0:
          if self.use_hint and self.learn_alpha:
              print(f'Actor: {self.learn_counter} {self.rho} {self.alpha}')
          elif self.use_hint:
              print(f'Actor: {self.learn_counter} {self.rho}')
          elif self.learn_alpha:
              print(f'Actor: {self.learn_counter} {self.alpha}')

        self.learn_counter+=1

        self.update_network_parameters(self.target_critic_1, self.critic_1)
        self.update_network_parameters(self.target_critic_2, self.critic_2)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.replaymem.save_checkpoint()

    def load_replaybuffer(self):
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.update_network_parameters(self.target_critic_1, self.critic_1, tau=1.)
        self.update_network_parameters(self.target_critic_2, self.critic_2, tau=1.)

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

#a=DemixingAgent(gamma=0.99, batch_size=32, n_actions=2, n_meta=4,
#             max_mem_size=1000, input_dims=[1,128,128], lr_a=0.001, lr_c=0.001, use_influence=False)
