import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
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

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.filename='replaymem_sac.model' # for saving object

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
        self.checkpoint_file = os.path.join('./', name+'_sac_critic.model')

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

# input: state output: value
class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.bn1 = nn.LayerNorm(512)
        self.bn2 = nn.LayerNorm(256)
        self.bn3 = nn.LayerNorm(128)

        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_layer(self.fc4,0.003) # last layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = LBFGSNew(self.parameters(), history_size=7, max_iter=1, line_search_fn=True,batch_mode=True)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_sac_value.model')

        self.to(self.device)

    def forward(self, x):
        x=F.elu(self.bn1(self.fc1(x)))
        x=F.elu(self.bn2(self.fc2(x)))
        x=F.elu(self.bn3(self.fc3(x)))
        value=self.fc4(x) # scalar value

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))



# input: state output: action
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, max_action, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4mu = nn.Linear(128, self.n_actions)
        self.fc4sigma = nn.Linear(128, self.n_actions)
        self.bn1 = nn.LayerNorm(512)
        self.bn2 = nn.LayerNorm(256)
        self.bn3 = nn.LayerNorm(128)

        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_layer(self.fc4mu,0.003) # last layer
        init_layer(self.fc4sigma,0.003) # last layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.optimizer = LBFGSNew(self.parameters(), history_size=7, max_iter=1, line_search_fn=True,batch_mode=True)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_sac_actor.model')

        self.to(self.device)

    def forward(self, x):
        x=F.elu(self.bn1(self.fc1(x)))
        x=F.elu(self.bn2(self.fc2(x)))
        x=F.elu(self.bn3(self.fc3(x)))
        mu=self.fc4mu(x) # 
        sigma=self.fc4sigma(x) # 
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu,sigma 

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
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
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        #log_probs = log_probs.sum(1, keepdim=True)
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
            max_mem_size=100, tau=0.001, reward_scale=2):
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        # actions are always in [-1,1] (rescaled later by the env)
        self.max_action=1
        self.min_action=-1

        self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions) 
    
        # online nets
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, max_action=self.max_action,
                name='a_eval')
        self.critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_eval_1')
        self.critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_eval_2')
        self.value=ValueNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='v_eval')
        # target nets
        self.target_value=ValueNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='v_target')
        # reward scale
        self.scale= reward_scale

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        v_params = self.value.named_parameters()
        v_dict = dict(v_params)
        target_v_params = self.target_value.named_parameters()
        target_v_dict = dict(target_v_params)
        for name in target_v_dict:
            target_v_dict[name] = tau*v_dict[name].clone() + \
                                      (1-tau)*target_v_dict[name].clone()
        self.target_value.load_state_dict(target_v_dict)

    def store_transition(self, state, action, reward, state_, terminal):
        self.replaymem.store_transition(state,action,reward,state_,terminal)

    def choose_action(self, observation):
        state = T.cat((observation['eig'],observation['A'])).to(mydevice)
        self.actor.eval() # to disable batchnorm
        actions,_= self.actor.sample_normal(state, reparameterize=False)
        self.actor.train() # to enable batchnorm

        return actions.cpu().detach().numpy()[0]

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

        value = self.value(state_batch).view(-1)
        value_ = self.target_value(new_state_batch).view(-1)
        value_[terminal_batch] = 0.0

        actions, log_probs = self.actor.sample_normal(state_batch, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state_batch, actions)
        q2_new_policy = self.critic_2.forward(state_batch, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state_batch, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state_batch, actions)
        q2_new_policy = self.critic_2.forward(state_batch, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward_batch + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state_batch, action_batch).view(-1)
        q2_old_policy = self.critic_2.forward(state_batch, action_batch).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.replaymem.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.value.train()
        self.target_value.eval()
        self.critic_1.train()
        self.critic_2.train()
        self.update_network_parameters(tau=1.)

    def load_models_for_eval(self):
        self.actor.load_checkpoint_for_eval()
        self.critic_1.load_checkpoint_for_eval()
        self.critic_2.load_checkpoint_for_eval()
        self.value.load_checkpoint_for_eval()
        self.actor.eval()
        self.value.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def print(self):
        print(self.actor)
        print(self.critic_1)
        print(self.value)

#a=Agent(gamma=0.99, batch_size=32, n_actions=2,  
#                max_mem_size=1000, input_dims=[11], lr_a=0.001, lr_c=0.001)
