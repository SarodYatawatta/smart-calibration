import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
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

########################################
# Prioritized experience replay memory
# From https://github.com/Ullar-Kask/TD3-PER
def is_power_of_2 (n):
    return ((n & (n - 1)) == 0) and n != 0

# A binary tree data structure where the parent’s value is the sum of its children
class SumTree():
    #"""
    #This SumTree code is modified version of the code from:
    #https://github.com/jaara/AI-blog/blob/master/SumTree.py
    #https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
    #For explanations please see:
    #https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    #"""
    data_pointer = 0
    data_length = 0
    
    def __init__(self, capacity):
        # Initialize the tree with all nodes = 0,
        
        # Number of leaf nodes (final nodes) that contains experiences
        # Should be power of 2.
        self.capacity = int(capacity)
        assert is_power_of_2(self.capacity), "Capacity must be power of 2."
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        #""" tree:
        #    0
        #   / \
        #  0   0
        # / \ / \
        #0  0 0  0  [Size: capacity] it's at this line where the priority scores are stored
        #"""
        
        # Contains the experiences (so the size of data is capacity)
        #self.data = np.zeros(capacity, dtype=object)
    
    def __len__(self):
        return self.data_length
    
    def add(self, priority):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        # Index to update data frame
        data_index=self.data_pointer
        #self.data[self.data_pointer] = data
        # Update the leaf
        self.update (tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.data_length < self.capacity:
            self.data_length += 1
        return data_index
    
    def update(self, tree_index, priority):
        # change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:
            #"""
            #Here we want to access the line above
            #THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            # 
            #    0
            #   / \
            #  1   2
            # / \ / \
            #3  4 5  [6] 
            # 
            #If we are in leaf at index 6, we updated the priority score
            #We need then to update index 2 node
            #So tree_index = (tree_index - 1) // 2
            #tree_index = (6-1)//2
            #tree_index = 2 (because // round the result)
            #"""
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change    
    
    def get_leaf(self, v):
        # Get the leaf_index, priority value of that leaf and experience associated with that index
        #"""
        #Tree structure and array storage:
        #Tree index:
        #     0         -> storing priority sum
        #    / \
        #  1     2
        # / \   / \
        #3   4 5   6    -> storing priority for experiences
        #Array type for storing:
        #[0,1,2,3,4,5,6]
        #"""
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], data_index
    
    @property
    def total_priority(self):
        return self.tree[0]  # the root


class PER(object):  # stored as ( s, a, r, s_new, done ) in SumTree
    #"""
    #This PER code is modified version of the code from:
    #https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    #"""
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0..1] convert the importance of TD error to priority, often 0.6
    beta = 0.4  # importance-sampling, from initial value increasing to 1, often 0.4
    beta_increment_per_sampling = 1e-4  # annealing the bias, often 1e-3
    absolute_error_upper = 1.   # clipped abs error
    mem_cntr=0
    
    def __init__(self, capacity, input_shape, n_actions, M):
        #"""
        #The tree is composed of a sum tree that contains the priority scores at his leaf and also a indices to data arrays.
        #capacity: should be a power of 2
        #"""
        self.tree = SumTree(capacity)
        self.mem_size=capacity
        self.M=M

        self.state_memory_img=np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.state_memory_sky=np.zeros((self.mem_size,self.M),dtype=np.float32)
        self.new_state_memory_img=np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.new_state_memory_sky=np.zeros((self.mem_size,self.M),dtype=np.float32)
        self.action_memory=np.zeros((self.mem_size,1),dtype=int)
        self.reward_memory=np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory=np.zeros(self.mem_size,dtype=bool)
        self.hint_memory=np.zeros((self.mem_size,n_actions),dtype=np.float32)
        self.filename='prioritized_replaymem_sac.model'
    
    def __len__(self):
        return len(self.tree)
    
    def is_full(self):
        return len(self.tree) >= self.tree.capacity
    
    def store_transition(self, state, action, reward, state_, done, hint, error = None):
        if error is None:
            priority = np.amax(self.tree.tree[-self.tree.capacity:])
            if priority == 0: priority = self.absolute_error_upper
        else:
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.absolute_error_upper)
        index=self.tree.add(priority)
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        self.state_memory_img[index] = state['infmap']
        self.state_memory_sky[index] = state['metadata']
        self.new_state_memory_img[index] = state_['infmap']
        self.new_state_memory_sky[index] = state_['metadata']
        self.hint_memory[index]=hint

        self.mem_cntr+=1

    # method to copy replaymemory from another buffer into self
    def store_transition_from_buffer(self, state, action, reward, state_, done, hint, error = None):
        if error is None:
            priority = np.amax(self.tree.tree[-self.tree.capacity:])
            if priority == 0: priority = self.absolute_error_upper
        else:
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.absolute_error_upper)
        index=self.tree.add(priority)
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        self.state_memory_img[index] = state['infmap']
        self.state_memory_sky[index] = state['metadata']
        self.new_state_memory_img[index] = state_['infmap']
        self.new_state_memory_sky[index] = state_['metadata']
        self.hint_memory[index]=hint

        self.mem_cntr+=1

    
    def sample_buffer(self, batch_size):
        #"""
        #- First, to sample a minibatch of size k the range [0, priority_total] is divided into k ranges.
        #- Then a value is uniformly sampled from each range.
        #- We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        #- Then, we calculate IS weights for each minibatch element.
        #"""
        
        minibatch = []
        
        idxs = np.empty((batch_size,), dtype=np.int32)
        is_weights = np.empty((batch_size,), dtype=np.float32)
        data_idxs = np.empty((batch_size,), dtype=np.int32)
        
        # Calculate the priority segment
        # Divide the Range[0, ptotal] into batch_size ranges
        priority_segment = self.tree.total_priority / batch_size # priority segment
        # Increase the beta each time we sample a new minibatch
        self.beta = np.amin([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        for i in range(batch_size):
            #"""
            #A value is uniformly sampled from each range
            #"""
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            #"""
            #Experience that corresponds to each value is retrieved
            #"""
            index, priority, data_index = self.tree.get_leaf(value)
            sampling_probabilities = priority / self.tree.total_priority
            is_weights[i] = np.power(batch_size * sampling_probabilities, -self.beta)
            idxs[i]= index
            data_idxs[i]=data_index

        is_weights /= is_weights.max()
            
        states={ 'infmap':self.state_memory_img[data_idxs],
                'metadata':self.state_memory_sky[data_idxs] }
        actions=self.action_memory[data_idxs]
        rewards=self.reward_memory[data_idxs]
        states_={ 'infmap':self.new_state_memory_img[data_idxs],
                'metadata':self.new_state_memory_sky[data_idxs] }
        terminal=self.terminal_memory[data_idxs]
        hints=self.hint_memory[data_idxs]
        return states, actions, rewards, states_, terminal, hints, idxs, is_weights
    
    def batch_update(self, idxs, errors):
        #"""
        #Update the priorities on the tree
        #"""
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)
        
        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)
    
    # custom mean squared error with importance sampling weights
    def mse(self,expected,targets,is_weights):
        td_error=expected-targets
        weighted_squared_error=is_weights*td_error*td_error
        return T.sum(weighted_squared_error)/T.numel(weighted_squared_error)

    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
            pickle.dump(self,f)

    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
            temp=pickle.load(f)
            self.tree=temp.tree
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

########################################

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, M, filename='replaymem_sac.model'):
        self.mem_size = max_size
        self.M=M # metadata
        self.mem_cntr = 0
        self.state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_memory_sky= np.zeros((self.mem_size, self.M), dtype=np.float32)
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
        self.state_memory_img[index] = state['infmap']
        self.state_memory_sky[index] = state['metadata']
        self.new_state_memory_img[index] = state_['infmap']
        self.new_state_memory_sky[index] = state_['metadata']
        self.hint_memory[index]=hint

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = {'infmap': self.state_memory_img[batch],
                  'metadata': self.state_memory_sky[batch]}
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = {'infmap': self.new_state_memory_img[batch],
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
          self.state_memory_img=temp.state_memory_img
          self.state_memory_sky=temp.state_memory_sky
          self.new_state_memory_img=temp.new_state_memory_img
          self.new_state_memory_sky=temp.new_state_memory_sky
          self.action_memory=temp.action_memory
          self.reward_memory=temp.reward_memory
          self.terminal_memory=temp.terminal_memory
          self.hint_memory=temp.hint_memory

# input: state, output: action space \in R^|action|
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

        # network to pass M values forward
        self.fc1=nn.Linear(M+n_actions,128)
        self.fc2=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from metadata network
        self.head=nn.Linear(linear_input_size,1)

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
        x=F.relu(self.bn1(self.conv1(x))) # image
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        z=F.relu(self.fc1(T.cat((z,action),1))) # action, sky
        z=F.relu(self.fc2(z))

        qval=self.head(T.cat((x,z),1))

        return qval 

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# input: state output: [0,1]^|action|
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, max_action, name, M):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.max_action = max_action
        self.reparam_noise=1e-6
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

        # network to pass  M values (metadata) forward
        self.fc11=nn.Linear(M,128)
        self.fc12=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from sky network
        self.fc21=nn.Linear(linear_input_size,128)
        self.fc22mu=nn.Linear(128,n_actions)
        self.fc22logsigma=nn.Linear(128,n_actions)

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
        x=F.elu(self.bn1(self.conv1(x)))
        x=F.elu(self.bn2(self.conv2(x)))
        x=F.elu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        z=T.flatten(z,start_dim=1) # metadata
        z=F.relu(self.fc11(z)) # metadata
        z=F.relu(self.fc12(z))
        x=F.elu(self.fc21(T.cat((x,z),1)))
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
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))

class DemixingAgent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001, M=30, reward_scale=2, alpha=0.1, hint_threshold=0.1, admm_rho=1.0, name_prefix='', use_hint=False):
        # Note: M is metadata size
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        # actions are always in [-1,1]
        self.max_action=1
        self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions, M)
    
        # online nets
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=M,
                max_action=self.max_action, name=name_prefix+'a_eval')
        self.critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name=name_prefix+'q_eval_1')
        self.critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name=name_prefix+'q_eval_2')

        self.target_critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name=name_prefix+'q_target_1')
        self.target_critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name=name_prefix+'q_target_2')
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
        state = T.tensor(observation['infmap'].astype(np.float32),dtype=T.float32).to(mydevice)
        state = state[None,]
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
 
        state_batch = T.tensor(state['infmap']).to(mydevice)
        state_batch_sky = T.tensor(state['metadata']).to(mydevice)
        new_state_batch = T.tensor(new_state['infmap']).to(mydevice)
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

#a=DemixingAgent(gamma=0.99, batch_size=32, n_actions=2, M=4,
#             max_mem_size=1000, input_dims=[1,128,128], lr_a=0.001, lr_c=0.001)
