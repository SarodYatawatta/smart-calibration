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
        self.filename='prioritized_replaymem_sac.model'
    
    def __len__(self):
        return len(self.tree)
    
    def is_full(self):
        return len(self.tree) >= self.tree.capacity
    
    def store_transition(self, state, action, reward, state_, done, error = None):
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

        self.mem_cntr+=1

    # method to copy replaymemory from another buffer into self
    def store_transition_from_buffer(self, state, action, reward, state_, done, error = None):
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
        return states, actions, rewards, states_, terminal, idxs, is_weights
    
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

########################################

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, M):
        self.mem_size = max_size
        self.M=M # metadata
        self.mem_cntr = 0
        self.state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_memory_sky= np.zeros((self.mem_size, self.M), dtype=np.float32)
        self.new_state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory_sky = np.zeros((self.mem_size, self.M), dtype=np.float32)
        # action is a scalar value in [0,2^n_actions]
        self.action_memory = np.zeros((self.mem_size,1), dtype=int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.filename='replaymem_sac.model' # for saving object

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.state_memory_img[index] = state['infmap']
        self.state_memory_sky[index] = state['metadata']
        self.new_state_memory_img[index] = state_['infmap']
        self.new_state_memory_sky[index] = state_['metadata']

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
        self.fc1=nn.Linear(M,128)
        self.fc2=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from metadata network
        self.head=nn.Linear(linear_input_size,n_actions)

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

    def forward(self, x, z): # x is image, z is the sky tensor
        x=F.relu(self.bn1(self.conv1(x))) # image
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        z=F.relu(self.fc1(z)) # action, sky
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
        self.fc22=nn.Linear(128,n_actions)
        self.head=nn.Softmax(dim=1)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc11)
        init_layer(self.fc12)
        init_layer(self.fc21)
        init_layer(self.fc22,0.003) # last layer

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
        mu=self.fc22(x)
        return self.head(mu)

    def get_action_info(self, x, z):
        mu = self.forward(x, z)
        mu_eps= mu==0.0
        mu_eps = mu_eps.float()*1e-8

        log_mu = T.log(mu+mu_eps)

        return mu, log_mu

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))

# SAC in discrete action space, based on
# https://github.com/Felhof/DiscreteSAC
# and
# https://github.com/ku2482/sac-discrete.pytorch
class DemixingAgent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001, M=30, warmup=1000, update_interval=4, prioritized=True):
        # Note: M is metadata size
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        # actions are always in [-1,1]
        self.max_action=1
        self.warmup=warmup
        self.update_interval=update_interval
        self.time_step=0
        self.learn_step_cntr=0

        self.prioritized=prioritized
        if not self.prioritized:
           self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions, M)
        else:
           self.replaymem=PER(max_mem_size, input_dims, n_actions, M)
           self.idxs=None
    
        # online nets
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=M,
                max_action=self.max_action, name='a_eval')
        self.critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_eval_1')
        self.critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_eval_2')

        self.target_critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_target_1')
        self.target_critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_target_2')

        # reward scale ~ 1/alpha where alpha*entropy(pi(.|.)) is used for regularization of future reward
        self.target_entropy=0.98* -np.log(1/n_actions)
        self.log_alpha=T.tensor(np.log(1.),requires_grad=True)
        self.alpha= self.log_alpha.exp()
        self.alpha_optimizer=T.optim.Adam([self.log_alpha],lr=1e-4)

        # initialize targets (hard copy)
        self.update_network_parameters(self.target_critic_1, self.critic_1, tau=1.)
        self.update_network_parameters(self.target_critic_2, self.critic_2, tau=1.)

    def update_network_parameters(self, target_model, origin_model, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau* local_param.data + (1-tau) * target_param.data)

    def store_transition(self, state, action, reward, state_, terminal):
        self.replaymem.store_transition(state,action,reward,state_,terminal)

    def choose_action(self, observation, evaluation_episode=False):
        if self.time_step<self.warmup:
            self.time_step+=1
            action=np.random.choice(range(self.n_actions))
            return action

        self.actor.eval() # to disable batchnorm
        state = T.tensor(observation['infmap'].astype(np.float32),dtype=T.float32).to(mydevice)
        state = state[None,]
        state_sky = T.tensor(observation['metadata'].astype(np.float32),dtype=T.float32).to(mydevice)
        state_sky = state_sky[None,]
        action_probabilities = self.actor.forward(state,state_sky)
        self.actor.train() # to enable batchnorm

        action_probs=action_probabilities.squeeze(0).cpu().detach().numpy()
        if evaluation_episode:
           action=np.argmax(action_probs)
        else:
           action=np.random.choice(range(self.n_actions),p=action_probs)

        self.time_step+=1
        return action

    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return
        
        if not self.prioritized:
            state, action, reward, new_state, done = \
                                self.replaymem.sample_buffer(self.batch_size)
        else:
            state, action, reward, new_state, done, idxs, is_weights = \
                                self.replaymem.sample_buffer(self.batch_size)
            self.idxs=idxs
 
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(state['infmap']).to(mydevice)
        state_batch_sky = T.tensor(state['metadata']).to(mydevice)
        new_state_batch = T.tensor(new_state['infmap']).to(mydevice)
        new_state_batch_sky = T.tensor(new_state['metadata']).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice)
        terminal_batch = T.tensor(done).to(mydevice)

        if self.prioritized:
            is_weight=T.tensor(is_weights).to(mydevice)
        else:
            is_weight=1

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

        critic_1_loss, critic_2_loss =\
           self.critic_loss(state_batch,state_batch_sky,new_state_batch,new_state_batch_sky,action_batch,reward_batch,terminal_batch, is_weight)

        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        actor_loss, log_action_probabilities=self.actor_loss(state_batch,state_batch_sky, is_weight)
        actor_loss.backward()
        self.actor.optimizer.step()

        alpha_loss=self.temperature_loss(log_action_probabilities)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha=self.log_alpha.exp()
        
        self.learn_step_cntr+=1
        if self.learn_step_cntr % self.update_interval == 0:
           self.update_network_parameters(self.target_critic_1, self.critic_1)
           self.update_network_parameters(self.target_critic_2, self.critic_2)

    def critic_loss(self,state_batch,state_batch_sky,new_state_batch,new_state_batch_sky,action_batch,reward_batch,terminal_batch, is_weight):
        with T.no_grad():
            action_probabilities,log_action_probabilities=self.actor.get_action_info(new_state_batch,new_state_batch_sky)
            next_q_1_target=self.target_critic_1.forward(new_state_batch,new_state_batch_sky)
            next_q_2_target=self.target_critic_2.forward(new_state_batch,new_state_batch_sky)
            soft_state_values=(action_probabilities 
                    * (T.min(next_q_1_target,next_q_2_target) - self.alpha *log_action_probabilities)).sum(dim=1)

            next_q_values=reward_batch + ~terminal_batch * self.gamma* soft_state_values
        # select q values indexes by the action (scalar value)
        soft_q_1=self.critic_1(state_batch,state_batch_sky).gather(1,action_batch).squeeze(-1)
        soft_q_2=self.critic_2(state_batch,state_batch_sky).gather(1,action_batch).squeeze(-1)
        critic1_square_err=T.nn.MSELoss(reduction='none')(soft_q_1,next_q_values)
        critic2_square_err=T.nn.MSELoss(reduction='none')(soft_q_2,next_q_values)

        if self.prioritized:
            errors1=critic1_square_err.detach().cpu().numpy()
            errors2=critic2_square_err.detach().cpu().numpy()
            self.replaymem.batch_update(self.idxs,0.5*(errors1+errors2))

        if self.prioritized:
          critic_1_loss=(is_weight*critic1_square_err).mean()
          critic_2_loss=(is_weight*critic2_square_err).mean()
        else:
          critic_1_loss=critic1_square_err.mean()
          critic_2_loss=critic2_square_err.mean()

        return critic_1_loss, critic_2_loss

    def actor_loss(self, state_batch, state_batch_sky, is_weight):
        action_probabilities, log_action_probabilities=self.actor.get_action_info(state_batch, state_batch_sky)
        q_1=self.critic_1(state_batch,state_batch_sky)
        q_2=self.critic_2(state_batch,state_batch_sky)

        inner_term=self.alpha*log_action_probabilities-T.min(q_1,q_2)
        policy_loss=(is_weight*action_probabilities*inner_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self,log_action_probabilities):
        alpha_loss=-(self.log_alpha*(log_action_probabilities+self.target_entropy).detach()).mean()
        return alpha_loss

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

#a=DemixingAgent(gamma=0.99, batch_size=32, n_actions=2**2, M=4,
#             max_mem_size=1000, input_dims=[1,128,128], lr_a=0.001, lr_c=0.001)
