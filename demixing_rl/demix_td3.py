import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.action_memory=np.zeros((self.mem_size,n_actions),dtype=np.float32)
        self.reward_memory=np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory=np.zeros(self.mem_size,dtype=bool)
        self.filename='prioritized_replaymem_td3.model'
    
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
        self.M=M # how many medatadata components
        self.mem_cntr = 0
        self.state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_memory_sky= np.zeros((self.mem_size, self.M), dtype=np.float32)
        self.new_state_memory_img = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory_sky = np.zeros((self.mem_size, self.M), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.filename='replaymem_td3.model' # for saving object

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

        # network to pass K values (action) and M values forward
        self.fc1=nn.Linear(n_actions+M,128)
        self.fc2=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from metadata+action network
        self.head=nn.Linear(linear_input_size,1)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.head,0.003)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_td3_critic.model')

        self.to(self.device)

    def forward(self, x, y, z): # x is image, y is the action z: metadata
        x=F.relu(self.bn1(self.conv1(x))) # image
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        z=T.flatten(z,start_dim=1) # matadata
        y=F.relu(self.fc1(T.cat((y,z),1))) # action, metadata
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

        # network to pass  M values (metadata) forward
        self.fc11=nn.Linear(M,128)
        self.fc12=nn.Linear(128,16)

        # function to calculate output image size per single conv operation
        def conv2d_size_out(size,kernel_size=5,stride=2):
           return (size-(kernel_size-1)-1)//stride + 1

        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size=convw*convh*32+16 # +16 from metadata network
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
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_td3_actor.model')

        self.to(self.device)

    def forward(self, x, z): # x is image, z: metadata
        x=F.elu(self.bn1(self.conv1(x)))
        x=F.elu(self.bn2(self.conv2(x)))
        x=F.elu(self.bn3(self.conv3(x)))
        x=T.flatten(x,start_dim=1)
        z=T.flatten(z,start_dim=1) # metadata
        z=F.relu(self.fc11(z)) # metadata
        z=F.relu(self.fc12(z))
        x=F.elu(self.fc21(T.cat((x,z),1)))
        actions=T.tanh(self.fc22(x)) # in [-1,1], scale and shift as needed (in the env)

        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


class DemixingAgent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001, M=30, update_actor_interval=2, warmup=1000, noise=0.1):
        # Note: M is metadata size
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        # actions are always in [-1,1]
        self.max_action=1
        self.min_action=-1
        self.learn_step_cntr=0
        self.time_step=0
        self.warmup=warmup
        self.update_actor_interval=update_actor_interval

        self.prioritized=True
        if not self.prioritized:
         self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions, M)
        else:
         self.replaymem=PER(max_mem_size, input_dims, n_actions, M)

    
        # online nets
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=M,
                name='a_eval')
        self.critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_eval_1')
        self.critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_eval_2')
        # target nets
        self.target_actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, M=M,
                name='a_target')
        self.target_critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_target_1')
        self.target_critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, M=M, name='q_target_2')
        # noise fraction
        self.noise = noise

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau


        actor_state_dict = self.actor.state_dict()
        target_actor_dict = self.target_actor.state_dict()
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        critic_state_dict = self.critic_1.state_dict()
        target_critic_dict = self.target_critic_1.state_dict()
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
        self.target_critic_1.load_state_dict(critic_state_dict)

        critic_state_dict = self.critic_2.state_dict()
        target_critic_dict = self.target_critic_2.state_dict()
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
        self.target_critic_2.load_state_dict(critic_state_dict)


    def store_transition(self, state, action, reward, state_, terminal):
        self.replaymem.store_transition(state,action,reward,state_,terminal)

    def choose_action(self, observation):
        if self.time_step<self.warmup:
          mu=T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(mydevice)
        else:
          self.actor.eval() # to disable batchnorm
          state = T.tensor([observation['infmap']],dtype=T.float32).to(mydevice)
          state_sky = T.tensor([observation['metadata']],dtype=T.float32).to(mydevice)
          mu = self.actor.forward(state,state_sky).to(mydevice)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise,size=(self.n_actions,)),
                                 dtype=T.float).to(mydevice)
        mu_prime = T.clamp(mu_prime,self.min_action,self.max_action)
        self.time_step +=1

        return mu_prime.cpu().detach().numpy()


    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return

        
        if not self.prioritized:
          state, action, reward, new_state, done = \
                                 self.replaymem.sample_buffer(self.batch_size)
        else:
          state, action, reward, new_state, done, idxs, is_weights = \
                                self.replaymem.sample_buffer(self.batch_size)
 
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

        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        target_actions = self.target_actor.forward(new_state_batch,new_state_batch_sky)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action,
                                self.max_action)

        q1_ = self.target_critic_1.forward(new_state_batch, target_actions, new_state_batch_sky)
        q2_ = self.target_critic_2.forward(new_state_batch, target_actions, new_state_batch_sky)
        q1_[terminal_batch] = 0.0
        q2_[terminal_batch] = 0.0
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)
        critic_value_ = T.min(q1_, q2_)

        target = reward_batch + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        # update priorities in the replay buffer
        if self.prioritized:
          q1=self.critic_1.forward(state_batch,action_batch,state_batch_sky)
          errors1=T.abs(q1-target).detach().cpu().numpy()
          q2=self.critic_2.forward(state_batch,action_batch,state_batch_sky)
          errors2=T.abs(q2-target).detach().cpu().numpy()
          self.replaymem.batch_update(idxs,0.5*(errors1+errors2))

        self.critic_1.train()
        self.critic_2.train()

        q1 = self.critic_1.forward(state_batch, action_batch, state_batch_sky)
        q2 = self.critic_2.forward(state_batch, action_batch, state_batch_sky)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        if not self.prioritized:
          q1_loss = F.mse_loss(target, q1)
          q2_loss = F.mse_loss(target, q2)
        else:
          q1_loss = self.replaymem.mse(target, q1, is_weight)
          q2_loss = self.replaymem.mse(target, q2, is_weight)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval == 0:
          self.actor.train()
          self.actor.optimizer.zero_grad()
          actor_q1_loss = self.critic_1.forward(state_batch, self.actor.forward(state_batch,  state_batch_sky), state_batch_sky)
          actor_loss = -T.mean(actor_q1_loss)
          actor_loss.backward()
          self.actor.optimizer.step()

          self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        self.replaymem.save_checkpoint()


    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.target_actor.eval()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self.update_network_parameters(tau=1.)

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
#             max_mem_size=4096, input_dims=[1,128,128], lr_a=0.001, lr_c=0.001)
