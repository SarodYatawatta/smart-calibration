import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from lbfgsnew import LBFGSNew # custom optimizer
import pickle # for saving replaymemory
from enet_sac import ReplayBuffer,PER

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
        self.checkpoint_file = os.path.join('./', name+'_td3_critic.model')

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
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.n_actions)
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
        self.checkpoint_file = os.path.join('./', name+'_td3_actor.model')

        self.to(self.device)

    def forward(self, x):
        x=F.elu(self.bn1(self.fc1(x)))
        x=F.elu(self.bn2(self.fc2(x)))
        x=F.elu(self.bn3(self.fc3(x)))
        actions=T.tanh(self.fc4(x)) # in [-1,1], scale up and shift as needed (in the environment)

        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


class Agent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions,
            max_mem_size=100, tau=0.001, update_actor_interval=2, warmup=1000, noise=0.1, prioritized=False, use_hint=False):
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
        self.prioritized=prioritized
        self.use_hint=use_hint
        self.admm_rho=0.1
        self.Nadmm=5

        if not self.prioritized:
         self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions) 
        else:
         self.replaymem=PER(max_mem_size, input_dims, n_actions) 
    
        # online nets
        self.actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, 
                name='a_eval')
        self.critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_eval_1')
        self.critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_eval_2')
        # target nets
        self.target_actor=ActorNetwork(lr_a, input_dims=input_dims, n_actions=n_actions, 
                name='a_target')
        self.target_critic_1=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_target_1')
        self.target_critic_2=CriticNetwork(lr_c, input_dims=input_dims, n_actions=n_actions, name='q_target_2')
        # noise fraction
        self.noise = noise

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        actor_state_dict = dict(actor_params)
        target_actor_params = self.target_actor.named_parameters()
        target_actor_dict = dict(target_actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)


        critic_params = self.critic_1.named_parameters()
        target_critic_params = self.target_critic_1.named_parameters()
        critic_state_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
        self.target_critic_1.load_state_dict(critic_state_dict)

        critic_params = self.critic_2.named_parameters()
        target_critic_params = self.target_critic_2.named_parameters()
        critic_state_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
        self.target_critic_2.load_state_dict(critic_state_dict)


    def store_transition(self, state, action, reward, state_, terminal, hint):
        if not self.prioritized:
          self.replaymem.store_transition(state,action,reward,state_,terminal,hint)
        else:
          # Set reward as initial priority (assuming positive reward), see:
          #   https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
          self.replaymem.store_transition(state,action,reward,state_,terminal,hint,reward)

    def choose_action(self, observation):
        if self.time_step<self.warmup:
          mu=T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(mydevice)
        else:
          self.actor.eval() # to disable batchnorm
          state = T.cat((observation['eig'],observation['A'])).to(mydevice)
          mu = self.actor.forward(state).to(mydevice)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise,size=(self.n_actions,)),
                                 dtype=T.float).to(mydevice)
        mu_prime = T.clamp(mu_prime,self.min_action,self.max_action)

        self.time_step +=1
        return mu_prime.cpu().detach().numpy()

    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return

        
        if not self.prioritized:
          state, action, reward, new_state, done, hint = \
                                 self.replaymem.sample_buffer(self.batch_size)
        else:
          state, action, reward, new_state, done, hint, idxs, is_weights = \
                                self.replaymem.sample_buffer(self.batch_size)

        state_batch = T.tensor(state).to(mydevice)
        new_state_batch = T.tensor(new_state).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice)
        terminal_batch = T.tensor(done).to(mydevice)
        hint_batch = T.tensor(hint).to(mydevice)

        if self.prioritized:
            is_weight=T.tensor(is_weights).to(mydevice)

        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        target_actions = self.target_actor.forward(new_state_batch)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action,
                                self.max_action)

        q1_ = self.target_critic_1.forward(new_state_batch, target_actions)
        q2_ = self.target_critic_2.forward(new_state_batch, target_actions)
        q1_[terminal_batch] = 0.0
        q2_[terminal_batch] = 0.0
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)
        critic_value_ = T.min(q1_, q2_)
        target = reward_batch + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        # update priorities in the replay buffer
        if self.prioritized:
          q1=self.critic_1.forward(state_batch,action_batch)
          errors1=T.abs(q1-target).detach().cpu().numpy()
          q2=self.critic_2.forward(state_batch,action_batch)
          errors2=T.abs(q2-target).detach().cpu().numpy()
          self.replaymem.batch_update(idxs,0.5*(errors1+errors2))

        self.critic_1.train()
        self.critic_2.train()

        def closure():
          if T.is_grad_enabled():
            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
          q1 = self.critic_1.forward(state_batch, action_batch)
          q2 = self.critic_2.forward(state_batch, action_batch)
          if self.prioritized:
            q1_loss = self.replaymem.mse(target, q1, is_weight)
            q2_loss = self.replaymem.mse(target, q2, is_weight)
          else:
            q1_loss = F.mse_loss(target, q1)
            q2_loss = F.mse_loss(target, q2)
          critic_loss = q1_loss + q2_loss
          if critic_loss.requires_grad:
            critic_loss.backward(retain_graph=True)

          return critic_loss

        self.critic_1.optimizer.step(closure)
        self.critic_2.optimizer.step(closure)


        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval == 0:
          self.actor.train()

          if not self.use_hint:
            self.actor.optimizer.zero_grad()
            actor_q1_loss = self.critic_1.forward(state_batch, self.actor.forward(state_batch))
            if not self.prioritized:
               actor_loss = -T.mean(actor_q1_loss)
            else:
               actor_loss = -T.mean(actor_q1_loss*is_weight)
            actor_loss.backward()
            self.actor.optimizer.step()
          else:
            lagrange_y=T.zeros(hint_batch.numel(),requires_grad=False).to(mydevice)
            for admm in range(self.Nadmm):
               self.actor.optimizer.zero_grad()
               actions=self.actor.forward(state_batch)
               actor_q1_loss = self.critic_1.forward(state_batch, actions)
               if not self.prioritized:
                 actor_loss = -T.mean(actor_q1_loss)
               else:
                 actor_loss = -T.mean(actor_q1_loss*is_weight)
               diff1=(actions-hint_batch).view(-1)
               if not self.prioritized:
                 loss1=(T.dot(lagrange_y,diff1)+self.admm_rho/2*F.mse_loss(actions,hint_batch)).mean()/actions.numel()
               else:
                 loss1=((T.dot(lagrange_y,diff1)+self.admm_rho/2*F.mse_loss(actions,hint_batch))*is_weight).mean()/actions.numel()
               actor_loss=actor_loss+loss1
               actor_loss.backward()
               self.actor.optimizer.step()
               with T.no_grad():
                  lagrange_y=lagrange_y+self.admm_rho*(actions-hint_batch).view(-1)

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
        self.actor.load_checkpoint_for_eval()
        self.critic_1.load_checkpoint_for_eval()
        self.critic_2.load_checkpoint_for_eval()
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def print(self):
        print(self.actor)
        print(self.critic_1)

#a=Agent(gamma=0.99, batch_size=32, n_actions=2, max_mem_size=1024, input_dims=[11],
#        lr_a=0.001, lr_c=0.001, prioritized=True, use_hint=True)
