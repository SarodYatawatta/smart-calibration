import gym
import numpy as np
from demix_sac import DemixingAgent
from demixingenv import DemixingEnv
import pickle

K=6
Ninf=128
M=3*K+2
env = DemixingEnv(K=K,Nf=3,Ninf=128,Npix=1024,Tdelta=10,provide_hint=True)

# Trained agent, without hints
agent_0 = DemixingAgent(gamma=0.99, batch_size=256, n_actions=K-1, tau=0.005, max_mem_size=4096,
   input_dims=[1,Ninf,Ninf], M=M, lr_a=1e-3, lr_c=1e-3, alpha=0.03, name_prefix='./archive/nohint/', use_hint=False)
# Trained agent, with hints
agent_1 = DemixingAgent(gamma=0.99, batch_size=256, n_actions=K-1, tau=0.005, max_mem_size=4096,
   input_dims=[1,Ninf,Ninf], M=M, lr_a=1e-3, lr_c=1e-3, alpha=0.03, name_prefix='./archive/withhint/', use_hint=True)
# Untrained agent
agent_2 = DemixingAgent(gamma=0.99, batch_size=256, n_actions=K-1, tau=0.005, max_mem_size=4096,
   input_dims=[1,Ninf,Ninf], M=M, lr_a=1e-3, lr_c=1e-3, alpha=0.03, use_hint=False)

# Load models, no model loaded for untrained agent
agent_0.load_models_for_eval()
agent_1.load_models_for_eval()

n_games=100
n_steps=K

# storage to keep the results

# loop over episodes
for cn in range(n_games):
  observation = env.reset()
  # make copies
  observation_0=observation
  observation_1=observation.copy()
  observation_2=observation.copy()
  for ci in range(n_steps):

    action_0 = agent_0.choose_action(observation_0)
    observation_0_, reward_0, done, hint_0, info = env.step(action_0)
    action_1 = agent_1.choose_action(observation_1)
    observation_1_, reward_1, done, hint_1, info = env.step(action_1)
    action_2 = agent_2.choose_action(observation_2)
    observation_2_, reward_2, done, hint_2, info = env.step(action_2)

    # keep the action with highest reward for output
    if ci==0:
        action_recomm_0=action_0
        action_recomm_1=action_1
        action_recomm_2=action_2
        reward_recomm_0=reward_0
        reward_recomm_1=reward_1
        reward_recomm_2=reward_2
    else:
        if reward_recomm_0<reward_0:
          reward_recomm_0=reward_0
          action_recomm_0=action_0
        if reward_recomm_1<reward_1:
          reward_recomm_1=reward_1
          action_recomm_1=action_1
        if reward_recomm_2<reward_2:
          reward_recomm_2=reward_2
          action_recomm_2=action_2

    print(f'Iter {cn}:{ci} without/with hint/ notrain')
    print(action_0)
    print(reward_0)
    print(action_1)
    print(reward_1)
    print(action_2)
    print(reward_2)

    observation_0=observation_0_
    observation_1=observation_1_
    observation_2=observation_2_

  # also get the reward for the hint
  _, reward_hint, _, _, _= env.step(hint_1)

  print(f'Episode {cn}: actions')
  print(action_recomm_0)
  print(action_recomm_1)
  print(action_recomm_2)
  print(hint_1)
  print(f'Rewards {reward_recomm_0} {reward_recomm_1} {reward_recomm_2} {reward_hint}')
