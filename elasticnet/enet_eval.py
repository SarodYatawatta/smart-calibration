import gym
from enet_dqn import Agent
import numpy as np
from enetenv import ENetEnv
import matplotlib.pyplot as plt
import pickle

# load a pre-trained model and evaluate with new observations
if __name__ == '__main__':
    M=20
    N=25
    env = ENetEnv(M,N)
    # actions: 2,
    agent = Agent(gamma=0.99, batch_size=64, n_actions=2,
                  max_mem_size=1000, input_dims=[N], lr_a=0.001, lr_c=0.001) # most settings are not relevant because we only evaluate 
    scores = []
    n_games = 1
    
    # load from previous sessions
    agent.load_models_for_eval()
   
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        env.initsol()
        env.render()
        loop=0
        while (not done) and loop<10: # limit number of loops as well
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action,keepnoise=True)
            score += reward
            observation = observation_
            loop+=1
            print('loop %d reward %f'%(loop,reward))
        env.render()
        scores.append(score.cpu().numpy())
