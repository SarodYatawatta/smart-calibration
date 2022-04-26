import gym
import torch
import numpy as np
import pickle
from enetenv import ENetEnv
from enet_td3 import Agent

np.random.seed(0)

if __name__ == '__main__':
    N=20 # rows = data points
    M=20 # columns = parameters, note, if N<M, no unique solution
    env = ENetEnv(M,N)
    # actions: 2
    # prioritized=True for prioritized experience replay memory
    agent = Agent(gamma=0.99, batch_size=64, n_actions=2, tau=0.005,
                  max_mem_size=1024, input_dims=[N+N*M], lr_a=1e-3, lr_c=1e-3,
                 update_actor_interval=2, warmup=100, noise=0.1, prioritized=False)
    # note: input dims: N eigenvalues+ N*M size of design matrix, 
    # lr_a: learning rate actor, lr_c:learning rate critic
    scores=[]
    n_games= 1000
    
    # load from previous sessions
    #agent.load_models()
    #with open('scores.pkl','rb') as f:
    #   scores=pickle.load(f)

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        loop=0
        while (not done) and loop<4: # limit number of loops as well
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            loop+=1
            #env.render()
        score=score.cpu().data.item()/loop
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)
        if i%10==0:
          env.render()
          # save models to disk
          agent.save_models()

    with open('scores.pkl','wb') as f:
      pickle.dump(scores,f)
