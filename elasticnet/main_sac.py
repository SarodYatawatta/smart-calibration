import gymnasium as gym
import torch
import numpy as np
import pickle
from enetenv import ENetEnv
from enet_sac import Agent

import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser(
      description='Elastic net regression hyperparamter tuning',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seed',default=0,type=int,metavar='s',
       help='random seed to use')
    parser.add_argument('--episodes',default=1000,type=int,metavar='g',
       help='number of episodes')
    parser.add_argument('--steps',default=5,type=int,metavar='t',
       help='number of steps per episode')
    parser.add_argument('--use_hint', action='store_true',default=False,
       help='use hint or not')
    args=parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    N=20 # rows = data points
    M=20 # columns = parameters, note, if N<M, no unique solution
    provide_hint=args.use_hint # to enable generation of hint from env
    env = ENetEnv(M,N,provide_hint=provide_hint)
    # actions: 2
    # prioritized=True for prioritized experience replay memory
    agent = Agent(gamma=0.99, batch_size=64, n_actions=2, tau=0.005,
                  max_mem_size=1024, input_dims=[N+N*M], lr_a=1e-3, lr_c=1e-3,
                 reward_scale=N, alpha=0.03, prioritized=False, use_hint=provide_hint)
    # note: input dims: N eigenvalues+ N*M size of design matrix, 
    # lr_a: learning rate actor, lr_c:learning rate critic
    scores=[]
    n_games= args.episodes
    
    # load from previous sessions
    #agent.load_models()
    #with open('scores.pkl','rb') as f:
    #   scores=pickle.load(f)

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        loop=0
        while (not done) and loop<args.steps: # limit number of loops as well
            action = agent.choose_action(observation)
            if provide_hint:
               observation_, reward, done, hint, info = env.step(action)
               agent.store_transition(observation, action, reward, 
                                    observation_, done, hint)
            else:
               observation_, reward, done, info = env.step(action)
               agent.store_transition(observation, action, reward, 
                                    observation_, done, np.zeros_like(action))
            score += reward
            agent.learn()
            observation = observation_
            loop+=1
            #env.render()
        score=score.cpu().data.item()/loop
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)
        if i%500==0:
          #env.render()
          # save models to disk
          agent.save_models()

    with open('scores.pkl','wb') as f:
      pickle.dump(scores,f)
