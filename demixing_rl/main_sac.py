import gymnasium as gym
from demix_sac import DemixingAgent
import numpy as np
from demixingenv import DemixingEnv
import pickle

import torch
import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser(
      description='Determine optimal settings in calibration, directions and max. iterations',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seed',default=0,type=int,metavar='s',
       help='random seed to use')
    parser.add_argument('--use_hint', action='store_true',default=False,
       help='use hint or not')
    parser.add_argument('--load', action='store_true',default=False,
       help='load model')
    parser.add_argument('--iteration', default=1000, type=int, help='max episodes')
    parser.add_argument('--warmup', default=30, type=int, help='warmup episodes')


    args=parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # directions in order: CasA,CygA,HerA,TauA,VirA and target
    K=6 # directions: last is the target direction, the rest are outlier sources
    # input dimensions, determined by
    Ninf=128 # influence map Ninf x Ninf
    # metadata = (separation,azimuth,elevation) K + (lowest)frequency + n_stations= 3K+2
    M=3*K+2
    provide_hint=args.use_hint # to enable generation of hint from env
    env = DemixingEnv(K=K,Nf=3,Ninf=128,Npix=1024,Tdelta=10,provide_hint=provide_hint)
    # number of actions = (K-1 for the K-1 outlier directions) + (1 for ADMM iterations)
    agent = DemixingAgent(gamma=0.99, batch_size=256, n_actions=K-1+1, tau=0.005, max_mem_size=16000,
                  input_dims=[1,Ninf,Ninf], M=M, lr_a=3e-4, lr_c=1e-3, alpha=0.03, hint_threshold=0.01, admm_rho=1.0, use_hint=provide_hint)
    scores=[]
    n_games = args.iteration

    total_steps=0
    warmup_steps=args.warmup*7 # during warmup, random actions are taken
    
    # load from disk networks, replaymem
    if args.load:
      agent.load_models()
      # load from disk replaymem
      agent.load_replaybuffer()
      with open('scores.pkl','rb') as f:
         scores=pickle.load(f)

    for i in range(n_games):
        score = 0
        done = False

        # observation: influence/residual sep,az,el,freq
        observation = env.reset()
        loop=0
        while (not done) and loop<7:
            if total_steps<warmup_steps:
              action = env.action_space.sample()
              action = action.squeeze(-1)
            else:
              action = agent.choose_action(observation)

            if provide_hint:
              observation_, reward, done, hint, info = env.step(action)
              scaled_reward = reward *10 if reward>0 else reward
              agent.store_transition(observation, action, scaled_reward,
                                    observation_, done, hint)
            else:
              observation_, reward, done, info = env.step(action)
              scaled_reward = reward *10 if reward>0 else reward
              agent.store_transition(observation, action, scaled_reward,
                                    observation_, done, np.zeros(K-1+1))

            print(action)
            if provide_hint:
              print(hint)
            score += reward
            agent.learn()
            observation = observation_
            loop+=1
            total_steps+=1
        score=score/loop
        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)

        # after each game, save DQN, replaymem to disk
        agent.save_models()
        with open('scores.pkl','wb') as f:
          pickle.dump(scores,f)
