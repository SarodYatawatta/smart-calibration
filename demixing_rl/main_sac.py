import gym
from demix_sac import DemixingAgent
import numpy as np
from demixingenv import DemixingEnv
import pickle

if __name__ == '__main__':
    # directions in order: CasA,CygA,HerA,TauA,VirA and target
    K=6 # directions: last is the target direction, the rest are outlier sources
    # input dimensions, determined by
    Ninf=128 # influence map Ninf x Ninf
    # metadata = (separation,azimuth,elevation) K + (lowest)frequency + n_stations= 3K+2
    M=3*K+2
    provide_hint=True # to enable generation of hint from env
    env = DemixingEnv(K=K,Nf=3,Ninf=128,Npix=1024,Tdelta=10,provide_hint=provide_hint)
    # number of actions = K-1 for the K-1 outlier directions
    agent = DemixingAgent(gamma=0.99, batch_size=64, n_actions=K-1, tau=0.005, max_mem_size=4096,
                  input_dims=[1,Ninf,Ninf], M=M, lr_a=1e-3, lr_c=1e-3, alpha=0.03, use_hint=provide_hint)
    scores=[]
    n_games = 100

    total_steps=0
    warmup_steps=400
    
    # load from disk DQN, replaymem
    #agent.load_models()
    #with open('scores.pkl','rb') as f:
    #    scores=pickle.load(f)

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
              agent.store_transition(observation, action, reward, 
                                    observation_, done, hint)
            else:
              observation_, reward, done, info = env.step(action)
              agent.store_transition(observation, action, reward, 
                                    observation_, done, np.zeros(K-1))

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
