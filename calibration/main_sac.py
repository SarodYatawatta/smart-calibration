import gym
from calib_sac import Agent
import numpy as np
from calibenv import CalibEnv
import pickle

if __name__ == '__main__':
    M=10 # maximum number of directions, one sky model component per direction
    # the actual number of direction in each episode will be K, M>=K
    # Note: K will be randomly generated from [2,M] in each episode

    provide_hint=True
    env = CalibEnv(M,provide_hint=provide_hint)
    # number of actions = 2*K for the K directions (spectral and spatial)
    agent = Agent(gamma=0.99, batch_size=32, n_actions=2*M, tau=0.005, max_mem_size=10000,
                  input_dims=[1,128,128], M=M, lr_a=1e-3, lr_c=1e-3, 
                  reward_scale=M, alpha=0.03, 
                  hint_threshold=0.01, admm_rho=1.0, use_hint=provide_hint)
    scores=[]
    n_games = 30
    
    total_steps=0
    warmup_steps=100 # warmup before using agent
    # load from disk DQN, replaymem
    #agent.load_models()
    #with open('scores.pkl','rb') as f:
    #    scores=pickle.load(f)


    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        loop=0
        while (not done) and loop<4 :
            if total_steps < warmup_steps:
                action=env.action_space.sample()
                action=action.squeeze(-1)
            else:
                action = agent.choose_action(observation)

            if provide_hint:
              observation_, reward, done, hint, info = env.step(action)
            else:
              observation_, reward, done, info = env.step(action)
              hint=np.zeros(2*M)

            agent.store_transition(observation, action, reward, 
                                    observation_, done, hint)
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
