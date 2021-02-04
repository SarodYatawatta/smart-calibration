import gym
from calib_ddpg import Agent
import numpy as np
from calibenv import CalibEnv
import pickle

if __name__ == '__main__':
    K=4 # directions: first is the target direction, the rest are outlier sources
    M=4 # sky model (for calibration) components, M>K , when each direction has multiple sources
    env = CalibEnv(K,M)
    # number of actions K for the K directions
    agent = Agent(gamma=0.99, batch_size=32, n_actions=K, tau=0.001,
                  input_dims=[1,128,128], K=K, M=M, lr_a=1e-4, lr_c=1e-3, max_mem_size=2000)
    scores=[]
    n_games = 30
    
    # load from disk DQN, replaymem
    #agent.load_models()
    #with open('scores.pkl','rb') as f:
    #    scores=pickle.load(f)

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        loop=0
        while (not done) and loop<10 :
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            loop+=1
        score=score/loop
        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)

        # after each game, save DQN, replaymem to disk
        agent.save_models()

    with open('scores.pkl','wb') as f:
      pickle.dump(scores,f)
