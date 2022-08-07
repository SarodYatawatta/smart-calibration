import os,sys,argparse
import numpy as np
import torch
import torch.distributed.rpc as rpc
import time,threading
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef, rpc_async, remote
from demix_sac import ReplayBuffer, PER, ActorNetwork, DemixingAgent
from demixingenv import DemixingEnv

# unique name strings
LEARNER_NAME='learner'
ACTOR_NAME='actor{}'

################ Learner: 
# runs the loop: 
# sample from replay mem, learn(), update priority, send actor weights to remote actors
# - init networks
# loop: 
#  - sample batch from replaybuffer
#  - learn()
#  - update priorities of the batch
class Learner:
    def __init__(self,world_size,K=6,Ninf=128):
        self.K=K
        self.Ninf=Ninf
        M=3*K+2
        self.learner_rref=RRef(self)
        self.actor_rrefs=[]
        # create remote actors
        for ob_rank in range(1,world_size):
            ob_info=rpc.get_worker_info(ACTOR_NAME.format(ob_rank))
            print(f'Learner: got actor {ob_info}')
            self.actor_rrefs.append(remote(ob_info,Actor,args=([1,Ninf,Ninf],2**(K-1),K,Ninf)))

        # create local actor/critic networks and 
        # prioritized experience replay mem (PER)
        self.agent=DemixingAgent(gamma=0.99, batch_size=64, n_actions=2**(K-1), tau=0.005,
                  max_mem_size=1024, input_dims=[1,Ninf,Ninf], M=M, lr_a=1e-3, lr_c=1e-3, warmup=100, update_interval=10, prioritized=True)

        self.lock=threading.Lock()

    def download_replaybuffer(self,obs_id,replaybuffer):
          with self.lock:
            print(f'Learner: update replaybuffer from {obs_id}')
            for i in range(min(replaybuffer.mem_cntr,replaybuffer.mem_size)):
              state={'infmap':replaybuffer.state_memory_img[i],
                        'metadata':replaybuffer.state_memory_sky[i]}
              action=replaybuffer.action_memory[i]
              reward=replaybuffer.reward_memory[i]
              new_state={'infmap':replaybuffer.new_state_memory_img[i],
                        'metadata':replaybuffer.new_state_memory_sky[i]}
              done=replaybuffer.terminal_memory[i]
              self.agent.replaymem.store_transition_from_buffer(state,action,reward,new_state,done)
              self.agent.learn()

            print(f'Learner: replaybuffer size {self.agent.replaymem.mem_cntr}')

    # start all actors, and loop
    def run_episodes(self,max_episodes):
        # each iteration is one episode
        for episode in range(max_episodes):
          print(f'Learner: episode {episode}')
          futs=[]
          for obs_rref in self.actor_rrefs:
              futs.append(
                 rpc_async(
                   obs_rref.owner(),
                   obs_rref.rpc_sync().run_observations,
                   args=(self.learner_rref,),
                 )
              )
          for fut in futs:
            fut.wait()

          # uploading actor replaymemory to learner is 
          # initiated by the actors,
          # also the learn() step is called thereafter

          # also save models/replaybuffer
          if episode%10==0:
            self.agent.save_models()

    def get_actor_params(self):
        params=self.agent.actor.named_parameters()
        params_cpu=dict()
        # copy tensors to CPU before sending
        for name, p in params:
            params_cpu[name]=p.cpu()
        return params_cpu


################ Actor: 
# runs the loop:
# generate observation, get action, environment step(), store transition to buffer, upload buffer to learner, update actor from learners weights
# - init actor from learner
# - init env
# loop :
#  - get action 
#  - env.step()
#  - add state transition to local replaybuffer 
#  - if local replaybuffer is full, upload buffer to learner
#  - update actor from learner 
class Actor:
    def __init__(self, input_dims, n_actions, K=6, Ninf=128, max_mem_size=100):
        self.id=rpc.get_worker_info().id
        print(f'Actor: id {self.id}')
        # create env
        self.K=K
        self.Ninf=Ninf
        M=3*K+2
        self.env=DemixingEnv(K=self.K,Nf=3,Ninf=self.Ninf,Npix=1024,Tdelta=10)

        self.warmup=100 
        self.time_step=0
        self.n_actions=n_actions
        
        # create local actor
        self.actor=ActorNetwork(lr=1e-3,input_dims=input_dims, n_actions=n_actions, M=M, max_action=1, name='a_'+str(self.id)+'_eval')

        self.device=self.actor.device
        self.actor.eval()

        # create replay buffer, with smaller size
        self.replaymem=ReplayBuffer(max_mem_size,input_dims,n_actions,M)

    # one call is one episode
    def run_observations(self,learner_rref):

        print(f'Actor: updating actor from learner')
        fut=learner_rref.rpc_async().get_actor_params()
        params=fut.wait()
        self.actor.load_state_dict(params,strict=False)

        for epoch in range(10):
          observation=self.env.reset()
          # loop action, step, store
          done=False
          loop=0
          while (not done) and loop<10:
              # get action (deterministic)
              action_=self.choose_action(observation,evaluation_episode=True)
              # map action in 2^(K-1) to K-1 vector
              action=self.scalar_to_kvec(action_,self.K-1)
              # env.step
              observation_, reward, done, info = self.env.step(action)
              self.replaymem.store_transition(observation,action_,reward,observation_,done)
              observation=observation_
              print(f'Actor: {self.id} {epoch} {loop} {reward}')
              loop+=1

        # upload replaybuffer to learner, ideally when this buffer is full
        learner_rref.rpc_sync().download_replaybuffer(self.id,self.replaymem)
        # reset replaybuffer
        self.replaymem.mem_cntr=0


    def choose_action(self,observation,evaluation_episode=False):
        if self.time_step<self.warmup:
            self.time_step+=1
            action=np.random.choice(range(self.n_actions))
            return action

        state = torch.tensor(observation['infmap'].astype(np.float32),dtype=torch.float32).to(self.device)
        state = state[None,]
        state_sky = torch.tensor(observation['metadata'].astype(np.float32),dtype=torch.float32).to(self.device)
        state_sky = state_sky[None,]
        action_probabilities = self.actor.forward(state,state_sky)

        action_probs=action_probabilities.squeeze(0).cpu().detach().numpy()
        if evaluation_episode:
           action=np.argmax(action_probs)
        else:
           action=np.random.choice(range(self.n_actions),p=action_probs)
        self.time_step+=1

        return action

    # convert integer to binary bits, return array of size K
    def scalar_to_kvec(self,n,K=5):
        ll=[1 if digit=='1' else 0 for digit in bin(n)[2:]]
        a=np.zeros(K)
        a[-len(ll):]=ll
        return a

def run_process(rank,world_size,addr,port):
    os.environ['MASTER_ADDR']=addr
    os.environ['MASTER_PORT']=port

    options=rpc.TensorPipeRpcBackendOptions(
        rpc_timeout=0 # infinite timeout
    )
    if rank==0:
        print(f'invoking learner at {addr}:{port}')
        # Learner
        rpc.init_rpc(LEARNER_NAME,rank=rank,world_size=world_size,rpc_backend_options=options)
        learner=Learner(world_size,K=6,Ninf=128)
        
        # load from saved models
        learner.agent.load_models()

        # run main loop 
        learner.run_episodes(1000)
    else:
        print(f'invoking actor {rank}')
        # Actor (note that Actors are already created by the Learner init())
        rpc.init_rpc(ACTOR_NAME.format(rank),rank=rank,world_size=world_size,rpc_backend_options=options)

    rpc.shutdown()

if __name__=='__main__':
    parser=argparse.ArgumentParser(
            description='Demixing tuning with distributed PER soft actor critic',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--rank',default=0,type=int,metavar='r',
            help='rank of this process')
    parser.add_argument('--world-size',default=2,type=int,metavar='w',
            help='number of processes, one learner and actors')
    parser.add_argument('--learner-addr',default='localhost',type=str,metavar='l',
            help='learner (rank 0) address')
    parser.add_argument('--learner-port',default='59999',type=str,metavar='p',
            help='learner (rank 0) port')
    args=parser.parse_args()
    
    # spawn each process
    run_process(rank=args.rank,world_size=args.world_size,addr=args.learner_addr,
            port=args.learner_port)
