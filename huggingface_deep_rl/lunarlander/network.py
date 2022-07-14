
from tensorflow.keras import models, Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf 
import numpy as np
import random
import os 
import shutil 
import tqdm 
import collections 
import statistics
from typing import Tuple 



class ReplayBuffer:

    def __init__(self,N:int):
        self.memory = collections.deque(maxlen = N)
        self.cache = [] 

    def insert_single(self,trajectory : Tuple):
        assert(len(trajectory)  ==  6) # trajectory must be of form (s,a,r,sp,v,d)
        self.memory.append(trajectory)
        self.cache.append(trajectory)
    
    def get_minibatch(self,N,numpy=True):
        states = []
        actions = []
        rewards = [] 
        s_primes = [] 
        values = [] 
        terminated = [] 

        # priority = lambda x:abs(x[4] - (x[2] + gamma*tf.reduce_max(model(np.expand_dims(x[3],axis=0)),axis=0)*(1-x[5]) ))  
        # sorted_memory = sorted(self.memory,key = priority)
        sorted_memory = self.memory
        #choosing the examples with the highest error, as those will be the ones the model can learn the most from
        for i in range(N):
            s,a,r,sp,v,d = sorted_memory[random.randrange(len(self.memory))] #random.randrange(len(self.memory))
            states.append(s)
            actions.append([a])
            rewards.append([r])
            s_primes.append(sp)
            values.append([v])
            terminated.append([d])
        return [np.array(x) if numpy else x for x in [states,actions,rewards,s_primes,values,terminated]]
    
    def size(self):
        return len(self.memory)

        
        
        

        






class QNetwork:

    def __init__(self,
        num_actions: int,
        num_state_inputs: int, 
        model_name='lunarlanderV0',
        MODEL_DIR = 'model', 
        params = {  
                    'layer_dims': [64,128,64],
                    'lr': 1e-4,
                    'eps': 1,
                    'eps_decay': 0.995, 
                    'min_eps': 0.02,
                    'gamma': 0.99,
                    'tau': 0.001,
                    'batch_size': 64,
                    'num_episodes': 10000,
                    'episode_time_steps': 999,
                    'buffer_size': int(1e5) ,
                    'update_step': 4},
        ):
        tf.keras.utils.set_random_seed(42)
        self.num_actions = num_actions
        self.num_state_inputs = num_state_inputs
        self.params = params 
        self.MODEL_DIR = MODEL_DIR
        self.model_name = model_name 
        self.dirpath = os.path.join(self.MODEL_DIR,self.model_name)
        self.optimizer = Adam(learning_rate = self.params['lr'])
        self.loss_fn = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        #self.loss_fn = self.compute_loss

        self.model = self.build_network(hidden_layer_dims = self.params['layer_dims'])
        self.target_model = self.build_network(hidden_layer_dims = self.params['layer_dims'])
        self.replay_buffer = ReplayBuffer(self.params['buffer_size'])

    def sync_models(self):
        target_model_weights = self.target_model.get_weights()
        model_weights = self.model.get_weights() 

        for i in range(len(target_model_weights)):
            target_model_weights[i] = (self.params['tau']*model_weights[i]) + ((1-self.params['tau'])*target_model_weights[i])
        self.target_model.set_weights(target_model_weights)


    def model_exists(self):
        return os.path.exists(self.dirpath) and os.path.isdir(self.dirpath)
    
    def save_model(self):
        if self.model_exists(): shutil.rmtree(self.dirpath)
        self.model.save(self.dirpath)
    

    def instantiate_model(self,hidden_layer_dims):
        model = Sequential() 
        for layer in hidden_layer_dims:
            model.add(Dense(layer))
        model.add(Dense(self.num_actions))
        model.compile(optimizer = self.optimizer, loss = self.loss_fn )
        model.build((None,self.num_state_inputs))
        return model
    
    def build_network(self,**kwargs):
        if self.model_exists():
            model = models.load_model(self.dirpath)
            return model
        return self.instantiate_model(**kwargs)
    
    def optimal_action_selection(self,state,target = False):
        model = self.target_model if target else self.model 
        action_values = model(np.expand_dims(state,axis=0))
        return np.argmax(action_values)
    


    def e_greedy_policy(self,state,learning=True,target=False):
        action_space = [a for a in range(self.num_actions)]

        if (np.random.rand() <= max(self.params['eps'],self.params['min_eps'])) and learning: 
            return random.choice(action_space) 
        return self.optimal_action_selection(state,target=target)

    @staticmethod
    def compute_loss(q_target,A):
        loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        assert A.shape[0] == q_target.shape[0]
        return loss(A,q_target)/A.shape[0]

    def run_episode(self,env,episode):
        s = env.reset()
        rewards = []
        ctr = 0 
        for t in range(self.params['episode_time_steps']):

            action = self.e_greedy_policy(s)
            sp,r,terminated,info = env.step(action)
            ap = self.e_greedy_policy(sp,target = True)
            v = tf.squeeze(tf.gather(self.model(np.expand_dims(s,axis=0)),action,axis=1))
            self.replay_buffer.insert_single((s,action,r,sp,v,terminated*1))
            rewards.append(r)
            ctr += 1
            s = sp                
            if terminated: 
                break 
        
        return np.sum(rewards), ctr
        



    def train(self,env):
        with tqdm.trange(self.params['num_episodes']) as t: 
            episode_rewards  = collections.deque(maxlen=100)
            total_timesteps = 0 
            for i in t:
                episode_reward,episode_len = self.run_episode(env,i)
                self.params['eps'] = self.params['eps']*self.params['eps_decay']
                episode_rewards.append(episode_reward)
                total_timesteps += episode_len
                if (self.replay_buffer.size() > self.params['batch_size']): # and (ctr % self.params['update_step'] == 0)
                    S,A,R,SP,V,D = self.replay_buffer.get_minibatch(self.params['batch_size'])
                    temp = tf.squeeze(self.model(np.expand_dims(S,axis=0)))
                    QP = self.target_model(np.expand_dims(SP,axis=0))
                    Y = tf.expand_dims(tf.reduce_max(tf.squeeze(R + (self.params['gamma'] *tf.math.multiply(tf.reduce_max(QP,axis=1),1-D))),axis=1),axis=1)
                    Yhat = tf.squeeze(self.model(np.expand_dims(S,axis=0)))
                    indices = np.array([i for i in range(A.shape[0])]).reshape((A.shape[0],1))
                    Yhat = tf.tensor_scatter_nd_update(Yhat,np.concatenate((indices,A),axis=1),tf.squeeze(Y.numpy()))
                    self.model.fit(S,Yhat)
                self.sync_models()
                if i % 10 == 0:
                    running_reward_avg = statistics.mean(episode_rewards)
                    print(f'Episode {i}: average reward: {running_reward_avg}')
                    print(f"total timesteps so far: {total_timesteps}")

        self.save_model()








if __name__ == "__main__":
    import gym
    env = gym.make('LunarLander-v2')
    #env = gym.make('MountainCar-v0')
    print('s',env.reset())
    nn = QNetwork(num_state_inputs = env.observation_space.shape[0],num_actions = env.action_space.n)
    nn.train(env)
