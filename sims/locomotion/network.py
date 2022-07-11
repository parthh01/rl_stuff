from typing import Tuple, List
from tensorflow.keras import models,Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import numpy as np 
import tqdm
import collections 
import statistics


class BaseNetworkClass(Model):

    def __init__(self,*args,**kwargs):
        super().__init__()
        tf.config.run_functions_eagerly(True)
    

    def sync_target_networks(self):
        for i in range(len(self.target_network)):
            self.target_network[i].set_weights(self.network[i].get_weights())
        

class ActorNetwork(BaseNetworkClass):

    def __init__(self,state_input_shape,num_actions, hidden_layers: List = [128],params = {'lr': 0.01}):
        super().__init__()
        self.params = params 
        #self.state_input_layer = InputLayer(input_shape=sample_state.shape)
        self.network = self.build_network(num_actions,hidden_layers)
        self.target_network = self.build_network(num_actions,hidden_layers)
        self.optimizer = Adam(learning_rate=self.params['lr'])
        self.sync_target_networks()

    
    def build_network(self,num_actions,hidden_layers: List = [128]):
        layers = [] #tf.keras.Input(shape=state_input_shape,name='state_input')
        for l in hidden_layers:
            layers.append(Dense(l,activation = 'tanh'))
        layers.append(Dense(num_actions,activation='tanh'))
        return layers



    def call(self,state: tf.Tensor,target: bool = False) -> tf.Tensor: 
        layers = self.target_network if target else self.network
        A = layers[0](state)
        for l in layers[1:]:
            A = l(A)
        return A 
    

    def episode_update(self,S,A,R,S_P,D): 
        pass 
    


class CriticNetwork(BaseNetworkClass):

    def __init__(
        self,
        state_input_shape, 
        num_actions,
        hidden_layers: List =[128],
        params = {
            'gamma': 0.98,
            'lr': 0.01
        }
        ):
        """
        actor/policy network will take the current state as an input, and output a tensor representing the action. 
        critic network takes the current state and action from the policy network to produce the Q value of the (s,a) pair
        critic network loss (r + gamma*critic(s_prime)) computed against the TARGET CRITIC (cached previous version of critic network to stabilize training)
        policy network loss is the (-ve) of the critic value from the critic network (AFTER critic has just been backpropped) 
        policy network is backpropped according to this loss 

        considerations: 
        polyak averaging for target network update? 

        """

        super().__init__()
        

        self.params = params 
        self.state_input = tf.keras.Input(shape=state_input_shape,name='state_input')
        self.action_input = tf.keras.Input(shape=(num_actions,),name='action_input')
        self.network = self.build_network(hidden_layers)
        self.target_network = self.build_network(hidden_layers)
        self.sync_target_networks()

        self.optimizer = Adam(learning_rate=self.params['lr'])
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    
    def build_network(self,hidden_layers):
        layers = [Concatenate(axis=-1)] 
        for l in hidden_layers:
            layers.append(Dense(l,activation = 'tanh'))
        layers.append(Dense(1))
        return layers

    

    def call(self,state: tf.Tensor,action: tf.Tensor,target:bool = False) -> tf.Tensor:
        layers = self.target_network if target else self.network
        A = layers[0]([state,action]) 
        for l in layers[1:]:
            A = l(A)
        return A 
    
    
        

    # def run_episode(self,initial_state,actor,max_steps: int):
    #     states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     action_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     critic_q_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     target_q_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    
    #     s = tf.constant(self.environment.reset(),dtype=tf.float32)
    #     s_shape = s.shape
    #     for t in tf.range(max_steps):
    #         s = tf.expand_dims(s,0)
    #         a,q = self(s)
    #         a = a * self.environment.action_space.high # so this is because output of the policy network is scaled in the range
    #         # [-1,1] per action space so multiplying it back to the original space limits will give the appropriate action. 
    #         # assumption here is the environment action space range is centered around 0 and symmetrical.
    #         states = states.write(t,s)
    #         action_values = action_values.write(t,a)
    #         critic_q_values = critic_q_values.write(t,tf.squeeze(q))

    #         s,reward,terminated = self.tf_env_step(a) #really s here is s_prime
    #         #s = s.set_shape(s_shape)
    #         a_prime,q_prime = self(tf.expand_dims(s,0),target=True)
    #         target_q_value = reward if terminated else reward + (self.params['gamma']*q_prime)
    #         target_q_values = target_q_values.write(t,tf.squeeze(target_q_value))
    #         rewards = rewards.write(t,reward)
            

    #         if tf.cast(terminated,tf.bool):
    #             break 

            
    #     for cache in [states,action_values,critic_q_values,target_q_values,rewards]:
    #         cache.stack()
        
    #     return states,action_values,critic_q_values,target_q_values,rewards

    # @staticmethod
    # def compute_critic_loss(pred_q_values: tf.Tensor,target_q_values: tf.Tensor) -> tf.Tensor:
    #     huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    #     critic_loss = huber_loss(target_q_values,pred_q_values)
    #     return critic_loss 
    
    # @staticmethod
    # def compute_actor_loss(pred_q_values: tf.Tensor) -> tf.Tensor: 
    #     return tf.math.reduce_sum(tf.math.scalar_mul(-1,pred_q_values)) #to keep the loss a minimizing function 
    
    # @staticmethod
    # def compute_combined_loss(pred_q_values: tf.Tensor,target_q_values: tf.Tensor) -> tf.Tensor:
    #     huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    #     critic_loss = huber_loss(target_q_values,pred_q_values)
    #     actor_loss = -tf.math.reduce_sum(pred_q_values)
    #     return critic_loss + actor_loss

    

    # @tf.function
    # def train_step(self,max_steps:int):
    #     with tf.GradientTape() as tape: 
    #         states,action_values,critic_q_values,target_q_values,rewards = self.run_episode(max_steps=max_steps)
    #         states.mark_used()
    #         action_values.mark_used()

            
    #         critic_q_values,target_q_values = [tf.expand_dims(x.concat(),1) for x in [critic_q_values,target_q_values] ] # Convert training data to appropriate TF tensor shapes
    #         print([v for v in self.trainable_variables if v.name.startswith('actor')])
    #         combined_loss = tf.reshape(self.compute_combined_loss(critic_q_values,target_q_values),(1,1))
    #     grads = tape.gradient(combined_loss,[v for v in self.trainable_variables if v.name.startswith('actor')],unconnected_gradients=tf.UnconnectedGradients.ZERO) # here i think i can split the loss to the different policy and actor layers by naming 
    #     # the original variables when creating the layers  and then scoping them

    #     self.optimizer.apply_gradients(zip(grads,[v for v in self.trainable_variables if v.name.startswith('actor')]))
    #     #self.optimizer.minimize(combined_loss,var_list = self.trainable_variables,tape=tape)

    #     episode_reward = tf.math.reduce_sum(rewards)
    #     #self.sync_target_networks()
    #     return episode_reward
    

    # def learn(self,max_steps_per_episode: int = 1000,max_episodes:int=10000,success_criterion: List = []):
    #     """
    #     in this function the success criterion should be a list:
    #     [0] = a desired average reward 
    #     [1] =  minimum number of episodes 
    #     over which this reward should be achieved. 
    #     """
    #     # Keep last episodes reward
    #     episodes_reward  = collections.deque(maxlen=success_criterion[1] if len(success_criterion) == 2 else max_episodes)


    #     with tqdm.trange(max_episodes) as t:
    #         for i in t:
    #             episode_reward = float(self.train_step(max_steps_per_episode))
    #             episodes_reward.append(episode_reward)
    #             running_reward_avg = statistics.mean(episodes_reward)
    #             t.set_description(f'Episode {i}')
    #             t.set_postfix(episode_reward=episode_reward, running_reward=running_reward_avg)
    #             # Show average episode reward every 10 episodes
    #             if i % 10 == 0:
    #                 pass # print(f'Episode {i}: average reward: {avg_reward}')

    #             if (len(success_criterion) == 2) and running_reward_avg > success_criterion[0] and i >= success_criterion[1]:  
    #                 break

    #     print(f'\nSolved at episode {i}: average reward: {running_reward_avg:.2f}!')

            






            




            


