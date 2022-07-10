from typing import Tuple, List
from tensorflow.keras import models,Model 
from tensorflow.keras.layers import Dense, InputLayer 
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import numpy as np 
import tqdm
import collections 
import statistics


class ActorCriticNetwork(Model):

    def __init__(
        self,
        env,
        policy_hidden_layers: List = [128],
        critic_hidden_layers: List =[128],
        params = {
            'gamma': 0.95,
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
        assert tf.executing_eagerly()

        self.env = env 
        self.params = params 
        sample_state = env.reset()
        sample_action = env.action_space.sample()
        assert len(sample_action.shape) == 1 #ensure that the action is a n X 1 vector. (n, ) tuple where n is the number of actions 
        #self.state_input_layer = InputLayer(input_shape=sample_state.shape)
        self.policy_network_layers = [Dense(l,activation = 'tanh') for l in policy_hidden_layers ] #policy hidden layers
        self.policy_network_layers.append( Dense(sample_action.shape[0],activation = 'tanh')) # policy output layer for control problems is typically within action_limit*tanh()
        #create duplicate critic network to serve as target network 
        self.target_policy_network_layers = [Dense(l,activation = 'tanh',trainable=False) for l in policy_hidden_layers ] #policy hidden layers
        self.target_policy_network_layers.append( Dense(sample_action.shape[0],activation = 'tanh',trainable=False)) # policy output layer for control problems is typically within action_limit*tanh()

        self.critic_network_layers = [Dense(l,activation = 'tanh') for l in critic_hidden_layers ] #critic hidden layers
        self.critic_network_layers.append(Dense(1,activation='linear')) #final critic output layer 

        #create duplicate critic network to serve as target network 
        self.target_critic_network_layers = [Dense(l,activation = 'tanh',trainable=False) for l in critic_hidden_layers ] 
        self.target_critic_network_layers.append(Dense(1,activation='linear',trainable=False)) 

        self.sync_target_networks()

        self.optimizer = Adam(learning_rate=self.params['lr'])
    

    def get_policy_action(self,state: tf.Tensor,target: bool = False) -> tf.Tensor: 
        A = state 
        layers = self.target_policy_network_layers if target else self.policy_network_layers
        for l in layers:
            A = l(A)
        return A


    def get_critic_value(self,state_action: tf.Tensor,target: bool =False) -> tf.Tensor: 
        A = state_action 
        layers = self.target_critic_network_layers if target else self.critic_network_layers
        for l in layers:
            A = l(A)
        return A 
    
    def call(self,state: tf.Tensor,target:bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        action = self.get_policy_action(state,target=target)
        state = tf.cast(state,tf.float32) # not sure how/why this becomes a float64 but turning it back to 32 bit
        state_action = tf.concat([state,action],1)
        value = self.get_critic_value(state_action,target=target)
        return action,value
    
    def sync_target_networks(self):
        for network,target in [(self.critic_network_layers,self.target_critic_network_layers),(self.policy_network_layers,self.target_policy_network_layers)]:
            for i in range(len(network)):
                target[i].set_weights(network[i].get_weights())
        


    

    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.
    def env_step(self,action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        state, reward, done, info = self.env.step(action)
        return (state.astype(np.float32), 
                np.array(reward, np.float32), 
                np.array(done, np.int32))


    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        state, reward, done, info = self.env.step(action)
        #return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.int32])
        return [tf.convert_to_tensor(x,dtype=tf.float32) for x in [state,reward,done]]


    

    def run_episode(self,max_steps: int):
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        action_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        critic_q_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        target_q_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    
        s = tf.constant(self.env.reset(),dtype=tf.float32)
        s_shape = s.shape
        for t in tf.range(max_steps):
            s = tf.expand_dims(s,0)
            a,q = self.call(s)
            a = a * self.env.action_space.high # so this is because output of the policy network is scaled in the range
            # [-1,1] so multiplying it back to the original space limits will give the appropriate action. 
            # only assumption here is the environment action space range is centered around 0 and symmetrical. 
            states = states.write(t,s)
            action_values = action_values.write(t,a)
            critic_q_values = critic_q_values.write(t,tf.squeeze(q))

            s,reward,terminated = self.tf_env_step(a) #really s here is s_prime
            a_prime,q_prime = self.call(s)
            target_q_value = reward if terminated else reward + (self.params['gamma']*q_prime)
            target_q_values = target_q_values.write(t,tf.squeeze(target_q_value))
            rewards = rewards.write(t,reward)
            s = s.set_shape(s_shape)

            if tf.cast(terminated,tf.bool):
                break 

            
        for cache in [states,action_values,critic_q_values,target_q_values,rewards]:
            cache.stack()
        
        return states,action_values,critic_q_values,target_q_values,rewards

    @staticmethod
    def compute_critic_loss(pred_q_values: tf.Tensor,target_q_values: tf.Tensor) -> tf.Tensor:
        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        critic_loss = huber_loss(target_q_values,pred_q_values)
        return critic_loss 
    
    @staticmethod
    def compute_actor_loss(pred_q_values: tf.Tensor) -> tf.Tensor: 
        return tf.constant(-1)*pred_q_values #to keep the loss a minimizing function 
    

    @tf.function
    def train_step(self,max_steps:int):
        with tf.GradientTape() as tape: 
            states,action_values,critic_q_values,target_q_values,rewards = self.run_episode(max_steps=max_steps)

            for x in [critic_q_values,target_q_values]: # Convert training data to appropriate TF tensor shapes
                x = tf.expand_dims(x,1)

            critic_loss = self.compute_critic_loss(critic_q_values,target_q_values)
            actor_loss = self.compute_actor_loss(critic_q_values)

            grads = tape.gradient(critic_loss + actor_loss,self.trainable_variables) # here i think i can split the loss to the different policy and actor layers by naming 
            # the original variables when creating the layers  and then scoping them

        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)
        #self.sync_target_networks()
        return episode_reward
    

    def learn(self,max_steps_per_episode: int = 1000,max_episodes:int=10000,success_criterion: List = []):
        """
        in this function the success criterion should be a list:
        [0] = a desired average reward 
        [1] =  minimum number of episodes 
        over which this reward should be achieved. 
        """
        # Keep last episodes reward
        episodes_reward  = collections.deque(maxlen=success_criterion[1] if len(success_criterion) == 2 else max_episodes)


        with tqdm.trange(max_episodes) as t:
            for i in t:
                episode_reward = float(self.train_step(max_steps_per_episode))
                episodes_reward.append(episode_reward)
                running_reward_avg = statistics.mean(episodes_reward)
                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward_avg)
                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    pass # print(f'Episode {i}: average reward: {avg_reward}')

                if (len(success_criterion) == 2) and running_reward_avg > success_criterion[0] and i >= success_criterion[1]:  
                    break

        print(f'\nSolved at episode {i}: average reward: {running_reward_avg:.2f}!')

            






            




            


