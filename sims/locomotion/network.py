from typing import Tuple, List
from tensorflow.keras import Sequential, models,Model 
from tensorflow.keras.layers import Dense, Conv2D,Flatten, InputLayer, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import numpy as np 



class ActorCriticNetwork(Model):

    def __init__(
        self,
        env,
        policy_hidden_layers: List = [128],
        critic_hidden_layers: List =[128]
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

        self.env = env 
        sample_state = env.reset()
        sample_action = env.action_space.sample()
        assert len(sample_action.shape) == 1 #ensure that the action is a n X 1 vector. (n, ) tuple where n is the number of actions 
        self.state_input_layer = InputLayer(input_shape=sample_state.shape)
        self.policy_network_layers = [Dense(l,activation = 'tanh') for l in policy_hidden_layers ] #policy hidden layers
        self.policy_network_layers.append( Dense(sample_action.shape[0],activation = 'tanh')) # policy output layer for control problems is typically within action_limit*tanh()

        self.critic_network_layers = [Dense(l,activation = 'tanh') for l in critic_hidden_layers ] #critic hidden layers
        self.critic_network_layers.append(Dense(1,activation='linear')) #final critic output layer 

        #create duplicate critic network to serve as target network 
        self.target_critic_network_layers = [Dense(l,activation = 'tanh') for l in critic_hidden_layers ] 
        self.target_critic_network_layers.append(Dense(1,activation='linear')) 
    

    def get_policy_action(self,state: tf.Tensor) -> tf.Tensor: 
        A = state 
        for l in self.policy_network_layers:
            A = l(A)
        return A


    def get_critic_value(self,state_action: tf.Tensor,target: bool =False) -> tf.Tensor: 
        A = state_action 
        layers = self.target_critic_network_layers if target else self.critic_network_layers
        for l in layers:
            A = l(A)
        return A 
    

    

    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.
    def env_step(self,action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        state, reward, done, info = self.env.step(action)
        return (state.astype(np.float32), 
                np.array(reward, np.float32), 
                np.array(done, np.int32))


    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.int32])
    

    def run_episode(self,
    initial_state: tf.Tensor, 
    model: Model, 
    max_steps: int = 1000 ):
        action_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        Q_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    
        s = initial_state 
        for t in tf.range(max_steps):
            s = tf.expand_dims(s,0)
            a = self.get_policy_action(s)
            action_values = action_values.write(t,a)
            q = self.get_critic_value(tf.concat([s,a],0))
            Q_values = Q_values.write(t,tf.squeeze(q))

            s_prime,reward,terminated = self.tf_env_step(a) 

            rewards = rewards.write(t,tf.squeeze(reward))




            


