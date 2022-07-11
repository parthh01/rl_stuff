
import numpy as np 
from typing import Tuple, List 
import tensorflow as tf 
import gym 



class Environment: 

    def __init__(self,gym_env):
        self.gym = gym.make(gym_env)
        self.num_actions = self.gym.action_space.sample().shape[0]
        self.action_scalar = self.gym.action_space.high
        self.state_shape = self.gym.reset().shape


    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.
    def environment_step(self,action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        n = action.shape[1]
        state, reward, done, info = self.gym.step(action.T.reshape(n))
        return (state.astype(np.float32), 
                np.array(reward, np.float32), 
                np.array(done, np.int32))


    def run_episode(self,policy,critic,max_steps: int = 1000,tensor=True):
        rewards = []
        s = tf.expand_dims(tf.convert_to_tensor(self.gym.reset()),0) 
        for t in range(max_steps):
            with tf.GradientTape(persistent=True) as tape:
                a = self.action_scalar*policy(s)
                q = critic(s,a)
                s,r,terminated,info = self.gym.step(a.numpy()[0])
                rewards.append(r)
                s = tf.expand_dims(tf.convert_to_tensor(s),0) 
                if terminated: 
                    target_q = tf.constant(r)
                else: 
                    a_prime = self.action_scalar*policy(s,target=True)
                    q_prime = critic(s,a_prime,target=True) # s here is actually s_prime
                    target_q = tf.math.add(tf.math.scalar_mul(critic.params['gamma'],q_prime),r)

                critic_loss = critic.loss_fn(q,target_q)
                actor_loss = tf.scalar_mul(-1,q)
            
            critic_grads = tape.gradient(critic_loss,critic.trainable_variables)
            actor_grads = tape.gradient(critic_loss,policy.trainable_variables)
            critic.optimizer.apply_gradients(zip(critic_grads,critic.trainable_variables))
            policy.optimizer.apply_gradients(zip(actor_grads,policy.trainable_variables))
            if terminated: break 
        
        critic.sync_target_networks()
        policy.sync_target_networks()


        print('done')
        return np.mean(rewards)

                

                
                

            





    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        #state, reward, done, info = self.environment.step(action)
        #return [tf.convert_to_tensor(x,dtype=tf.float32) for x in [state,reward,done]]
        state,reward,done = tf.numpy_function(self.environment_step, [action], [tf.float32, tf.float32, tf.int32])
        return state,reward,done