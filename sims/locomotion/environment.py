
from email import policy
import numpy as np 
from typing import Tuple, List 
import tensorflow as tf 
import gym 
import collections
import tqdm
import statistics



class Environment: 

    def __init__(self,gym_env,*args,**kwargs):
        self.gym = gym.make(gym_env,**kwargs)
        self.sample_state = self.gym.reset()
        self.sample_action = self.gym.action_space.sample()
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


    def run_episode(self,policy,critic,max_steps: int,sync_targets: bool = False ):
        rewards = []
        losses = []
        s = tf.expand_dims(tf.convert_to_tensor(self.gym.reset()),0) 
        with tf.GradientTape(persistent=True) as tape:
            for t in range(max_steps):
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

                losses.append(critic_loss + actor_loss)
                critic_grads = tape.gradient(critic_loss,critic.trainable_variables)
                actor_grads = tape.gradient(actor_loss,policy.trainable_variables)

                critic.optimizer.apply_gradients(zip(critic_grads,critic.trainable_variables))
                policy.optimizer.apply_gradients(zip(actor_grads,policy.trainable_variables))
                if terminated: break 
            
        
        return np.sum(rewards),np.mean(losses)
    

    def train(self,actor,critic,max_steps_per_episode: int = 1000,max_episodes:int=1000,success_criterion: List = []):
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
                episode_reward,avg_loss = self.run_episode(actor,critic,max_steps_per_episode)
                episodes_reward.append(episode_reward)
                running_reward_avg = statistics.mean(episodes_reward)
                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward_avg)
                # Show average episode reward every 10 episodes
                if i % 2 == 0:
                    critic.sync_target_networks()
                    actor.sync_target_networks()
                if i % 10 == 0:
                    print(f'Episode {i}: average reward: {running_reward_avg}')
                    sample_s = tf.expand_dims(tf.convert_to_tensor(self.sample_state),0) 
                    a = self.action_scalar*actor(sample_s)
                    c = critic(sample_s,a)
                    print('critic ',c)
                    print('action ',a)

                print('avf loss', avg_loss)

                if (len(success_criterion) == 2) and running_reward_avg > success_criterion[0] and i >= success_criterion[1]:  
                    break

        print(f'\nSolved at episode {i}: average reward: {running_reward_avg:.2f}!')
        actor.save_model()
        critic.save_model()

    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        #state, reward, done, info = self.environment.step(action)
        #return [tf.convert_to_tensor(x,dtype=tf.float32) for x in [state,reward,done]]
        state,reward,done = tf.numpy_function(self.environment_step, [action], [tf.float32, tf.float32, tf.int32])
        return state,reward,done