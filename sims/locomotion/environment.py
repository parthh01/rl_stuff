
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
    

    def generate_episode_replay_buffer(self,policy,critic,max_steps: int):
        S = []
        A = [] 
        R = [] 
        SP = []
        d = [] 
        S = self.gym.reset()
        for t in range(max_steps):
            pass




    def run_episode(self,policy,critic,max_steps: int,sync_targets: bool = False ):
        rewards = []
        critic_losses = []
        actor_losses = [] 
        pred_qs = [] #tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) 
        target_qs = [] #tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        critic_grads = [] 
        actor_grads = [] 

        s = tf.expand_dims(tf.convert_to_tensor(self.gym.reset()),0) 
        with tf.GradientTape(persistent=True) as tape:
            for t in range(max_steps):
                a = self.action_scalar*policy(s)
                q = critic(s,a)
                q_loss = critic(s,a,target=True)
                s,r,terminated,info = self.gym.step(a.numpy()[0])
                rewards.append(r)
                s = tf.expand_dims(tf.convert_to_tensor(s),0) 
                if terminated: 
                    target_q = tf.constant(r)
                else: 
                    a_prime = policy(s,target=True) # dont believe we need to multiply by action scalar here 
                    q_prime = critic(s,a_prime,target=True) # s here is actually s_prime
                    #target_q = tf.math.add(tf.math.scalar_mul(critic.params['gamma'],q_prime),r)
                    target_q = (critic.params['gamma']*q_prime) + r
                
                pred_qs.append(tf.cast(tf.squeeze(q),tf.float32)) #pred_qs.write(t,tf.squeeze(q))
                target_qs.append(tf.cast(tf.squeeze(target_q),tf.float32))  #target_qs.write(t,tf.squeeze(target_q))
                if terminated: break

                #print(policy.summary())
                critic_loss = critic.loss_fn(q,target_q)
                actor_loss =  -q   #-q #tf.scalar_mul(-1,q)      
                critic_losses.append(critic_loss)  
                actor_losses.append(actor_loss)    
                critic_grad = tape.gradient(critic_loss,critic.trainable_variables)
                actor_grad = tape.gradient(actor_loss, policy.trainable_variables) 
                critic_grads.append(critic_grad)
                actor_grads.append(actor_grad)
                critic.optimizer.apply_gradients(zip(critic_grad,critic.trainable_variables))
                policy.optimizer.apply_gradients(zip(actor_grad,policy.trainable_variables))
            
            mean_critic_grads = []
            mean_actor_grads = [] 
            for grad,mean in [(critic_grads,mean_critic_grads),(actor_grads,mean_actor_grads)]:
                for i in range(len(grad[0])):
                    grads = []
                    for j in range(len(grad)):
                        grads.append(grad[j][i])
                    mean.append(tf.reduce_mean(np.array(grads),axis=0))

            # critic.optimizer.apply_gradients(zip(mean_critic_grads,critic.trainable_variables))
            # policy.optimizer.apply_gradients(zip(mean_actor_grads,policy.trainable_variables))
                 
            
        
        return np.sum(rewards),np.mean(critic_losses),np.mean(actor_losses)
    

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
                episode_reward,avg_critic_loss,avg_actor_loss = self.run_episode(actor,critic,max_steps_per_episode)
                episodes_reward.append(episode_reward)
                running_reward_avg = statistics.mean(episodes_reward)
                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward_avg)
                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    critic.sync_target_networks()
                    actor.sync_target_networks()
                    print(f'Episode {i}: average reward: {running_reward_avg}')
                    sample_s = tf.expand_dims(tf.convert_to_tensor(self.sample_state),0) 
                    a = self.action_scalar*actor(sample_s)
                    c = critic(sample_s,a)
                    print('critic ',c)
                    print('action ',a)

                print('avg critic loss', avg_critic_loss)
                print('avg actor loss', avg_actor_loss )

                if (len(success_criterion) == 2) and running_reward_avg > success_criterion[0] and i >= success_criterion[1]:  
                    break

        actor.save_model()
        critic.save_model()
        print(f'\nSolved at episode {i}: average reward: {running_reward_avg:.2f}!')
    

    def run(self,actor,critic,max_steps: int = 999):
        s = self.gym.reset()
        s = tf.expand_dims(tf.convert_to_tensor(s),0)
        terminated = False 
        for t in range(max_steps):
            a = self.action_scalar*actor.model.predict(s)
            s,r,terminated,info = self.gym.step(a[0])
            s = tf.expand_dims(tf.convert_to_tensor(s),0)
            a = tf.convert_to_tensor(a)
            q = critic.model(s,a)
            self.gym.render()
            if terminated: 
                break 
            print('THE REWARD IS: ',r)
            print('THE ESTIMATED Q IS: ',q)

        
        print('terminated')




    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        #state, reward, done, info = self.environment.step(action)
        #return [tf.convert_to_tensor(x,dtype=tf.float32) for x in [state,reward,done]]
        state,reward,done = tf.numpy_function(self.environment_step, [action], [tf.float32, tf.float32, tf.int32])
        return state,reward,done