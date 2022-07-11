import gym 
import numpy as np 
from tensorflow.keras import models
import tensorflow as tf 
import os 

class Agent: 

    def __init__(
        self,
        env,
        params = {
            'eps': 0.1, 
            'gamma': 0.9, 
            'lambda': 0.5,
            'alpha': 0.01
        },
        MODEL_DIR = 'models',
        model_name = None):
        self.network_input_shape = self.generate_input(env.gym.reset(),env.gym.action_space.sample())
        self.params = params
        self.value_network = models.load_model(os.path.join(MODEL_DIR,model_name)) if model_name else self.build_model()


    @staticmethod
    def generate_input(self,state,action):
        return np.concatenate((state,action),axis=0) 
    


    def build_model(self):
        pass 
    


def main():
    """
    flow: 
    PolicyNetwork(state) -> action 
    action,state -> CriticNetwork(state,action) -> Q Value 
    Environment(action) -> reward, state_prime 
    if state_prime is not Terminal: 
        PolicyNetwork(state_prime) -> action_prime
        PolicyNetwork(state_prime,action_prime) -> q_prime
        target_q = reward  + gamma*q_prime
    Else: 
        target_q = reward
    Critic Loss = mse(q_value,target_q)
    CriticNetworkBackprop(Critic Loss) -> DCritic Loss/daction
    Actor Loss = - Q Value 
    ActorNetworkBackprop(DCritic Loss/daction) 
    

    """

    from network import ActorNetwork,CriticNetwork
    from environment import Environment
    
    #env = Environment('Humanoid-v4')
    #env = Environment('MountainCarContinuous-v0')
    env = Environment('Pendulum-v1', g=9.81)
    policy = ActorNetwork('actor_v0',env.num_actions)
    critic = CriticNetwork('critic_v0',env.state_shape,env.num_actions)
    #g = env.run_episode(policy,critic)
    env.train(policy,critic,success_criterion=[50,10])


    print('done')




if __name__ == "__main__":
    main()
