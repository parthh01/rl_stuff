import gym 
import numpy as np 
from tensorflow.keras import models
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
    from network import ActorCriticNetwork


    env = gym.make('Humanoid-v4')
    s,info = env.reset(return_info=True)
    terminated = False 
    for _ in range(1000):
        if terminated: 
            break
        s_prime, r,terminated, info = env.step(env.action_space.sample())
        env.render()
        # print(env.action_space.sample().shape)
        # print(s.shape)
        # print(r)
        # print(np.concatenate((s,env.action_space.sample()),axis=0).shape)
    
    nn = ActorCriticNetwork(env)
    print(nn.summary())


    print('done')




if __name__ == "__main__":
    main()
