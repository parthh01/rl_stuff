
import gym
from pidcontroller import PIDController 

class Environment: 

    def __init__(self,sim):
        self.sim = sim 
        self.env = gym.make(sim)
        self.discrete_action_space = hasattr(self.env.action_space,'n')
        self.terminated = False
    

    def run_simulation(self,controller):
        observation, info = self.env.reset(return_info=True)
        for _ in range(1000):
            if not self.terminated: 
                u = controller.action_signal(observation)
                observation, reward, self.terminated, info = self.env.step([u]) #env.action_space.sample()
                self.env.render()
        
        print('simulation terminated')



def main():
    environment = Environment('MountainCarContinuous-v0')
    controller = PIDController()
    print(environment.discrete_action_space)
    environment.run_simulation(controller)
    
    


if __name__ == "__main__":
    main()