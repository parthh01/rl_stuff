
from os import environ
import gym
from pidcontroller import PIDController 
import numpy as np 
import math 

class RLController: 

    def __init__(self,q_values,action_space,x_step,v_step):
        self.q_values = q_values 
        self.action_space = action_space
        self.x_step = x_step
        self.v_step = v_step
    

    def action_signal(self,state):
        x,v = state
        x = round(self.x_step * round(x/self.x_step),2)
        v = round(self.v_step * round(v/self.v_step),2)
        action = np.argmax([self.q_values[(x,v,a)] for a in self.action_space]) # works because action space will be indexed same as the list index
        return action 



class Environment: 

    def __init__(self,sim):
        self.sim = sim 
        self.env = gym.make(sim)
        self.discrete_action_space = hasattr(self.env.action_space,'n')
        self.terminated = False
    
    def continuous_to_discrete(self,controller,u):
        #convert continuous signal from controller into discrete action space signal. 
        lower,upper = controller.control_signal_bounds 
        upper = upper - lower 
        u = u - lower 
        lower = 0 
        c = (upper / self.env.action_space.n) 
        return int(min(self.env.action_space.n -1 , u // c))  # - 1 since discrete actions are 0 indexed



    def run_simulation(self,controller = None,controller_discrete = True):
        observation, info = self.env.reset(return_info=True)
        for _ in range(1000):
            if not self.terminated: 
                u = controller.action_signal(observation) if controller else 2
                u = self.continuous_to_discrete(controller,u) if controller and controller_discrete else u 
                observation, reward, self.terminated, info = self.env.step(u if self.discrete_action_space else [u]) #env.action_space.sample()
                self.env.render()
                print(observation,u)

        print('simulation terminated')
        print(self.terminated)



def main():
    #environment = Environment('MountainCarContinuous-v0')
    environment = Environment('MountainCar-v0')
    #controller = PIDController()
    print(environment.discrete_action_space)
    with open('data.json') as f:
        q_values = eval(f.read())
    controller = RLController(q_values,list(range(environment.env.action_space.n)),0.05,0.01)
    environment.run_simulation(controller,controller_discrete=False)
    #environment.run_simulation()
    
    


if __name__ == "__main__":
    main()