
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from mountaincar import Environment
import os 
import json 




class RLdrivingAgent: 
    """
    RL agent learns to drive the mountain car using Q learning policy algorithm. 
    Using discrete action space example just to make life easier. 
    it saves the Q values after learning in data.json, 
    and then the controller simply picks the action with max Q-value for a given state
    
    """

    @staticmethod
    def custom_round(x,base):
        return round(base * round(x/base),2)

    def __init__(self,x_space = [-1.2,0.5],v_space= [-0.07,0.07],action_space = [0,1,2],eps = 0.05,gamma = 0.9,alpha = 0.5):
        #the key that corresponds to a state-action pair will be (x,v,a) where x (position),v (velocity) 
        # are the two signals comprising the state, and a is the signal corresponding to the action. 
        # 35 possible X positions, 15 possible V positions, 3 possible actions = 1680 possible state-action pairs 
        self.params = {
            'eps': eps,
            'gamma': gamma, 
            'alpha': alpha } 
        self.x_step = 0.05
        self.v_step = 0.01
        self.Q_s_a = {}
        self.x_space = x_space 
        self.v_space = v_space 
        self.action_space = action_space
        self.training_history = []
        self.training_avg = []

        # initialize Q_s_a 
        for x in np.arange(x_space[0],x_space[1] + self.x_step,self.x_step):
            for v in np.arange(v_space[0],v_space[1] ,self.v_step):
                for a in action_space:
                    self.Q_s_a[(self.custom_round(x,self.x_step),self.custom_round(v,self.v_step),a)] =  100 if x == self.x_space[1] else 0

                    
    
    

    def e_greedy_policy(self,s,eps):
        """
        policy will pick the greedy action according to Q (1-eps)% of the time. 
        eps% of the time it will choose a random action.
        """
        x,v = s 
        q = [self.Q_s_a[(self.custom_round(x,self.x_step),self.custom_round(v,self.v_step),a)] for a in self.action_space]
        greedy_action = np.argmax(q)
        rng = random.random()
        return random.choice(self.action_space) if rng <= eps else greedy_action
    

    def run_q_learning(self,env,episodes = 1e5,max_episode_length = 1000):
        
        
        for episode in range(int(episodes)):
            terminated = False 
            observation, info = env.env.reset(return_info=True)
            x,v = observation
            for t in range(max_episode_length):
                if terminated: 
                    self.training_history.append((episode,t))
                    #if t < 200: print('ended with reward', t, r)
                    break
                
                else:
                    a = self.e_greedy_policy(observation,eps = self.params['eps'])
                    s_prime, r, terminated, info = env.env.step(a)
                    x_prime,v_prime = s_prime
                    if x_prime >= self.x_space[1]:
                        x_prime = self.x_space[1] # if the car is past the checkpoint it should count as the same checkpoint state
                        r = 100  #for some reason sometimes the program allows the car to overrun slightly so doesnt give these terminal states the right reward
                    a_prime = self.e_greedy_policy((x_prime,v_prime),eps = 0) # this is the difference between q learning and sarsa 
                    self.Q_s_a[(self.custom_round(x,self.x_step),self.custom_round(v,self.v_step),a)] += self.params['alpha']*(r + (self.params['gamma']*(self.Q_s_a[self.custom_round(x_prime,self.x_step),self.custom_round(v_prime,self.v_step),a_prime])) - self.Q_s_a[(self.custom_round(x,self.x_step),self.custom_round(v,self.v_step),a)] )
                    x,v = x_prime,v_prime 
                    observation = [x_prime,v_prime]
                    if x_prime >= self.x_space[1]: terminated = True #for some reason sometimes the program allows the car to overrun slightly 

            if not terminated:
                self.training_history.append((episode,max_episode_length))
            
            if episode % (episodes/100) == 0: 
                print(f"completed {episode+1} episodes")
                self.training_avg.append((episode,np.mean([l for _,l in self.training_history[-int(episodes/100):]])))
                print(f"current avg episode length: {self.training_avg[-1]}")
            
        print('finished training agent')

        
        if os.path.exists('data.json'): os.remove('data.json')
        with open('data.json', 'w') as fp:
            fp.write(str(self.Q_s_a))


    
    
    

    def show_training_history(self,avg=True):
        plt.plot([x[0] for x in self.training_avg], [x[1] for x in self.training_avg])
        l = [ (k,self.Q_s_a[k]) for k in self.Q_s_a if self.Q_s_a[k] != 0 ]
        print(len(l))
        plt.show()
    




                    









if __name__ == "__main__":
    environment = Environment('MountainCar-v0')
    agent = RLdrivingAgent()
    agent.run_q_learning(environment)
    agent.show_training_history()

    

