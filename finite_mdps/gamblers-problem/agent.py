"""
Chapter 4 example 4.3 from Sutton Barto RL book 

Gambler's problem: make bets on the outcomes of a series of coin flips. reward is 1:1 for heads. 
game ends when you reach a desired sum ($100) from starting capital ($1)

states: gambler's capital {1,2,3,....n} where n is terminal state 
actions: amount to stake {1,2,3,...c} where c is amount of capital currently owned 

"""
import numpy as np 
import matplotlib.pyplot as plt 




class BettingAgent: 


    def __init__(self,DESIRED_SUM = 100, STARTING_CAPITAL = 1,PROB_HEADS = 0.4,eps = 1e-25,gamma = 0.99):
        self.target = DESIRED_SUM
        self.capital = STARTING_CAPITAL
        self.V = np.zeros((self.target+1,))
        self.V[-1] = 1 # set the value for the target state to be 1 
        self.eps = eps  
        self.p_h = PROB_HEADS
        self.R = {0: -1, 1: 1}
        self.gamma = gamma
        self.iteration_ctr = 0 
        self.pi_star = {}

    def get_action_space(self,s):
        """
        action space is [1,2,3,...c] where c is the current capital. 
        upper bounded by the different between target and state, ie if c = 99 and target = 100,
        then no point betting more than t - c 
        """
        return [i+1 for i in range(s) if i+1 <= self.target - s ] 

    def get_state_action_s_primes(self,s,a):
        """
        returns the list of states you can reach, and the probabilities of those states for a given state-action pair, 
        and the reward at each state
        returns: [(p,r1',s1'),(1-p,r2',s2')]
        """
        s1 = min(self.target,s + a)
        s2 = max(0,s - a)
        return [(self.p_h, 0 ,s1),(1 - self.p_h, 0 ,s2)]

    def coin_flip(self):
        """
        returns True if heads else False 
        """
        return np.random.rand() <= self.p_h
    


    def get_maximal_value_action(self,s):
        """
        evaluate all the actions to return the action that produces the maximal value, and the maximal value itself.
        """
        actions = self.get_action_space(s)
        action_values = [] 
        for a in actions:
            triples = self.get_state_action_s_primes(s,a)
            expected_value = 0 
            for p,r,s_prime in triples:
                if s_prime in [self.target,0]:
                    payoff = 1 if s_prime == self.target else 0 
                else:
                    payoff = r + (self.gamma*self.V[s_prime])
                expected_value += p*(payoff) 
            action_values.append((a,expected_value))
        return max(action_values,key = lambda x: x[1])


    def determine_optimal_policy(self):
        """
        populates optimal policy self.pi_star, and returns a list containing state and value estimate at that state
        """
        value_estimates = []
        for s in range(1,len(self.V)-1):
            a_star,v_star = self.get_maximal_value_action(s)
            self.pi_star[s] = a_star
            value_estimates.append((s,v_star))
        
        return value_estimates


    def perform_value_iteration(self):
        grad = 2*self.eps
        while (grad > self.eps):
            grads = []
            self.iteration_ctr += 1 
            for s in range(1,len(self.V)):
                if s == self.target:
                    continue
                current_v = self.V[s]
                a,updated_v = self.get_maximal_value_action(s)
                self.V[s] = updated_v
                grads.append(abs(current_v - updated_v))
            grad = max(grads)
                
        print(f"finished value iteration after {self.iteration_ctr} sweeps")
        return 1 




if __name__ == "__main__":
    agent = BettingAgent(PROB_HEADS = 0.4)
    agent.perform_value_iteration()
    estimates = agent.determine_optimal_policy()
    optimal_policy = agent.pi_star
    graph_policy = [(k,optimal_policy[k]) for k in optimal_policy.keys()] 
    plt.figure("STATE VALUE ESTIMATES")
    plt.bar(*zip(*estimates))
    plt.figure("POLICY (BET SIZE) AT each state")
    plt.bar(*zip(*graph_policy))
    plt.show()


