import numpy as np 
import math 
import random
import matplotlib.pyplot as plt

class Bandit:


	def __init__(self,arms=10,stationary=True,reward_mean = 0, reward_variance = 1):
		self.is_stationary = stationary
		self.arms = arms 
		self.reward_mean = reward_mean
		self.reward_var = reward_variance
		self.q_a = np.random.normal(loc=reward_mean,scale=math.sqrt(reward_variance),size=(arms,))
		self.t = 0 
		self.n_t = np.zeros((arms,))
	
		
	def crank_arm(self,arm):
		reward = self.q_a[arm]
		self.t += 1
		self.n_t[arm] += 1
		return reward 



class LearningAlgorithm:


	def __init__(self,bandit,initial_values="realistic",action_method="eps-greedy",eps = 0.1,c=2):
		self.n = bandit.arms
		self.initial_val_policy = initial_values
		self.action_sel_policy = action_method
		if action_method == "eps-greedy":
			self.epsilon = eps
		if action_method == "UCB":
			self.c = c
		self.N_t = np.zeros((bandit.arms,))	
		self.k = 0 
		self.instantiate_Q()
		self.avg_reward = [0]


	def select_optimal_action(self):
		if self.action_sel_policy == "eps-greedy":
			i = random.uniform(0,1)
			return random.randint(0,self.n-1) if i <= self.epsilon else np.argmax(self.Q_a)
		
		if self.action_sel_policy == "UCB":
			metric = self.Q_a + self.c*np.sqrt(np.log(self.k)/self.N_t) if self.k > 0 else self.Q_a
			return np.argmax(metric)
		

	def instantiate_Q(self):
		if self.initial_val_policy == "realistic":
			self.Q_a = np.zeros((self.n,))
		elif self.initial_val_policy == "optimistic":
			self.Q_a = np.zeros((n,)) + 5 # the constant added for optimistic needs to be adjusted depending on the bandit's distribution, admittedly the learning alg doesn't know the dist but might need to give it just to speed things along 
			



	
	def update_Q(self,arm,reward):
		self.Q_a[arm] += (reward - self.Q_a[arm])/self.k
		return self.Q_a[arm]
	

	def forward_propagate(self,bandit,num_trials = 1):
		for _ in range(num_trials):
			action = self.select_optimal_action()
			reward = bandit.crank_arm(action)
			self.k += 1
			self.N_t[action] += 1
			self.update_Q(action,reward) 
			new_avg = ((self.k-1)*self.avg_reward[-1] + reward)/self.k
			self.avg_reward.append(new_avg)
		return self.avg_reward


if __name__ == "__main__":
	bandit = Bandit()
	alg = LearningAlgorithm(bandit)

	print(bandit.arms)
	r = alg.forward_propagate(bandit,num_trials = 1000)
	plt.plot(r)
	plt.show()





			
