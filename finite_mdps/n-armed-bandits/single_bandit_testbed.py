from components import * 
import numpy as np 
import matplotlib.pyplot as plt


def run_test(tasks=2000,steps = 1000):
	avg_reward = []
	for _ in range(tasks):
		b = Bandit()
		alg = LearningAlgorithm(b,action_method = 'UCB')
		reward = alg.forward_propagate(b,num_trials = steps)
		avg_reward.append(reward)

	R = np.stack(avg_reward,axis=0)
	R_avg = np.mean(R,axis=0)
	print(R_avg.shape)
	plt.plot(R_avg)
	plt.show()





if __name__ == "__main__":
	run_test()
