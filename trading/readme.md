
setup: 
 - tbd 


plan: 
 - build an environment that can be used to train/test a variety of rl agents. 
	- parameters: 
		- crypto/securites 
		- increment 
		- backtest/livetrading 
		- single security/multiple
	- features:
		- quant indicators, 
 - train rl agents on them 


 todos: 
  - build a far more representative state (both for a single timestep, and how to build a state for multiple timesteps)
  - investigate rl Transformers 
  - create generate df pipeline 
  - rework reward function to use sortino ratio
  - setup live trading env 
