from alpaca_functions import AlpacaUtils
from trading_env import create_env
from stable_baselines3 import PPO,DQN,A2C
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

def main(): 
    alpacaHelper = AlpacaUtils()
    btc_df = alpacaHelper.build_env_df('2022-07-01','2022-09-01','day','BTC/USD',True)
    env = create_env(btc_df)
    # Parallel environments
    model = PPO("MlpPolicy", env, verbose=1,device='cpu') #seems to be some kind of parallelization error occurring when using 'mps' device
    vec_env = model.get_env()
    model.learn(total_timesteps=10_000_000)

    # Run an episode until it ends :
    done,truncated = False, False
    observation = vec_env.reset()
    while not done and not truncated:
        action, _states = model.predict(observation, deterministic=True)
        #Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        # action = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
        observation, reward, done, info = vec_env.step(action)
        print(reward)
    print(info)
    env.close()
    vec_env.close()



if __name__ == "__main__":
    main()
