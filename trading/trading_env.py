import gymnasium as gym
import gym_trading_env


def run_env(df):
    env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        # borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )
    # Run an episode until it ends :
    done, truncated = False, False
    observation, info = env.reset()
    while not done and not truncated:
        #Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
        observation, reward, done, truncated, info = env.step(position_index)

    print(info)
