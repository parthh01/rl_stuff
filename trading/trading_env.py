
import gymnasium as gym
import gym_trading_env
import numpy as np 
import random
from stable_baselines3.common.env_util import make_vec_env

#PARAMS:
INIT_VAL = 10000
SHORT_INTEREST = 0.0003/100
def adjusted_sharpe_ratio(rx,rf,sd,min_sharpe=1,clip = 5): # todo: switch to sortino ratio
    return min(clip,max(-clip,((rx-rf)/(1+sd)) - min_sharpe)) # 1+ sd to avoid divide by zero, adding clipping here to avoid exploding gradients 

def custom_reward_function(history):
    rx = (history['portfolio_valuation',-1] - history['portfolio_valuation',0])/(history['step',-1]+1)
    rf = (history['data_spy_close',-1] - history['data_spy_close',0])/(history['step',-1]+1)
    sd = np.std(history['portfolio_valuation'])
    return adjusted_sharpe_ratio(rx,rf,sd)

def create_env(df):
    env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions =  [i for i in np.arange(0,1.1,0.1)],#[ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        # trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= SHORT_INTEREST, # 0.0003% per timestep (one timestep = 1h here)
        portfolio_initial_value=INIT_VAL,
        reward_function= custom_reward_function,
        dynamic_feature_functions = []
    )
    return env

def build_feature_arr(row,last_position=1.0,last_value=1.0):
    features = [row[c] for c in row.keys() if c.startswith('feature_')]
    arr = np.array(features)
    # arr = np.append(arr,[last_position,last_value]) # THIS ORDER MATTERS: TAKEN FROM https://gym-trading-env.readthedocs.io/en/latest/environment_desc.html
    return arr


def test_env(env):
    done,truncated = False, False
    observation,info = env.reset()
    print('action space',env.action_space)

    while not done and not truncated:
        #Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        action = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
        observation, reward, done,truncated, info = env.step(action)
        # print('reward',reward)
    print(info)
    env.close()


if __name__ == "__main__":
    from alpaca_functions import AlpacaUtils
    ah = AlpacaUtils()
    btc_df = ah.build_env_df('2022-07-01','2022-09-01','day','BTC/USD',True) 
    env = create_env(btc_df)
    print(env.action_space)
    obs = env.reset()
    row = btc_df.iloc[0].to_dict()
    features = build_feature_arr(row)
    print(row)
    print('observation: ',obs[0])
    print('features: ',features)
    print('is equivalent: ', obs[0] == features)
    #env = gym.make('CartPole-v1')
    test_env(env)

