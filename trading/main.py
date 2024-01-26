from alpaca_functions import AlpacaUtils
from trading_env import create_env, build_feature_arr
from stable_baselines3 import PPO,DQN,A2C
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
import os 
import numpy as np
import pandas as pd 
from datetime import datetime, timedelta


def run_live(model,symbol='BTC/USD',interval='min',action_space = [round(i,4) for i in np.arange(0,1.1,0.1)]):
    alpacaHelper = AlpacaUtils()
    today_date = datetime.now()
    result_date = today_date - timedelta(days=1)
    start_date = result_date.strftime("%Y-%m-%d")
    df = alpacaHelper.get_market_data(start=start_date,interval=interval,symbols=symbol,crypto=True).reset_index()
    rows = df.to_dict('records')
    #apparently its faster to have a maintain a list of rows, convert to df everytime we need to get the ta indicators.
    while True:
        new_bar = dict(alpacaHelper.get_latest_bar(symbol))
        while rows[-1]['timestamp'] >= new_bar['timestamp']:
           new_bar = dict(alpacaHelper.get_latest_bar(symbol))
        rows.pop(0)
        rows.append(new_bar)
        print('current bar: ',rows[-1]['timestamp']) 
        df = pd.DataFrame(rows)
        ta_df = alpacaHelper.generate_features(df,spy=False)
        obs_row = ta_df.iloc[-1].to_dict()
        features = build_feature_arr(obs_row)
        action_idx = model.predict([features])[0]
        action = action_space[action_idx[0]]
        print('order size: ',action)
        order = alpacaHelper.size_to_order(action,symbol.replace('/',''))
        






def main(): 
    alpacaHelper = AlpacaUtils()
    btc_df = alpacaHelper.build_env_df('2024-01-01','2024-01-10','min','BTC/USD',True)
    env = create_env(btc_df)
    #Parallel environments
    model_path = 'ppo_tradingEnv.zip'
    if os.path.exists(model_path):
        model = PPO.load(model_path,env,verbose=1,device='cpu')
    else:
        model = PPO("MlpPolicy", env,verbose=1,device='cpu') #seems to be some kind of parallelization error occurring when using 'mps' device
    vec_env = model.get_env()
    model.learn(total_timesteps=10_000)
    # Run an episode until it ends :
    done,truncated = True, False
    info = None
    observation = vec_env.reset()
    model.save("ppo_tradingEnv")
    while not done and not truncated:
        action, _states = model.predict(observation, deterministic=True)
        print('action: ',[round(i,4) for i in np.arange(0,1.1,0.1)][action[0]])
        #Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        # action = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
        observation, reward, done, info = vec_env.step(action)
        print('reward',reward)
    print(info)
    env.close()
    vec_env.close()

## psuedo code for live trading 
    # model = PPO.load(model_name)
    # action_space = [round(i,4) for i in np.arange(0,1.1,0.1)]
    # while true:
    #     bars  = get_market_data(start_date) #get last n bars needed to fill features
    #     row = bars.iloc[-1]
    #     X = build_feature_arr(row) # input for NN 
    #     action =model.predict(X)
    #     btc_size = action_space[action]
    #     market_data = size_to_order(btc_size)
    #     send_market_order(market_data)

    

    

if __name__ == "__main__":
    # main()
    model = PPO.load('ppo_tradingEnv.zip')
    run_live(model)