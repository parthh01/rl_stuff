import numpy as np 
import pandas as pd

from trading_env import create_env
from stable_baselines3 import PPO 
from main import run_model
from alpaca_functions import AlpacaUtils

def k_fold_cross_validation(dataset,k=10,device='cpu'):
    dataset_chunks = np.array_split(dataset,k)
    for i in range(k):
        holdout = dataset_chunks[i]
        print(holdout)
        train = pd.concat([dataset_chunks[x] for x in range(k) if x != i])
        train_env = create_env(train)
        test_env = create_env(holdout)
        model = PPO("MlpPolicy", train_env,verbose=1,device=device)
        model.learn(total_timesteps=15_00)
        run_model(model,test_env)
        train_env.close()
        test_env.close()

if __name__ == "__main__":
    alpacaHelper = AlpacaUtils()
    df = alpacaHelper.build_env_df(start='2023-01-01',interval='min',symbols='BTC/USD',crypto=True)
    k_fold_cross_validation(df)
