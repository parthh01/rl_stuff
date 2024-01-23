from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from dotenv import load_dotenv
import os
from ta import add_all_ta_features
from ta.utils import dropna
import pandas as pd
# BASEDIR = os.path.abspath(os.path.dirname(__file__))
# load_dotenv(os.path.join(BASEDIR, '../.env')) 

class AlpacaUtils:
       def __init__(self):
              # no keys required for crypto data
              self.crypto_client = CryptoHistoricalDataClient()
              self.stock_client = StockHistoricalDataClient(os.environ.get('APCA_API_KEY_ID'),  os.environ.get('APCA_API_SECRET_KEY'))
              self.interval_map = {'day': TimeFrame.Day, 'min': TimeFrame.Minute, 'hour': TimeFrame.Hour, 'week': TimeFrame.Week, 'month': TimeFrame.Month}

       def str_to_datetime(self,date:str):
              y,m,d = [int(i) for i in date.split('-')]
              return datetime(y,m,d)

       def generate_features(self,df,start,end,interval):
              spy_data = self.get_market_data(start,end,interval,["SPY"],crypto=False)
              spy_data.rename(columns={c: 'spy_' + c for c in spy_data.columns},inplace=True)
              df = pd.merge_asof(df,spy_data[["spy_close"]],left_index=True,right_index=True)
              df = add_all_ta_features(df,'open','high','low','close','volume',colprefix='feature_',fillna=True)
              #df.rename({c:'feature_'+c for c in ['vwap','open']},inplace=True)
              return df 
        
       def get_market_data(self,start: str,end: str,interval: str,symbols: list,crypto=True):
              r_obj = CryptoBarsRequest if crypto else StockBarsRequest
              request_params = r_obj(
                        symbol_or_symbols=symbols,
                        timeframe=self.interval_map[interval],
                        start=self.str_to_datetime(start),
                        end=self.str_to_datetime(end)
                 )

              bars = self.crypto_client.get_crypto_bars(request_params) if crypto else self.stock_client.get_stock_bars(request_params)


              # convert to dataframe
              return bars.df.reset_index(level=0,drop=True)
       
       def build_env_df(self,start,end,interval,symbols,crypto=True):
              bars = self.get_market_data(start,end,interval,symbols,crypto)
              env_df = self.generate_features(bars,start,end,interval)
              env_df.dropna(axis=1,how='all',inplace=True)
              env_df.dropna(axis=0,subset=[c for c in env_df.columns if c.startswith('feature_') ],inplace=True)
              return env_df



# access bars as list - important to note that you must access by symbol key
# even for a single symbol request - models are agnostic to number of symbols
# bars["BTC/USD"]


if __name__ == "__main__":
       alpacaHelper = AlpacaUtils()
       btc_df = alpacaHelper.build_env_df('2022-07-01','2022-09-01','day','BTC/USD',True)
       print(btc_df) 