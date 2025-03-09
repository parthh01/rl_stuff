from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest,CryptoLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from datetime import datetime
from dotenv import load_dotenv
import os
from ta import add_all_ta_features
from ta.utils import dropna
import pandas as pd

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '../.env'),override=True)

class AlpacaUtils:
       def __init__(self,paper=True):
              # no keys required for crypto data
              self.crypto_client = CryptoHistoricalDataClient()
              self.stock_client = StockHistoricalDataClient(os.environ.get('APCA_API_KEY_ID'),  os.environ.get('APCA_API_SECRET_KEY'))
              self.interval_map = {'day': TimeFrame.Day, 'min': TimeFrame.Minute, 'hour': TimeFrame.Hour, 'week': TimeFrame.Week, 'month': TimeFrame.Month}
              self.trading_client = TradingClient(os.environ.get('APCA_API_KEY_ID'),  os.environ.get('APCA_API_SECRET_KEY'),paper=paper)
       
       def size_to_order(self,size,symbol,crypto=True):
       #"""converting desired portfolio size to order"""
              account = self.trading_client.get_account()
              positions = self.trading_client.get_all_positions()
              current_holding = 0
              market_order = None
              for p in positions:
                     if p.symbol == symbol:
                            current_holding = float(self.trading_client.get_open_position(symbol).market_value)

              if size == 0 and current_holding != 0:
                     self.trading_client.close_position(symbol_or_asset_id=symbol)
              else:
                     portfolio_value = float(account.equity)
                     desired_holding = size*portfolio_value
                     # latest_quote = self.crypto_client.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=symbol))[symbol].ask_price
                     order_size = (desired_holding - current_holding)
                     order_size = max(min(float(account.cash),order_size),-current_holding) # ensure order size is not bigger than current holding if sell, and current cash if buy
                     order_size = round(order_size,2)
                     if abs(order_size) > 1: # magnitude of change is greater than a dollar
                            market_order_data = MarketOrderRequest(
                            symbol=symbol,
                            notional=abs(order_size),
                            side=OrderSide.BUY if order_size > 0 else OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                            )
                            market_order = self.trading_client.submit_order(order_data=market_order_data)
                     
              return market_order

       def get_latest_bar(self,symbol):
              return self.crypto_client.get_crypto_latest_bar(CryptoLatestBarRequest(symbol_or_symbols=[symbol]))[symbol]

       def str_to_datetime(self,date:str):
              y,m,d = [int(i) for i in date.split('-')]
              return datetime(y,m,d)

       def generate_features(self,df,start="",end="",interval="",spy=True):
              if spy:
                     spy_data = self.get_market_data(start,end,interval,["SPY"],crypto=False)
                     spy_data.rename(columns={c: 'spy_' + c for c in spy_data.columns},inplace=True)
                     df = pd.merge_asof(df,spy_data[["spy_close"]],left_index=True,right_index=True)
              df = add_all_ta_features(df,'open','high','low','close','volume',colprefix='feature_',fillna=True)
              #df.rename({c:'feature_'+c for c in ['vwap','open']},inplace=True)
              df.dropna(axis=1,how='all',inplace=True)
              df.dropna(axis=0,subset=[c for c in df.columns if c.startswith('feature_') ],inplace=True)
              return df 
        
       def get_market_data(self,start: str = "",end: str = "",interval: str = "min",symbols: list = [],crypto=True):
              r_obj = CryptoBarsRequest if crypto else StockBarsRequest
              request_params = r_obj(
                        symbol_or_symbols=symbols,
                        timeframe=self.interval_map[interval],
                        start=self.str_to_datetime(start) if len(start) > 0 else None,
                        end=self.str_to_datetime(end) if len(end) > 0 else None
                 )

              bars = self.crypto_client.get_crypto_bars(request_params) if crypto else self.stock_client.get_stock_bars(request_params)


              # convert to dataframe
              return bars.df.reset_index(level=0,drop=True)
       
       def build_env_df(self,start = "",end = "",interval = "min",symbols = [],crypto=True):
              bars = self.get_market_data(start,end,interval,symbols,crypto)
              env_df = self.generate_features(bars,start,end,interval)
              return env_df



# access bars as list - important to note that you must access by symbol key
# even for a single symbol request - models are agnostic to number of symbols
# bars["BTC/USD"]


if __name__ == "__main__":
       alpacaHelper = AlpacaUtils()
       btc_df = alpacaHelper.build_env_df(start='2023-01-01',interval='min',symbols='BTC/USD',crypto=True)
       # alpacaHelper.size_to_order(0.1,"BTCUSD")
       print(btc_df)
       # latest_bar = alpacaHelper.crypto_client.get_crypto_latest_bar(CryptoLatestBarRequest(symbol_or_symbols=['BTC/USD']))
       # print(latest_bar)