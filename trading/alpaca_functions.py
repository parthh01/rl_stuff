from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime


class AlpacaUtils:
       def __init__(self):
              # no keys required for crypto data
              self.crypto_client = CryptoHistoricalDataClient()
              self.interval_map = {'day': TimeFrame.Day, 'min': TimeFrame.Minute, 'hour': TimeFrame.Hour, 'week': TimeFrame.Week, 'month': TimeFrame.Month}

       def str_to_datetime(self,date:str):
              y,m,d = [int(i) for i in date.split('-')]
              return datetime(y,m,d)
       
       def get_crypto_data(self,start: str,end: str,interval: str,symbols: list = ["BTC/USD","ETH/USD"]):
              request_params = CryptoBarsRequest(
                        symbol_or_symbols=symbols,
                        timeframe=self.interval_map[interval],
                        start=self.str_to_datetime(start),
                        end=self.str_to_datetime(end)
                 )

              bars = self.crypto_client.get_crypto_bars(request_params)

              # convert to dataframe
              return bars.df

# access bars as list - important to note that you must access by symbol key
# even for a single symbol request - models are agnostic to number of symbols
# bars["BTC/USD"]


if __name__ == "__main__":
       alpacaHelper = AlpacaUtils()
       bars = alpacaHelper.get_crypto_data('2022-07-01','2022-09-01','day','BTC/USD')
       print(bars) 