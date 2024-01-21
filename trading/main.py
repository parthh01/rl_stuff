from alpaca_functions import AlpacaUtils
from trading_env import run_env



def main(): 
    alpacaHelper = AlpacaUtils()
    df = alpacaHelper.get_crypto_data('2022-07-01','2022-09-01','day')
    run_env(df)



if __name__ == "__main__":
    main()
