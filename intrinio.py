import pandas as pd
import numpy as np
import os
import datetime

os.chdir(r'C:\Users\WORKSTATION\Downloads')
# os.chdir(r'C:\Users\WORKSTATION\Desktop\StashawayHandbook')
# os.chdir(r'C:\Users\WORKSTATION\Desktop\ETF')
prices_df = pd.read_csv('stock_prices_xkls_all_file-1.csv', dtype={'TICKER': str})
# prices_df = pd.read_csv('bursa_prices.csv', dtype={'TICKER': str})


prices_df = prices_df[['TICKER', 'DATE', 'ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'ADJ_VOLUME', 'ADJ_FACTOR',
                       'EX_DIVIDEND', 'SPLIT_RATIO', 'PERCENT_CHANGE']]
prices_df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_factor', 'ex_dividend',
                     'split_ratio', 'pct_change']
prices_df['date'] = pd.to_datetime(prices_df['date'])

tickers = np.unique(prices_df['ticker'])
dates = np.unique(prices_df['date'])
date_df = pd.DataFrame({'date': dates})
date_df = date_df.sort_values(by='date')

prices = {}
for t in tickers:
    temp = prices_df[prices_df['ticker'] == t].sort_values(by='date')
    price_df = pd.merge(date_df, temp, how='outer', on='date')
    price_df['volume'] = price_df['volume'].fillna(0)
    price_df['close'] = price_df['close'].fillna(method='ffill')

    # To cater for the mising OHL row
    price_df['open'] = price_df['open'].fillna(price_df['close'])
    price_df['high'] = price_df['high'].fillna(price_df['close'])
    price_df['low'] = price_df['low'].fillna(price_df['close'])
    prices[t] = price_df.copy()

dff = pd.concat([prices[t] for t in tickers])
dff.to_csv('prices_df.csv')