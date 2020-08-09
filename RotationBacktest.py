import pandas as pd
import numpy as np
import os
import datetime
import time
import math
from talib import abstract
import concurrent.futures
import matplotlib.pyplot as plt

# os.chdir(r'C:\Users\ruennhuah.lee\Downloads')
os.chdir(r'C:\Users\WORKSTATION\Downloads')
# os.chdir(r'C:\Users\WORKSTATION\Desktop\StashawayHandbook')
# os.chdir(r'C:\Users\WORKSTATION\Desktop\ETF')

prices_df = pd.read_csv('stock_prices_xkls_all_file-1.csv', dtype={'TICKER': str})


prices_df = prices_df[['TICKER', 'DATE', 'ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'ADJ_VOLUME','ADJ_FACTOR',
       'EX_DIVIDEND', 'SPLIT_RATIO', 'PERCENT_CHANGE']]
prices_df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_factor', 'ex_dividend', 'split_ratio', 'pct_change']
prices_df['date'] = pd.to_datetime(prices_df['date'])


tickers = np.unique(prices_df['ticker'])
dates = np.unique(prices_df['date'])
dates = pd.to_datetime(dates).sort_values()

date_df = pd.DataFrame({'date':dates})
date_df = date_df.sort_values(by = 'date')


prices = {}
for t in tickers:
    temp = prices_df[prices_df['ticker'] == t].sort_values(by = 'date')
    price_df = pd.merge(date_df, temp, how = 'outer', on = 'date')
    price_df['volume'] = price_df['volume'].fillna(0)
    price_df['close'] = price_df['close'].fillna(method='ffill')
    
    # To cater for the mising OHL row
    price_df['open'] = price_df['open'].fillna(price_df['close'])
    price_df['high'] = price_df['high'].fillna(price_df['close'])
    price_df['low'] = price_df['low'].fillna(price_df['close'])
    prices[t] = price_df.copy()
    prices[t] = prices[t].set_index('date').sort_index()
    
dff = pd.concat([prices[t] for t in tickers])
dff['tradable'] = dff['close'] != 0
dff.to_csv('prices_df.csv')

print("Done Exporting Prices.")

# dff.head()


# dff = pd.read_csv('prices_df.csv', dtype={'ticker': str})
# dff.head()



def polyfit(x, y, degree):
    #https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    results = {}
    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

def adj_slope_on_array(close):
    lookback_day = len(close)
    log_close = np.log(close)
    results = polyfit(x = np.arange(lookback_day), y = log_close, degree = 1)
    slope = results['polynomial'][0]
    r2 = results['determination']
    ann_slope = np.power((1+ slope),250)-1
    adj_slope = 100*ann_slope*r2
    return slope, r2, adj_slope

def adj_slope(eval_date, ticker, lookback_day):
    df1 = prices[t]
    # the evaluation date is excluded from slope calculation
    # order to be placed at open on evaluation date
    close = np.array(df1[:eval_date].tail(lookback_day+1)[:-1]['close'].dropna())
    if len(close) < lookback_day:
        return None
    else:
        s, r, adj_s = adj_slope_on_array(close)
        return adj_s
    
def adj_slope_function(y):
    x = range(1, len(y)+1)
    results = polyfit(x = x, y = y, degree = 1)
    rsq_value = results["determination"]
    expslope_value = results["polynomial"][0]
    annslope_value = np.power((1+ expslope_value),250)-1
    return 100*annslope_value*rsq_value

def adj_slope_df(df, lookback_day, column = 'adj_close'):
    df = df.sort_values("date")
    df1 = df.copy()
    df1['log_close'] = np.log(df1[column])
    adj_slope = df1['log_close'].rolling(window = lookback_day).apply(lambda x: adj_slope_function(x), raw = True).fillna(0).values

    df1[f'adj_slope{lookback_day}'] = adj_slope
    return df1

def add_talib_ind(df, func, parameters_dict, col_names = None):
    func.set_parameters(parameters_dict)

    try:

        func.set_input_arrays({'open':df['open'].values, 'high': df['high'].values,
        'low': df['low'].values, 'close': df['close'].values, 'volume': df['volume'].values})

        if col_names == None:
            col_names = func.output_names
        else:
            if len(func.output_names)!=len(col_names):
                print("The number of outputs do not match the number of column names")
                return

        output_series = func.run()
        output_dict = {}

        if len(func.output_names) == 1:
            output_dict[col_names[0]] = output_series
        else:
            for i in range(len(func.output_names)):
                output_dict[col_names[i]] = output_series[i]

        df = df.assign(**output_dict)
        return df
    except:
        print(df['ticker'].head(0))
        
        
        
def initialize_indicators(df1):
    ATR_params = [20]
    SMA_params = list(range(5,301,5))
    slope_params = list(range(5,301,5))

    print("ATR")
    for n in ATR_params:
        df1 = add_talib_ind(df1, abstract.ATR, parameters_dict={'timeperiod':n},col_names=[f'ATR{n}'])

    print("Inv_Vol, avg_slope")
    df1['Inv_Vol'] = df1['close']/df1['ATR20']

    print("SMA")
    for n in SMA_params:
        df1 = add_talib_ind(df1, abstract.SMA, parameters_dict={'timeperiod':n},col_names=[f'SMA{n}'])

    print("SLOPE")
    try:
        for lookback_day in slope_params:
            df1 = adj_slope_df(df1, lookback_day, column = 'close')
    except:
        print(lookback_day,"| PROBLEM ENCOUNTERED")
        
    return df1
        

temp_date_df = date_df['date'].groupby(date_df['date'].astype(str).str[:8]).first().to_frame()
temp_date_df['date'] = pd.to_datetime(temp_date_df['date'])
temp_date_df.head()

order_dates = [x for x in temp_date_df['date']]
order_dates[:5]

start = time.time()

d = {}
print(f'Total Tickers: {len(tickers)}')
count = 1
for s in tickers:
    d[s] = initialize_indicators(prices[s])
    count = count + 1
    print(f'Current Count: {count}')
    
end = time.time()
print(end-start)


s = '6939'
z = d[s]
print(len(z[pd.isna(z['ticker'])]))
z['ticker'] = z['ticker'].fillna(s)
print(len(z[pd.isna(z['ticker'])]))


for s in d:
    d[s]['ticker'] = d[s]['ticker'].fillna(s)


df5 = pd.concat([d[s] for s in d])
df5.to_csv('full_klse_raw.csv')



param = [30,60]
top = 30
min_slope = 20
risk_factor = 0.001
initial_capital = 100000
fees_in_decimal = 0.004

target_dicts = []
equity_list = []
cash_list = []
cost_list = []
eod_values = []

orders = []


plot_dates = []
plot_equities = []

eval_date = [dates[np.nonzero(dates == x)[0][0] - 1] for x in order_dates]

first_eval_date_index = math.ceil(max(param)/30)
for i in range(len(order_dates)):
# for i in range(15):
    # i=4
    if i < first_eval_date_index:
        target_dicts.append({})
        equity_list.append(0)
        cash_list.append(initial_capital)
        cost_list.append(0)
        eod_values.append(initial_capital)
        continue
        
    # eval_date = order_date - 1
    eval_df = df5.loc[eval_date[i]].copy()
    
    # Filtering Logic
    
    
                    
    # Position Sizing
    # Exclude the condition to restrict the top N for position sizing, do position sizing for all stocks
    order_df = df5.loc[order_dates[i]].copy()
    
    sell_df = pd.DataFrame(target_dicts[i-1].items(), columns=['ticker', 'quantity_on_hand']).set_index('ticker')
    merged_sell_df = pd.merge(order_df, sell_df, left_on='ticker', right_index=True)[['ticker','open','quantity_on_hand']]
    merged_sell_df = merged_sell_df.assign(market_value=merged_sell_df['quantity_on_hand']*merged_sell_df['open']*(1-fees_in_decimal))
    merged_sell_df = merged_sell_df.assign(remark=f'{i} | sell')
    sell_records = merged_sell_df[['ticker', 'quantity_on_hand', 'open', 'remark']].copy()
    sell_records.columns = ['ticker', 'quantity', 'price', 'remark']
    sell_records['quantity'] = -sell_records['quantity']
    orders.append(sell_records)
    
    market_value_upon_selling1 = np.sum(merged_sell_df['market_value'])
    
    market_value_upon_selling = np.sum([order_df[order_df['ticker']==stock]['open'].values[0]*target_dicts[i-1][stock]*(1-fees_in_decimal) if order_df[order_df['ticker']==stock].empty == False else 0 for stock in target_dicts[i-1]])
    
    
    equity_list.append(market_value_upon_selling)
    current_portfolio_value = market_value_upon_selling + cash_list[i-1]
    
    
    eligible = eval_df[eval_df['ATR20']>0.0001]
    eligible = eligible[eligible['close']>1.0]
    eligible = eligible[eligible['volume'] != 0]
    eligible = eligible[['ticker','close','ATR20','Inv_Vol']]
    eligible = eligible.assign(ToBuy=np.floor(current_portfolio_value*risk_factor/(100*eligible['ATR20']))*100)
    eligible['ToBuy'] = eligible['ToBuy'].astype(int)
    
    
    
    # target_proportion = eligible.set_index('ticker')['Proportion']
    target_proportion = eligible.set_index('ticker')['ToBuy']
    
    order_df = df5.loc[order_dates[i]].copy()
    # order_df = order_df[order_df['ticker'].isin(target_proportion.index)]
    order_df = pd.merge(order_df, target_proportion, left_on='ticker', right_index=True)
    order_df = order_df.assign(avg_slope = 0.5*(order_df[f'adj_slope{param[0]}'] + order_df[f'adj_slope{param[1]}']))
    order_df = order_df.assign(rank = order_df['avg_slope'].rank(method = 'first', ascending = False))
    order_df = order_df[(order_df['avg_slope']>min_slope)].sort_values(by = 'rank')
    
    # To include: eval_df['rank']<=top
    
    # order_df = order_df.assign(ToBuy = (100*np.floor(order_df['Proportion']*(market_value_upon_selling+cash_list[i-1])/(1+fees_in_decimal)/order_df['close']/100)).astype(int))
    order_df = order_df.assign(Cost = order_df['ToBuy']*order_df['open'])
    order_df = order_df.assign(CumulativeCost = order_df['Cost'].cumsum())
    order_df = order_df[order_df['CumulativeCost']<=market_value_upon_selling+cash_list[i-1]]
    order_df = order_df.assign(remark=f'{i} | buy')
    buy_record = order_df[['ticker', 'ToBuy', 'open', 'remark']].copy()
    buy_record.columns = ['ticker', 'quantity', 'price', 'remark']
    
    # if i != range(len(order_dates)-1):
    orders.append(buy_record)
    # print(order_df)
    
    target_dict = order_df.set_index('ticker')['ToBuy'].to_dict()
    target_dict = {k:v for k,v in target_dict.items() if v != 0}
    
    
    total_cost = order_df['CumulativeCost'].max() if order_df.empty == False else 0.0
    cash = market_value_upon_selling + cash_list[i-1] - total_cost
    order_df = order_df.assign(Eod_value = order_df['ToBuy']*order_df['close'])
    eod_market_value = order_df['Eod_value'].sum()
    
    
    # print(i)
    # print(target_dict)
    # print(market_value_upon_selling)
    # print(cash)
    # print(total_cost)
    # print(eod_market_value)
    # print(f'{i}| Sold: {market_value_upon_selling} | PreviousCash: {cash_list[i-1]} === Cost: {total_cost} | EOD_MarketVal: {eod_market_value} | CurrentCash: {cash} | profit: {eod_market_value-total_cost}')
    # print(f'{i}| Buy: {equity_list[i-1]} Sold: {market_value_upon_selling} | diff = {market_value_upon_selling - equity_list[i-1]} | cash = {cash_list[i-1]}')
    
    if not target_dict:  #empty dict
        target_dicts.append({})
    else:
        target_dicts.append(target_dict)
    equity_list.append(market_value_upon_selling)
    cash_list.append(cash)
    cost_list.append(total_cost)
    eod_values.append(eod_market_value)
    
    
    plot_dates.append(order_dates[i-1])
    plot_equities.append(current_portfolio_value)
    print(f'{order_dates[i-1]} | {current_portfolio_value}')
    
    
    
    # Execute Orders on order_date[i]
    
plot_df = pd.DataFrame({'date':plot_dates, 'Equity':plot_equities}).set_index('date')
plot_df.plot()
plt.show()    
    
orders_dff = pd.concat([x for x in orders])
orders_dff = orders_dff[orders_dff['remark'] != "162 | buy"]
orders_dff = orders_dff.assign(net_amount = -1*orders_dff['quantity']*orders_dff['price'])
orders_dff = orders_dff.assign(portfolio_value = 100000 + orders_dff['net_amount'].cumsum())
                               
    
    
class Order:
    
    
    def __init__(self, order_date, ticker, order_quantity, order_price, order_remark):
        if order_quantity == 0:
            raise ValueError('Order Quantity must be nonzero')
        _commission = 0.004
        _sign = -1 if order_quantity > 0 else 1     # Purchase is negative, Sell is positive
        
        self.date = order_date
        self.ticker = ticker
        self.quantity = order_quantity
        self.price = order_price
        self.type = 'P' if _sign < 0 else 'S'
        self.amount =  abs(self.quantity * self.price)
        self.commission = self.amount * _commission
        self.net_amount = _sign * self.amount - self.commission
        self.remark = order_remark
        
    def __str__(self):
        return f'{self.date}|{self.ticker}|{self.quantity}|{self.price}|{self.remark}'
    
    @staticmethod
    def from_df(df, columns = None):
        if df.empty:
            return None
        num_order = len(df)
        
        if columns == None:
            columns = ['ticker','quantity','price','remark']
        d = df.index
        t = columns[0]
        q = columns[1]
        p = columns[2]
        r = columns[3]

        dates = np.array(d)
        tickers = np.array(df[t])
        quantities = np.array(df[q])
        prices = np.array(df[p])
        remarks = np.array(df[r])
        
        orders = []
        for i in range(num_order):
            order = Order(tickers[i], dates[i], quantities[i], prices[i], remarks[i])
            orders.append(order)
            
        return orders
        
class Algorithm:
    '''
    Example algorithm for trading. Must implement a "generate_orders" function which returns a list of orders.
    Each order is a tuple of the form
        ( Stock Ticker str, Current Price float, Order Amount in shares float)

    Algorithm trades for stocks via a rolling window strategy, and randomly liquidates positions
    '''
    def __init__(self):
        self._averages = {}
        self._lambda = .5
        # self._fee_estimate = lambda x : x*.04+10
        self._updates = 0
        self._price_window = 20
        self._trend = np.zeros(self._price_window)
        self._minimum_wait_between_trades = 5 # Must be less than price window
        self._last_trade = 0
        self._last_date = None

    def add_stock(self, stock, price):
        self._averages[stock] = price

    def _determine_if_trading(self, date, portfolio_value, cash_balance):
        time_delay_met = True
        trade = False
        override = False
        self._updates += 1

        if self._last_date is not None:
            if (date - self._last_date).days <= self._minimum_wait_between_trades:
                # Make orders based on previous day
                return False

        if self._updates == self._price_window+1:
            trade = True

        if (np.mean(self._trend)-portfolio_value)/portfolio_value > 0.05:
            override = True

        if cash_balance > portfolio_value*.03:
            override = True

        return trade or override

    def generate_orders(self, timestamp, portfolio):
        orders = []
        cash_balance = portfolio.balance
        portfolio_value = portfolio.get_total_value()
        self.add_trend_value(portfolio_value)

        if not self._determine_if_trading(timestamp,portfolio_value,cash_balance):
            return orders

        valid_stocks = [stock for stock in self._averages if portfolio.get_update_count(stock) > self._price_window]

        if len(valid_stocks) == 0:
            return orders

        for stock in np.random.choice(valid_stocks, replace=False, size=len(valid_stocks)):
            amt = cash_balance / len(valid_stocks) # Spend available cash
            relative_change = (self.get_window_average(stock=stock) - self.get_price(stock))/self.get_price(stock)

            if abs(relative_change) > .03:
                # Positive is buy, negative is sell
                order_type = np.sign(relative_change)
                if order_type > 0 and np.random.uniform(0,1,size=1)[0] < .9:
                    amt = np.round(amt/self.get_price(stock),0)
                else:
                    amt = - portfolio.get_shares(stock) # Liquidate! Why not?

                if abs(amt) < .01:
                    # Stop small trades
                    continue

                orders.append((stock, self.get_price(stock), amt))

        self._last_trade = self._updates
        self._last_date = timestamp

        return orders

    def get_window_average(self, stock):
        return np.mean(self._averages[stock]['History'])

    def update(self, stock, price):
        if stock in self._averages:
            self.add_price(stock, price)
        else:
            length = self._price_window
            self._averages[stock] = {'History' : np.zeros(length), 'Index' : 0, 'Length' : length}
            data = self._averages[stock]['History']
            data[0] = price

    def get_price(self, stock):
        # Assumes history is full
        return self._averages[stock]['History'][-1]

    def add_price(self, stock, price):
        history = self._averages[stock]['History']
        ind = self._averages[stock]['Index']
        length = self._averages[stock]['Length']
        if ind < length-1:
            history[ind+1] = price
            self._averages[stock]['Index'] = ind + 1
        elif ind == length-1:
            history[:-1] = history[1:]
            history[-1] = price

    def add_trend_value(self, value):
        history = self._trend
        if self._updates <= self._price_window - 1:
            history[self._updates] = value
        elif self._updates > self._price_window-1:
            history[:-1] = history[1:]
            history[-1] = value
            
            
            
class Portfolio:
    def __init__(self, balance=1000000):
        self._portfolio = {}
        self._portfolio['**CASH**'] = {'Shares' : balance, 'Price' : 1.0, 'Updates' : 1}

    def update(self, price, ticker):
        if ticker in self._portfolio:
            self._portfolio[ticker]['Price'] = price
            self._portfolio[ticker]['Updates'] = self._portfolio[ticker]['Updates'] + 1
        else:
            self._portfolio[ticker] = {}
            self._portfolio[ticker]['Price'] = price
            self._portfolio[ticker]['Shares'] = 0
            self._portfolio[ticker]['Updates'] = 1

    @property
    def balance(self):
        return self._portfolio['**CASH**']['Shares']

    @balance.setter
    def balance(self, balance):
        self._portfolio['**CASH**']['Shares'] = balance

    def adjust_balance(self, delta):
        self._portfolio['**CASH**']['Shares'] = self.balance + delta

    def __contains__(self, item):
        return (item in self._portfolio)

    def value_summary(self, date):
        sum = self.get_total_value()
        return '%s : Stock value: %s, Cash: %s, Total %s' % (date, sum-self.balance, self.balance, sum)

    def value_summary_print(self, date):
        sum = self.get_total_value()
        return f"{date},{sum}"

    def get_total_value(self):
        sum = 0
        for stock in self._portfolio.values():
            sum += stock['Shares'] * stock['Price']
        return sum

    def get_price(self, ticker):
        return self._portfolio[ticker]['Price']

    def get_shares(self, ticker):
        return self._portfolio[ticker]['Shares']

    def get_update_count(self, ticker):
        return self._portfolio[ticker]['Updates']

    def set_shares(self, ticker, shares):
        self._portfolio[ticker]['Shares'] = shares

    def update_shares(self, ticker, share_delta):
        self.set_shares(ticker, self.get_shares(ticker) + share_delta)

    def update_trade(self, ticker, share_delta, price, fee):
        # Assumes negative shares are sells, requires validation from Controller
        self.set_shares(ticker, self.get_shares(ticker) + share_delta)
        self.adjust_balance(-(price*share_delta + fee))

    def __str__(self):
        return self._portfolio.__str__()