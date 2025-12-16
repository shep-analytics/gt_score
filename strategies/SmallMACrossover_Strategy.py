import pandas as pd
from hyperopt import hp

def calculate_small_ma(data, short_window, long_window):
    short_ma = data['Close'].rolling(window=int(short_window)).mean()
    long_ma = data['Close'].rolling(window=int(long_window)).mean()
    return short_ma, long_ma

def check_small_ma_action(current_short_ma, current_long_ma, prev_short_ma, prev_long_ma):
    if current_short_ma > current_long_ma and prev_short_ma <= prev_long_ma:
        return 'buy'
    elif current_short_ma < current_long_ma and prev_short_ma >= prev_long_ma:
        return 'sell'
    return 'none'

def strategy(data, params={'short_window': 5, 'long_window': 10}):
    short_window = int(params.get('short_window'))
    long_window = int(params.get('long_window'))
    
    data_copy = data.copy()
    data_copy['Short MA'], data_copy['Long MA'] = calculate_small_ma(data_copy, short_window, long_window)
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        data_copy.at[i, 'action'] = check_small_ma_action(
            current_short_ma=data_copy['Short MA'].iloc[i-1],
            current_long_ma=data_copy['Long MA'].iloc[i-1],
            prev_short_ma=data_copy['Short MA'].iloc[i-2],
            prev_long_ma=data_copy['Long MA'].iloc[i-2]
        )
    
    return data_copy

def should_buy_live(data, params={'short_window': 5, 'long_window': 10}):
    if len(data) < params.get('long_window'):
        return ['none', 0]
    
    short_window = int(params.get('short_window'))
    long_window = int(params.get('long_window'))
    recent_data = data[-long_window:]
    short_ma, long_ma = calculate_small_ma(recent_data, short_window, long_window)
    
    if len(short_ma) < 2:
        return ['none', 0]
    
    current_short_ma = short_ma.iloc[-1]
    current_long_ma = long_ma.iloc[-1]
    prev_short_ma = short_ma.iloc[-2]
    prev_long_ma = long_ma.iloc[-2]
    
    decision = [check_small_ma_action(current_short_ma, current_long_ma, prev_short_ma, prev_long_ma), current_short_ma]
    return decision

# Define the param_space for Hyperopt optimization
param_space = {
    'short_window': hp.quniform('short_window', 3, 10, 1),
    'long_window': hp.quniform('long_window', 8, 15, 1)
}

# Define the bounds for Genetic Algorithm optimization
ga_bounds = {
    'short_window': (3, 10),
    'long_window': (8, 15)
}
