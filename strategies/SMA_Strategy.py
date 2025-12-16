import pandas as pd
from hyperopt import hp

def calculate_sma(data, short_window, long_window):
    short_sma = data['Close'].rolling(window=short_window).mean()
    long_sma = data['Close'].rolling(window=long_window).mean()
    return short_sma, long_sma

def check_sma_action(current_short_sma, current_long_sma, previous_short_sma, previous_long_sma):
    if current_short_sma > current_long_sma and previous_short_sma <= previous_long_sma:
        return 'buy'
    elif current_short_sma < current_long_sma and previous_short_sma >= previous_long_sma:
        return 'sell'
    return 'none'

def strategy(data, params={'short_window': 10, 'long_window': 50}):
    short_window = params.get('short_window')
    long_window = params.get('long_window')
    
    data_copy = data.copy()
    data_copy['Short SMA'], data_copy['Long SMA'] = calculate_sma(data_copy, int(short_window), int(long_window))
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        data_copy.at[i, 'action'] = check_sma_action(
            current_short_sma=data_copy['Short SMA'].iloc[i-1],
            current_long_sma=data_copy['Long SMA'].iloc[i-1],
            previous_short_sma=data_copy['Short SMA'].iloc[i-2],
            previous_long_sma=data_copy['Long SMA'].iloc[i-2]
        )
    
    return data_copy

def should_buy_live(data, params={'short_window': 10, 'long_window': 50}):
    if len(data) < params.get('long_window'):
        return ['none', 0]
    
    short_window = params.get('short_window')
    long_window = params.get('long_window')
    
    recent_data = data[-long_window:]  # Take the last `long_window` periods
    short_sma, long_sma = calculate_sma(recent_data, short_window, long_window)
    
    if len(short_sma) < 2:
        return ['none', 0]
    
    current_short_sma = short_sma.iloc[-1]
    current_long_sma = long_sma.iloc[-1]
    previous_short_sma = short_sma.iloc[-2]
    previous_long_sma = long_sma.iloc[-2]
    
    decision = [check_sma_action(current_short_sma, current_long_sma, previous_short_sma, previous_long_sma), current_short_sma]
    return decision

# Define the param_space for Hyperopt optimization
param_space = {
    'short_window': hp.quniform('short_window', 5, 20, 1),
    'long_window': hp.quniform('long_window', 30, 100, 1)
}

# Define the bounds for the Genetic Algorithm optimization
ga_bounds = {
    'short_window': (5, 20),
    'long_window': (30, 100)
}

# Example usage:
# df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
# params = {'short_window': 10, 'long_window': 50}
# print(should_buy_live(df, params))
