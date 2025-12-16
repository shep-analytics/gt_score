import pandas as pd
from hyperopt import hp

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return rolling_mean, upper_band, lower_band

def check_bollinger_action(current_close, upper_band, lower_band, previous_close, previous_upper_band, previous_lower_band):
    if current_close < lower_band and previous_close >= previous_lower_band:
        return 'buy'
    elif current_close > upper_band and previous_close <= previous_upper_band:
        return 'sell'
    return 'none'

def strategy(data, params={'window': 20, 'num_std_dev': 2}):
    window = params.get('window')
    num_std_dev = params.get('num_std_dev')
    
    data_copy = data.copy()
    # window must be an integer because the quniform hyperopt function returns floats
    data_copy['Rolling Mean'], data_copy['Upper Band'], data_copy['Lower Band'] = calculate_bollinger_bands(data_copy, int(window), num_std_dev)
    data_copy['action'] = "none"
    
    # Example: shift comparisons by 1 day
    for i in range(2, len(data_copy)):
        data_copy.at[i, 'action'] = check_bollinger_action(
            current_close=data_copy['Close'].iloc[i-1],
            upper_band=data_copy['Upper Band'].iloc[i-1],
            lower_band=data_copy['Lower Band'].iloc[i-1],
            previous_close=data_copy['Close'].iloc[i-2],
            previous_upper_band=data_copy['Upper Band'].iloc[i-2],
            previous_lower_band=data_copy['Lower Band'].iloc[i-2]
        )
    
    return data_copy

def should_buy_live(data, params={'window': 20, 'num_std_dev': 2}):
    if len(data) < params.get('window'):
        decision = ['none', 0]
    else:
        window = params.get('window')
        num_std_dev = params.get('num_std_dev')
        
        recent_data = data  # Only take the last `window` periods
        rolling_mean, upper_band, lower_band = calculate_bollinger_bands(recent_data, window, num_std_dev)
        
        current_close = recent_data['Close'].iloc[-1]
        previous_close = recent_data['Close'].iloc[-2]
        current_upper_band = upper_band.iloc[-1]
        current_lower_band = lower_band.iloc[-1]
        previous_upper_band = upper_band.iloc[-2]
        previous_lower_band = lower_band.iloc[-2]
        
        decision = [check_bollinger_action(current_close, current_upper_band, current_lower_band, previous_close, previous_upper_band, previous_lower_band), current_close]
    
    return decision

param_space = {
    'window': hp.quniform('window', 10, 50, 1),
    'num_std_dev': hp.uniform('num_std_dev', 1, 3)
}

# define the bounds for genetic algorithm optimization
ga_bounds = {
    'window': (10, 50),
    'num_std_dev': (1, 3)
}

# Example usage:
# df = fetch_historical_data('AAPL', '1d', '2018-01-01', '2024-01-01')
# params = {'window': 20, 'num_std_dev': 2}
# print(should_buy_live(df, params))