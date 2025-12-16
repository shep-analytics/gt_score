import pandas as pd
from hyperopt import hp

def calculate_donchian_channel(data, window):
    window = int(window)  # Ensure window is an integer
    high = data['High'].rolling(window=window).max()
    low = data['Low'].rolling(window=window).min()
    return high, low

def check_donchian_action(current_close, upper_band, lower_band, prev_close, prev_upper_band, prev_lower_band):
    if current_close > upper_band and prev_close <= prev_upper_band:
        return 'buy'
    elif current_close < lower_band and prev_close >= prev_lower_band:
        return 'sell'
    return 'none'

def strategy(data, params={'window': 20}):
    window = int(params.get('window'))  # Cast to integer
    data_copy = data.copy()
    data_copy['Upper Band'], data_copy['Lower Band'] = calculate_donchian_channel(data_copy, window)
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        data_copy.at[i, 'action'] = check_donchian_action(
            current_close=data_copy['Close'].iloc[i-1],
            upper_band=data_copy['Upper Band'].iloc[i-1],
            lower_band=data_copy['Lower Band'].iloc[i-1],
            prev_close=data_copy['Close'].iloc[i-2],
            prev_upper_band=data_copy['Upper Band'].iloc[i-2],
            prev_lower_band=data_copy['Lower Band'].iloc[i-2]
        )
    
    return data_copy

def should_buy_live(data, params={'window': 20}):
    if len(data) < params.get('window'):
        return ['none', 0]
    
    window = int(params.get('window'))
    recent_data = data.tail(window)
    upper_band, lower_band = calculate_donchian_channel(recent_data, window)
    
    current_close = recent_data['Close'].iloc[-1]
    previous_close = recent_data['Close'].iloc[-2]
    current_upper_band = upper_band.iloc[-1]
    current_lower_band = lower_band.iloc[-1]
    previous_upper_band = upper_band.iloc[-2]
    previous_lower_band = lower_band.iloc[-2]
    
    decision = [check_donchian_action(
        current_close, current_upper_band, current_lower_band,
        previous_close, previous_upper_band, previous_lower_band
    ), current_close]
    return decision

# Define the param_space for Hyperopt optimization
param_space = {
    'window': hp.quniform('window', 10, 50, 1)
}

# Define the bounds for Genetic Algorithm optimization
ga_bounds = {
    'window': (10, 50)
}
