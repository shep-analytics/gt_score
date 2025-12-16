import pandas as pd
from hyperopt import hp

def detect_elliott_wave(data, window=20):
    data_copy = data.copy()
    data_copy['Swing High'] = data['High'].rolling(window=int(window)).max()
    data_copy['Swing Low'] = data['Low'].rolling(window=int(window)).min()
    data_copy.dropna(inplace=True)  # Drop rows with NaN values from rolling calculations
    
    waves = []
    for i in range(1, len(data_copy) - 1):
        if data_copy['High'].iloc[i] > data_copy['High'].iloc[i - 1] and data_copy['High'].iloc[i] > data_copy['High'].iloc[i + 1]:
            waves.append(('peak', i, data_copy['High'].iloc[i]))
        elif data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i - 1] and data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i + 1]:
            waves.append(('trough', i, data_copy['Low'].iloc[i]))
    return waves

def check_elliott_action(waves, current_index, close_price):
    if len(waves) < 5:
        return 'none'
    last_wave = waves[-1]
    if last_wave[0] == 'peak' and last_wave[1] == current_index:
        return 'sell'
    elif last_wave[0] == 'trough' and last_wave[1] == current_index:
        return 'buy'
    return 'none'

def strategy(data, params={'window': 20}):
    window = int(params.get('window'))
    data_copy = data.copy()
    waves = detect_elliott_wave(data_copy, window)
    data_copy['action'] = "none"
    
    for i in range(len(data_copy)):
        data_copy.at[i, 'action'] = check_elliott_action(waves, i, data_copy['Close'].iloc[i])
    
    return data_copy.dropna()  # Drop rows with NaN actions to avoid downstream issues

def should_buy_live(data, params={'window': 20}):
    if len(data) < params.get('window'):
        return ['none', 0]
    
    window = int(params.get('window'))
    waves = detect_elliott_wave(data, window)
    current_index = len(data) - 1
    current_close = data['Close'].iloc[-1]
    decision = [check_elliott_action(waves, current_index, current_close), current_close]
    return decision

# Define the param_space for Hyperopt optimization
param_space = {
    'window': hp.quniform('window', 10, 50, 1)
}

# Define the bounds for Genetic Algorithm optimization
ga_bounds = {
    'window': (10, 50)
}
