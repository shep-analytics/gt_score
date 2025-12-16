import pandas as pd
from hyperopt import hp

def calculate_ichimoku(data, tenkan_window=9, kijun_window=26, senkou_span_b_window=52):
    # Cast windows to integers
    tenkan_window = int(tenkan_window)
    kijun_window = int(kijun_window)
    senkou_span_b_window = int(senkou_span_b_window)
    
    data_copy = data.copy()
    
    # Tenkan-sen (Conversion Line)
    data_copy['Tenkan-sen'] = (data_copy['High'].rolling(window=tenkan_window).max() + 
                               data_copy['Low'].rolling(window=tenkan_window).min()) / 2
    
    # Kijun-sen (Base Line)
    data_copy['Kijun-sen'] = (data_copy['High'].rolling(window=kijun_window).max() + 
                              data_copy['Low'].rolling(window=kijun_window).min()) / 2
    
    # Senkou Span A (Leading Span A)
    data_copy['Senkou Span A'] = ((data_copy['Tenkan-sen'] + data_copy['Kijun-sen']) / 2).shift(kijun_window)
    
    # Senkou Span B (Leading Span B)
    data_copy['Senkou Span B'] = ((data_copy['High'].rolling(window=senkou_span_b_window).max() + 
                                   data_copy['Low'].rolling(window=senkou_span_b_window).min()) / 2).shift(kijun_window)
    
    return data_copy

def check_ichimoku_action(current_close, tenkan, kijun, senkou_a, senkou_b, prev_close):
    if current_close > tenkan > kijun and current_close > max(senkou_a, senkou_b):
        return 'buy'
    elif current_close < tenkan < kijun and current_close < min(senkou_a, senkou_b):
        return 'sell'
    return 'none'

def strategy(data, params={'tenkan_window': 9, 'kijun_window': 26, 'senkou_span_b_window': 52}):
    data_copy = calculate_ichimoku(data, **params)
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        data_copy.at[i, 'action'] = check_ichimoku_action(
            current_close=data_copy['Close'].iloc[i-1],
            tenkan=data_copy['Tenkan-sen'].iloc[i-1],
            kijun=data_copy['Kijun-sen'].iloc[i-1],
            senkou_a=data_copy['Senkou Span A'].iloc[i-1],
            senkou_b=data_copy['Senkou Span B'].iloc[i-1],
            prev_close=data_copy['Close'].iloc[i-2]
        )
    
    return data_copy

def should_buy_live(data, params={'tenkan_window': 9, 'kijun_window': 26, 'senkou_span_b_window': 52}):
    if len(data) < params.get('senkou_span_b_window', 52):
        return ['none', 0]
    
    recent_data = calculate_ichimoku(data.tail(params['senkou_span_b_window']), **params)
    current_row = recent_data.iloc[-1]
    previous_row = recent_data.iloc[-2]
    
    decision = [check_ichimoku_action(
        current_close=current_row['Close'],
        tenkan=current_row['Tenkan-sen'],
        kijun=current_row['Kijun-sen'],
        senkou_a=current_row['Senkou Span A'],
        senkou_b=current_row['Senkou Span B'],
        prev_close=previous_row['Close']
    ), current_row['Close']]
    
    return decision

# Define the param_space for Hyperopt optimization
param_space = {
    'tenkan_window': hp.quniform('tenkan_window', 5, 20, 1),
    'kijun_window': hp.quniform('kijun_window', 21, 30, 1),
    'senkou_span_b_window': hp.quniform('senkou_span_b_window', 45, 60, 1)
}

# Define the bounds for Genetic Algorithm optimization
ga_bounds = {
    'tenkan_window': (5, 20),
    'kijun_window': (21, 30),
    'senkou_span_b_window': (45, 60)
}
