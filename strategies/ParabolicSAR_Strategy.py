import pandas as pd
import numpy as np
from hyperopt import hp

def calculate_parabolic_sar(data, af_start=0.02, af_step=0.02, af_max=0.2):
    """
    Calculate Parabolic SAR for the given data.
    - af_start: Initial acceleration factor
    - af_step: Step to increase acceleration factor
    - af_max: Maximum acceleration factor
    """
    data_copy = data.copy()
    data_copy['SAR'] = np.nan
    data_copy['EP'] = np.nan  # Extreme point
    data_copy['AF'] = af_start  # Acceleration factor
    data_copy['Trend'] = 1  # 1 for uptrend, -1 for downtrend
    
    # Initialize values
    data_copy.at[0, 'SAR'] = data_copy['Low'].iloc[0]
    data_copy.at[0, 'EP'] = data_copy['High'].iloc[0]
    
    for i in range(1, len(data_copy)):
        prev_sar = data_copy['SAR'].iloc[i - 1]
        prev_ep = data_copy['EP'].iloc[i - 1]
        prev_af = data_copy['AF'].iloc[i - 1]
        prev_trend = data_copy['Trend'].iloc[i - 1]
        
        if prev_trend == 1:  # Uptrend
            sar = prev_sar + prev_af * (prev_ep - prev_sar)
            sar = min(sar, data_copy['Low'].iloc[i - 1], data_copy['Low'].iloc[i])
        else:  # Downtrend
            sar = prev_sar + prev_af * (prev_ep - prev_sar)
            sar = max(sar, data_copy['High'].iloc[i - 1], data_copy['High'].iloc[i])
        
        if prev_trend == 1 and data_copy['Low'].iloc[i] < sar:
            data_copy.at[i, 'Trend'] = -1
            data_copy.at[i, 'SAR'] = prev_ep
            data_copy.at[i, 'EP'] = data_copy['Low'].iloc[i]
            data_copy.at[i, 'AF'] = af_start
        elif prev_trend == -1 and data_copy['High'].iloc[i] > sar:
            data_copy.at[i, 'Trend'] = 1
            data_copy.at[i, 'SAR'] = prev_ep
            data_copy.at[i, 'EP'] = data_copy['High'].iloc[i]
            data_copy.at[i, 'AF'] = af_start
        else:
            data_copy.at[i, 'Trend'] = prev_trend
            data_copy.at[i, 'SAR'] = sar
            if prev_trend == 1:
                ep = max(prev_ep, data_copy['High'].iloc[i])
            else:
                ep = min(prev_ep, data_copy['Low'].iloc[i])
            data_copy.at[i, 'EP'] = ep
            af = min(prev_af + af_step, af_max) if ep != prev_ep else prev_af
            data_copy.at[i, 'AF'] = af
    
    return data_copy[['SAR', 'Trend']]

def check_sar_action(current_sar, current_trend, prev_sar, prev_trend):
    if current_trend == 1 and prev_trend == -1:
        return 'buy'
    elif current_trend == -1 and prev_trend == 1:
        return 'sell'
    return 'none'

def strategy(data, params={'af_start': 0.02, 'af_step': 0.02, 'af_max': 0.2}):
    af_start = params.get('af_start', 0.02)
    af_step = params.get('af_step', 0.02)
    af_max = params.get('af_max', 0.2)
    
    data_copy = data.copy()
    sar_trend = calculate_parabolic_sar(data_copy, af_start, af_step, af_max)
    data_copy['SAR'] = sar_trend['SAR']
    data_copy['Trend'] = sar_trend['Trend']
    data_copy['action'] = "none"
    
    for i in range(1, len(data_copy)):
        data_copy.at[i, 'action'] = check_sar_action(
            current_sar=data_copy['SAR'].iloc[i],
            current_trend=data_copy['Trend'].iloc[i],
            prev_sar=data_copy['SAR'].iloc[i - 1],
            prev_trend=data_copy['Trend'].iloc[i - 1]
        )
    
    return data_copy

def should_buy_live(data, params={'af_start': 0.02, 'af_step': 0.02, 'af_max': 0.2}):
    if len(data) < 2:
        return ['none', 0]
    
    af_start = params.get('af_start', 0.02)
    af_step = params.get('af_step', 0.02)
    af_max = params.get('af_max', 0.2)
    
    sar_trend = calculate_parabolic_sar(data, af_start, af_step, af_max)
    data['SAR'] = sar_trend['SAR']
    data['Trend'] = sar_trend['Trend']
    current_sar = data['SAR'].iloc[-1]
    prev_sar = data['SAR'].iloc[-2]
    current_trend = data['Trend'].iloc[-1]
    prev_trend = data['Trend'].iloc[-2]
    
    decision = [check_sar_action(current_sar, current_trend, prev_sar, prev_trend), current_sar]
    return decision

# Define the param_space for Hyperopt optimization
param_space = {
    'af_start': hp.uniform('af_start', 0.01, 0.03),
    'af_step': hp.uniform('af_step', 0.01, 0.05),
    'af_max': hp.uniform('af_max', 0.1, 0.3)
}

# Define the bounds for Genetic Algorithm optimization
ga_bounds = {
    'af_start': (0.01, 0.03),
    'af_step': (0.01, 0.05),
    'af_max': (0.1, 0.3)
}
