import pandas as pd
from hyperopt import hp

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def check_macd_action(current_macd, current_signal, previous_macd, previous_signal):
    if current_macd > current_signal and previous_macd <= previous_signal:
        return 'buy'
    elif current_macd < current_signal and previous_macd >= previous_signal:
        return 'sell'
    return 'none'

def strategy(data, params={
    'short_window': 12, 
    'long_window': 26, 
    'signal_window': 9,
    'take_profit_stop_loss': 0,  # 0 = TP/SL off, 1 = TP/SL on
    'take_profit_pct': 0.005,    # Example: 0.5%
    'stop_loss_pct': 0.005       # Example: 0.5%
}):
    short_window = int(params.get('short_window', 12))
    long_window = int(params.get('long_window', 26))
    signal_window = int(params.get('signal_window', 9))
    tpsl_flag = params.get('take_profit_stop_loss', 0)
    take_profit_pct = params.get('take_profit_pct', 0.005)
    stop_loss_pct = params.get('stop_loss_pct', 0.005)

    data_copy = data.copy()
    # Calculate MACD & Signal
    data_copy['MACD'], data_copy['Signal'] = calculate_macd(
        data_copy, short_window, long_window, signal_window
    )
    data_copy['action'] = 'none'

    in_position = False
    buy_price = None
    
    for i in range(1, len(data_copy)):
        current_macd = data_copy['MACD'].iloc[i-1]
        current_signal = data_copy['Signal'].iloc[i-1]
        previous_macd = data_copy['MACD'].iloc[i-2] if i >= 2 else None
        previous_signal = data_copy['Signal'].iloc[i-2] if i >= 2 else None
        close_price = data_copy['Close'].iloc[i]
        
        if not in_position:
            # Check for MACD-based BUY
            if (current_macd > current_signal and 
                previous_macd is not None and 
                previous_signal is not None and
                previous_macd <= previous_signal):
                data_copy.at[i, 'action'] = 'buy'
                in_position = True
                buy_price = close_price
        else:
            # If TP/SL is ON, only sell when TP or SL is reached
            if tpsl_flag == 1:
                # Take Profit
                if close_price >= buy_price * (1 + take_profit_pct):
                    data_copy.at[i, 'action'] = 'sell'
                    in_position = False
                    buy_price = None
                # Stop Loss
                elif close_price <= buy_price * (1 - stop_loss_pct):
                    data_copy.at[i, 'action'] = 'sell'
                    in_position = False
                    buy_price = None
            else:
                # If TP/SL is OFF, use MACD-based SELL
                if (current_macd < current_signal and 
                    previous_macd is not None and 
                    previous_signal is not None and
                    previous_macd >= previous_signal):
                    data_copy.at[i, 'action'] = 'sell'
                    in_position = False
                    buy_price = None
    
    return data_copy

def should_buy_live(data, params={
    'short_window': 12, 
    'long_window': 26, 
    'signal_window': 9,
    'take_profit_stop_loss': 0,
    'take_profit_pct': 0.005,
    'stop_loss_pct': 0.005
}):
    """
    Simplified live function. In practice, you'd track in_position/buy_price
    outside this function for true TP/SL logic in real-time.
    """
    if len(data) < params.get('long_window', 26):
        return ['none', 0]
    
    short_window = int(params.get('short_window', 12))
    long_window = int(params.get('long_window', 26))
    signal_window = int(params.get('signal_window', 9))
    tpsl_flag = params.get('take_profit_stop_loss', 0)
    # The TP/SL logic here is non-trivial in a "stateless" live call,
    # so we'll focus on MACD-based signals unless you manage position states externally.

    macd, signal = calculate_macd(data, short_window, long_window, signal_window)
    if len(macd) < 2:
        return ['none', 0]
    
    current_macd = macd.iloc[-1]
    current_signal = signal.iloc[-1]
    previous_macd = macd.iloc[-2]
    previous_signal = signal.iloc[-2]
    
    # If TP/SL is OFF, return MACD-based signals
    if tpsl_flag == 0:
        if current_macd > current_signal and previous_macd <= previous_signal:
            return ['buy', current_macd]
        if current_macd < current_signal and previous_macd >= previous_signal:
            return ['sell', current_macd]
        return ['none', current_macd]

    # If TP/SL is ON, we can still allow MACD-based buy,
    # but selling would be triggered by TP/SL (tracked externally).
    if current_macd > current_signal and previous_macd <= previous_signal:
        return ['buy', current_macd]
    return ['none', current_macd]

# Updated Hyperopt space: add take_profit_stop_loss, take_profit_pct, stop_loss_pct
param_space = {
    'short_window': hp.quniform('short_window', 5, 20, 1),
    'long_window': hp.quniform('long_window', 21, 50, 1),
    'signal_window': hp.quniform('signal_window', 5, 20, 1),
    'take_profit_stop_loss': hp.choice('take_profit_stop_loss', [0, 1]),
    'take_profit_pct': hp.uniform('take_profit_pct', 0.001, 0.01),
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.001, 0.01)
}

# Updated GA bounds similarly
ga_bounds = {
    'short_window': (5, 20),
    'long_window': (21, 50),
    'signal_window': (5, 20),
    'take_profit_stop_loss': (0, 1),
    'take_profit_pct': (0.001, 0.01),
    'stop_loss_pct': (0.001, 0.01)
}
