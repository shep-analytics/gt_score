import pandas as pd
from hyperopt import hp

def calculate_ema(data, short_window, long_window):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    return short_ema, long_ema

def strategy(data, 
             params={
                 'short_window': 12, 
                 'long_window': 26,
                 'take_profit_stop_loss': 0,     # 0 = off, 1 = on
                 'take_profit_pct': 0.005,       # e.g. 0.5%
                 'stop_loss_pct': 0.005          # e.g. 0.5%
             }):
    
    short_window = int(params.get('short_window', 12))
    long_window = int(params.get('long_window', 26))
    tpsl_flag = params.get('take_profit_stop_loss', 0)
    take_profit_pct = params.get('take_profit_pct', 0.005)
    stop_loss_pct = params.get('stop_loss_pct', 0.005)

    data_copy = data.copy()
    data_copy['Short EMA'], data_copy['Long EMA'] = calculate_ema(data_copy, short_window, long_window)
    data_copy['action'] = 'none'
    
    in_position = False
    buy_price = None
    
    # Loop through data to generate signals
    for i in range(1, len(data_copy)):
        current_short_ema = data_copy['Short EMA'].iloc[i-1]
        current_long_ema = data_copy['Long EMA'].iloc[i-1]
        previous_short_ema = data_copy['Short EMA'].iloc[i-2] if i >= 2 else None
        previous_long_ema = data_copy['Long EMA'].iloc[i-2] if i >= 2 else None
        close_price = data_copy['Close'].iloc[i]
        
        if not in_position:
            # Check for EMA-based BUY
            if current_short_ema > current_long_ema and previous_short_ema <= previous_long_ema:
                data_copy.at[i, 'action'] = 'buy'
                in_position = True
                buy_price = close_price
        else:
            # If TP/SL is ON, only sell when TP or SL is triggered
            if tpsl_flag == 1:
                # Take Profit condition
                if close_price >= buy_price * (1 + take_profit_pct):
                    data_copy.at[i, 'action'] = 'sell'
                    in_position = False
                    buy_price = None
                # Stop Loss condition
                elif close_price <= buy_price * (1 - stop_loss_pct):
                    data_copy.at[i, 'action'] = 'sell'
                    in_position = False
                    buy_price = None
            else:
                # If TP/SL is OFF, sell on EMA crossover
                if current_short_ema < current_long_ema and previous_short_ema >= previous_long_ema:
                    data_copy.at[i, 'action'] = 'sell'
                    in_position = False
                    buy_price = None
    
    return data_copy

def should_buy_live(data, 
                    params={
                        'short_window': 12, 
                        'long_window': 26,
                        'take_profit_stop_loss': 0,
                        'take_profit_pct': 0.005,
                        'stop_loss_pct': 0.005
                    }):
    """
    Simplified live function: returns a single action and the short EMA value.

    Note: For a true TP/SL in live trading, you'd typically need to track
    your in_position state and entry price separately. This function alone
    doesn't hold that state across calls. Below is a minimal example.
    """
    if len(data) < params.get('long_window', 26):
        return ['none', 0]
    
    short_window = int(params.get('short_window', 12))
    long_window = int(params.get('long_window', 26))
    tpsl_flag = params.get('take_profit_stop_loss', 0)
    take_profit_pct = params.get('take_profit_pct', 0.005)
    stop_loss_pct = params.get('stop_loss_pct', 0.005)
    
    recent_data = data[-long_window:]
    short_ema, long_ema = calculate_ema(recent_data, short_window, long_window)
    
    if len(short_ema) < 2:
        return ['none', 0]
    
    current_short_ema = short_ema.iloc[-1]
    current_long_ema = long_ema.iloc[-1]
    previous_short_ema = short_ema.iloc[-2]
    previous_long_ema = long_ema.iloc[-2]
    
    # If TP/SL is OFF, just do EMA-based buy/sell checks
    if tpsl_flag == 0:
        # BUY signal check
        if current_short_ema > current_long_ema and previous_short_ema <= previous_long_ema:
            return ['buy', current_short_ema]
        # SELL signal check
        if current_short_ema < current_long_ema and previous_short_ema >= previous_long_ema:
            return ['sell', current_short_ema]
        return ['none', current_short_ema]
    
    # If TP/SL is ON, this function alone won't track an open position's buy price,
    # so we can't truly check TP/SL. Realistically you'd store your in_position state 
    # and entry_price externally. Returning 'none' here, or possibly just handle 
    # the buy side, is a minimal placeholder:
    # We'll still allow the BUY to happen on EMA. SELL must be triggered externally 
    # by checking your actual position price vs. current price.
    if current_short_ema > current_long_ema and previous_short_ema <= previous_long_ema:
        return ['buy', current_short_ema]
    return ['none', current_short_ema]

# Add take-profit/stop-loss parameters to the Hyperopt param_space
param_space = {
    'short_window': hp.quniform('short_window', 5, 20, 1),
    'long_window': hp.quniform('long_window', 21, 50, 1),
    'take_profit_stop_loss': hp.choice('take_profit_stop_loss', [0, 1]),
    'take_profit_pct': hp.uniform('take_profit_pct', 0.001, 0.01),
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.001, 0.01)
}

# Add the same bounds to the GA optimization if you use it
ga_bounds = {
    'short_window': (5, 20),
    'long_window': (21, 50),
    'take_profit_stop_loss': (0, 1),
    'take_profit_pct': (0.001, 0.01),
    'stop_loss_pct': (0.001, 0.01)
}
