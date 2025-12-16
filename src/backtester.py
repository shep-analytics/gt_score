"""
Backtesting Engine for Trading Strategies

This module provides a simple event-driven backtester for evaluating
trading strategies based on buy/sell signals.

Features:
- Supports commission and spread costs
- Tracks portfolio value over time
- Records detailed trade history
- Calculates various performance metrics
"""

import copy
from datetime import timedelta

import pandas as pd


def run_backtest(trading_signals_original, starting_cash=1000000, 
                 commission=0.0001, spread=0.0001):
    """
    Run a backtest based on buy and sell signals.
    
    Parameters
    ----------
    trading_signals : DataFrame
        DataFrame containing signals with columns:
        - 'Date': Timestamp of the signal
        - 'action': 'buy', 'sell', or 'hold'
        - 'Close': Closing price at the signal time
    starting_cash : float, default=1,000,000
        Initial cash balance.
    commission : float, default=0.0001 (0.01%)
        Commission rate as a decimal (applied to trade value).
    spread : float, default=0.0001
        Price spread in dollars (added to buy, subtracted from sell).
    
    Returns
    -------
    tuple of (dict, DataFrame)
        - dict: Performance metrics and trade history
        - DataFrame: Original signals with portfolio values attached
    
    Notes
    -----
    Transaction costs are supported but set to minimal defaults.
    For realistic simulations, adjust commission and spread parameters.
    
    The backtester assumes full position entry (100% of cash) on each buy
    and full exit on each sell. Partial positions are not supported.
    
    Examples
    --------
    >>> signals = pd.DataFrame({
    ...     'Date': pd.date_range('2020-01-01', periods=100),
    ...     'Close': np.random.randn(100).cumsum() + 100,
    ...     'action': ['buy', 'hold'] * 25 + ['sell', 'hold'] * 25
    ... })
    >>> results, signals_with_values = run_backtest(signals)
    >>> print(f"Total return: {results['total_percentage_gain']:.2%}")
    """
    cash = starting_cash
    position = 0
    entry_price = 0
    entry_time = None
    trades_history = []
    portfolio_values = []

    trading_signals = copy.deepcopy(trading_signals_original)

    # Ensure Date is sorted and in datetime format
    trading_signals['Date'] = pd.to_datetime(trading_signals['Date'])
    trading_signals = trading_signals.sort_values('Date').reset_index(drop=True)

    for index, row in trading_signals.iterrows():
        action = row['action']
        price = row['Close']
        time = row['Date']

        if action == 'buy' and position == 0:
            buy_price = price + spread
            buy_cost = cash * (1 - commission)
            position = buy_cost / buy_price
            cash -= buy_cost
            entry_price = buy_price
            entry_time = time

        elif action == 'sell' and position > 0:
            sell_price = price - spread
            sell_revenue = position * sell_price * (1 - commission)
            cash += sell_revenue

            profit = sell_revenue - (position * entry_price)
            profit_percent = profit / (position * entry_price)
            time_held = time - entry_time

            trades_history.append({
                'purchase_price': entry_price,
                'sale_price': sell_price,
                'purchase_date': entry_time,
                'sale_date': time,
                'profit_loss_percent': profit_percent,
                'profit_loss_dollars': profit,
                'time_held': time_held
            })

            position = 0
            entry_price = 0
            entry_time = None

        # Record portfolio value at each row
        current_value = cash + (position * price if position > 0 else 0)
        portfolio_values.append({
            "date_time": time, 
            "value": current_value, 
            "stock_value": row['Close']
        })

    # Close any open position at the end
    if position > 0:
        sell_price = trading_signals.iloc[-1]['Close'] - spread
        sell_revenue = position * sell_price * (1 - commission)
        cash += sell_revenue
        profit = sell_revenue - (position * entry_price)
        profit_percent = profit / (position * entry_price)
        time_held = trading_signals.iloc[-1]['Date'] - entry_time

        trades_history.append({
            'purchase_price': entry_price,
            'sale_price': sell_price,
            'purchase_date': entry_time,
            'sale_date': trading_signals.iloc[-1]['Date'],
            'profit_loss_percent': profit_percent,
            'profit_loss_dollars': profit,
            'time_held': time_held
        })

    # Calculate metrics
    total_trades = len(trades_history)
    total_money_made = cash - starting_cash
    total_percentage_gain = total_money_made / starting_cash if starting_cash else 0

    total_days = (trading_signals['Date'].iloc[-1] - trading_signals['Date'].iloc[0]).days or 1

    # Compute average hold time
    time_held_list = [trade['time_held'] for trade in trades_history]
    if time_held_list:
        average_time_holding_position = sum(time_held_list, timedelta()) / len(time_held_list)
        longest_time_position_held = max(time_held_list)
    else:
        average_time_holding_position = timedelta(0)
        longest_time_position_held = timedelta(0)

    # Average gains per trade
    if total_trades > 0:
        avg_gain_percent = sum(t['profit_loss_percent'] for t in trades_history) / total_trades
        avg_gain_dollars = sum(t['profit_loss_dollars'] for t in trades_history) / total_trades
    else:
        avg_gain_percent = 0
        avg_gain_dollars = 0

    # Annualized returns
    years_in_data = total_days / 365.0
    average_yearly_percentage_gain = (total_percentage_gain / years_in_data) if years_in_data > 0 else 0
    average_monthly_percentage_gain = average_yearly_percentage_gain / 12

    # Attach portfolio values to DataFrame
    trading_signals['portfolio_value'] = [p['value'] for p in portfolio_values]

    return {
        'trades_history': trades_history,
        'total_trades': total_trades,
        'final_cash': cash,
        'total_amount_of_money_made': total_money_made,
        'total_percentage_gain': total_percentage_gain,
        'average_time_holding_position': average_time_holding_position,
        'longest_time_position_held': longest_time_position_held,
        'average_gain_percent_per_trade': avg_gain_percent,
        'average_gain_dollars_per_trade': avg_gain_dollars,
        'average_yearly_percentage_gain': average_yearly_percentage_gain,
        'average_monthly_percentage_gain': average_monthly_percentage_gain,
        'portfolio_values_over_time': portfolio_values,
        'trading_signals': trading_signals
    }, trading_signals
