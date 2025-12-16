"""
Trading Strategies Package

This package contains various technical analysis-based trading strategies
including RSI, MACD, Bollinger Bands, SMA, EMA, Ichimoku, Donchian Channels,
Elliott Wave, and Parabolic SAR.

Each strategy module exports:
- strategy(ohlc_data, params): Generate trading signals
- should_buy_live(ohlc_data, params): Check if should buy now (live trading)
- param_space: Hyperopt parameter space
- ga_bounds: Genetic algorithm parameter bounds
"""

from .import_all import strategies, get_strategy_by_name, list_strategy_names

__all__ = ['strategies', 'get_strategy_by_name', 'list_strategy_names']
