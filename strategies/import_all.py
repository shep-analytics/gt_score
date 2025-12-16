"""
Strategy Import Module

This module provides convenient access to all trading strategies
with their default parameters and optimization bounds.
"""

from .RSI_Strategy import (
    strategy as RSI_strategy, 
    should_buy_live as RSI_should_buy_live, 
    param_space as RSI_param_space, 
    ga_bounds as RSI_ga_bounds
)
from .MACD_Strategy import (
    strategy as MACD_strategy, 
    should_buy_live as MACD_should_buy_live, 
    param_space as MACD_param_space, 
    ga_bounds as MACD_ga_bounds
)
from .BollingerBands_Strategy import (
    strategy as BollingerBands_strategy, 
    should_buy_live as BollingerBands_should_buy_live, 
    param_space as BollingerBands_param_space, 
    ga_bounds as BollingerBands_ga_bounds
)
from .SMA_Strategy import (
    strategy as SMA_strategy, 
    should_buy_live as SMA_should_buy_live, 
    param_space as SMA_param_space, 
    ga_bounds as SMA_ga_bounds
)
from .EMA_Strategy import (
    strategy as EMA_strategy, 
    should_buy_live as EMA_should_buy_live, 
    param_space as EMA_param_space, 
    ga_bounds as EMA_ga_bounds
)
from .IchimokuCloud_Strategy import (
    strategy as Ichimoku_strategy, 
    should_buy_live as Ichimoku_should_buy_live, 
    param_space as Ichimoku_param_space, 
    ga_bounds as Ichimoku_ga_bounds
)
from .DonchianChannel_Strategy import (
    strategy as Donchian_strategy, 
    should_buy_live as Donchian_should_buy_live, 
    param_space as Donchian_param_space, 
    ga_bounds as Donchian_ga_bounds
)
from .SmallMACrossover_Strategy import (
    strategy as SmallMACrossover_strategy, 
    should_buy_live as SmallMACrossover_should_buy_live, 
    param_space as SmallMACrossover_param_space, 
    ga_bounds as SmallMACrossover_ga_bounds
)
from .ElliottWave_Strategy import (
    strategy as ElliottWave_strategy, 
    should_buy_live as ElliottWave_should_buy_live, 
    param_space as ElliottWave_param_space, 
    ga_bounds as ElliottWave_ga_bounds
)
from .ParabolicSAR_Strategy import (
    strategy as ParabolicSAR_strategy, 
    should_buy_live as ParabolicSAR_should_buy_live, 
    param_space as ParabolicSAR_param_space, 
    ga_bounds as ParabolicSAR_ga_bounds
)


# All strategies with their configurations
strategies = [
    {
        'strategy': RSI_strategy,
        'name': 'RSI',
        'params': {'rsi_buy_threshold': 25, 'rsi_sell_threshold': 75, 'window': 14},
        'param_space': RSI_param_space,
        'ga_bounds': RSI_ga_bounds
    },
    {
        'strategy': MACD_strategy,
        'name': 'MACD',
        'params': {
            'short_window': 12,
            'long_window': 26,
            'signal_window': 9,
            'take_profit_stop_loss': 0,
            'take_profit_pct': 0.005,
            'stop_loss_pct': 0.005
        },
        'param_space': MACD_param_space,
        'ga_bounds': MACD_ga_bounds
    },
    {
        'strategy': BollingerBands_strategy,
        'name': 'BollingerBands',
        'params': {'window': 20, 'num_std_dev': 2},
        'param_space': BollingerBands_param_space,
        'ga_bounds': BollingerBands_ga_bounds
    },
    {
        'strategy': SMA_strategy,
        'name': 'SMA',
        'params': {'short_window': 10, 'long_window': 50},
        'param_space': SMA_param_space,
        'ga_bounds': SMA_ga_bounds
    },
    {
        'strategy': EMA_strategy,
        'name': 'EMA',
        'params': {
            'short_window': 12,
            'long_window': 26,
            'take_profit_stop_loss': 1,
            'take_profit_pct': 0.005,
            'stop_loss_pct': 0.005
        },
        'param_space': EMA_param_space,
        'ga_bounds': EMA_ga_bounds
    },
    {
        'strategy': Donchian_strategy,
        'name': 'Donchian',
        'params': {'window': 20},
        'param_space': Donchian_param_space,
        'ga_bounds': Donchian_ga_bounds
    },
    {
        'strategy': Ichimoku_strategy,
        'name': 'Ichimoku',
        'params': {'tenkan_window': 9, 'kijun_window': 26, 'senkou_span_b_window': 52},
        'param_space': Ichimoku_param_space,
        'ga_bounds': Ichimoku_ga_bounds
    },
    {
        'strategy': SmallMACrossover_strategy,
        'name': 'SmallMACrossover',
        'params': {'short_window': 5, 'long_window': 10},
        'param_space': SmallMACrossover_param_space,
        'ga_bounds': SmallMACrossover_ga_bounds
    },
    {
        'strategy': ElliottWave_strategy,
        'name': 'ElliottWave',
        'params': {'window': 20},
        'param_space': ElliottWave_param_space,
        'ga_bounds': ElliottWave_ga_bounds
    },
    {
        'strategy': ParabolicSAR_strategy,
        'name': 'ParabolicSAR',
        'params': {'af_start': 0.02, 'af_step': 0.02, 'af_max': 0.2},
        'param_space': ParabolicSAR_param_space,
        'ga_bounds': ParabolicSAR_ga_bounds
    }
]


def get_strategy_by_name(name):
    """Get a strategy configuration by name."""
    for s in strategies:
        if s['name'].lower() == name.lower():
            return s
    raise ValueError(f"Unknown strategy: {name}")


def list_strategy_names():
    """List all available strategy names."""
    return [s['name'] for s in strategies]