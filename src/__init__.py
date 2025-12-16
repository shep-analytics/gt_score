"""
GT-Score: A Composite Loss Function for Trading Strategy Optimization

This package provides the GT-Score loss function and supporting tools for
evaluating and optimizing quantitative trading strategies with emphasis on
reducing overfitting.

Modules
-------
gt_score
    Core GT-Score implementation
loss_functions
    Multiple loss functions for comparison
backtester
    Event-driven backtesting engine
optimizers
    Random, Hyperopt, and Genetic Algorithm optimizers
walkforward
    Walk-forward validation framework
statistics
    Statistical tests and metrics
"""

from .gt_score import gt_score, gt_function, find_stabilized_variance, get_period_returns
from .loss_functions import (
    simple_loss_function,
    sharpe_ratio_loss_function,
    sortino_ratio_loss_function,
    ridge_regression_loss_function,
    elastic_net_loss_function,
    LOSS_FUNCTIONS,
    get_loss_function
)
from .backtester import run_backtest
from .optimizers import optimize, compile_backtest_results_sequential
from .walkforward import (
    generate_walkforward_splits,
    count_walkforward_splits,
    WalkForwardValidator
)
from .statistics import (
    paired_ttest,
    bootstrap_ci,
    cohens_d,
    overfitting_ratio,
    probability_of_backtest_overfitting,
    multiple_testing_adjustment,
    summary_statistics
)

__version__ = "1.0.0"
__author__ = "GT-Score Research Team"

__all__ = [
    # GT-Score
    'gt_score',
    'gt_function',
    'find_stabilized_variance',
    'get_period_returns',
    # Loss functions
    'simple_loss_function',
    'sharpe_ratio_loss_function',
    'sortino_ratio_loss_function',
    'ridge_regression_loss_function',
    'elastic_net_loss_function',
    'LOSS_FUNCTIONS',
    'get_loss_function',
    # Backtesting
    'run_backtest',
    # Optimization
    'optimize',
    'compile_backtest_results_sequential',
    # Walk-forward
    'generate_walkforward_splits',
    'count_walkforward_splits',
    'WalkForwardValidator',
    # Statistics
    'paired_ttest',
    'bootstrap_ci',
    'cohens_d',
    'overfitting_ratio',
    'probability_of_backtest_overfitting',
    'multiple_testing_adjustment',
    'summary_statistics',
]
