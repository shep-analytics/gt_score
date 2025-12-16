"""
Loss Functions for Trading Strategy Optimization

This module contains various loss functions for evaluating and optimizing
trading strategies. All functions take a backtest_results dictionary and
return a loss value (lower is better).

Loss Functions Included:
1. simple_loss_function: Negative total profit
2. sharpe_ratio_loss_function: Negative Sharpe Ratio
3. ridge_regression_loss_function: Ridge-regularized return prediction
4. elastic_net_loss_function: Elastic Net-regularized return prediction
5. sortino_ratio_loss_function: Negative Sortino Ratio (downside-focused)
6. gt_score: GT-Score (imported from gt_score module)
"""

import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

# Re-export GT-Score for convenience
from .gt_score import gt_score, gt_function


def simple_loss_function(backtest_results):
    """
    Simple loss function based on total profit/loss.
    
    The simplest possible loss function - just negate the total profit.
    Does not account for risk, consistency, or other factors.
    
    Parameters
    ----------
    backtest_results : dict
        Backtest results containing 'total_amount_of_money_made' key.
    
    Returns
    -------
    float
        Negative total profit (lower = more profitable).
    
    Notes
    -----
    This is a naive loss function and may lead to high-variance strategies.
    Use more sophisticated loss functions for production.
    """
    total_profit_loss = backtest_results['total_amount_of_money_made']
    return -total_profit_loss


def sharpe_ratio_loss_function(backtest_results):
    """
    Loss function based on the Sharpe Ratio.
    
    Calculates the Sharpe Ratio of the portfolio returns and returns
    its negative value (since we minimize loss).
    
    Sharpe Ratio = E[R] / σ(R)
    
    Parameters
    ----------
    backtest_results : dict
        Backtest results containing 'portfolio_values_over_time' list.
    
    Returns
    -------
    float
        Negative Sharpe Ratio (lower = better risk-adjusted returns).
    
    Notes
    -----
    This is a simplified Sharpe ratio that does not subtract the risk-free rate.
    For more accurate calculations, use an annualized version with risk-free rate.
    """
    values = [d['value'] for d in backtest_results['portfolio_values_over_time']]
    if len(values) < 2:
        return 0.0

    returns = np.diff(values) / values[:-1]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0.0
    return -sharpe_ratio


def sortino_ratio_loss_function(backtest_results, target_return=0.0):
    """
    Loss function based on the Sortino Ratio.
    
    Similar to Sharpe Ratio but only penalizes downside volatility,
    making it more appropriate for strategies that may have positive skew.
    
    Sortino Ratio = (E[R] - target) / σd
    
    Where σd is the downside deviation (std of returns below target).
    
    Parameters
    ----------
    backtest_results : dict
        Backtest results containing 'portfolio_values_over_time' list.
    target_return : float, default=0.0
        The target or minimum acceptable return. Returns below this
        are considered "downside" volatility.
    
    Returns
    -------
    float
        Negative Sortino Ratio (lower = better downside-adjusted returns).
    
    References
    ----------
    Sortino, F.A. & van der Meer, R. (1991). "Downside Risk". 
    Journal of Portfolio Management.
    """
    values = [d['value'] for d in backtest_results['portfolio_values_over_time']]
    if len(values) < 2:
        return 0.0

    returns = np.diff(values) / values[:-1]
    mean_return = np.mean(returns)
    
    # Calculate downside deviation (only negative returns below target)
    downside_returns = [r - target_return for r in returns if r < target_return]
    
    if len(downside_returns) == 0:
        # No downside returns - perfect (but avoid division by zero)
        downside_deviation = 1e-6
    else:
        downside_deviation = np.sqrt(np.mean([r**2 for r in downside_returns]))
    
    sortino_ratio = (mean_return - target_return) / downside_deviation if downside_deviation != 0 else 0.0
    return -sortino_ratio


def ridge_regression_loss_function(backtest_results, objective='profit'):
    """
    Loss function using Ridge Regression for return prediction.
    
    Uses Ridge regression on lagged returns to either minimize
    prediction error (MSE) or maximize simulated profit from
    predicted return directions.
    
    Parameters
    ----------
    backtest_results : dict
        Backtest results containing 'portfolio_values_over_time' list.
    objective : str, default='profit'
        Either 'mse' (minimize prediction error) or 'profit' 
        (maximize profit from predicted directions).
    
    Returns
    -------
    float
        Loss value based on chosen objective.
    
    Raises
    ------
    ValueError
        If not enough data to create lagged features (needs at least 7 data points).
    """
    values = [d['value'] for d in backtest_results['portfolio_values_over_time']]
    if len(values) < 2:
        raise ValueError("Not enough data to compute returns for Ridge regression.")

    returns = np.diff(values) / values[:-1]
    min_length = len(returns) - 5
    if min_length <= 0:
        raise ValueError("Not enough data to create lagged features for Ridge regression.")
    
    # Create lagged features (5 lags)
    X = np.hstack([returns[i : i + min_length].reshape(-1, 1) for i in range(5)])
    y = returns[5 : 5 + min_length]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    if objective == 'mse':
        mse = np.mean((y - y_pred) ** 2)
        return mse
    elif objective == 'profit':
        profit = np.sum(np.sign(y_pred) * y)
        return -profit
    else:
        raise ValueError(f"Unknown objective '{objective}'")


def elastic_net_loss_function(backtest_results, objective='profit'):
    """
    Loss function using Elastic Net Regression for return prediction.
    
    Similar to Ridge, but uses Elastic Net regularization (L1 + L2)
    which can provide feature selection alongside regularization.
    
    Parameters
    ----------
    backtest_results : dict
        Backtest results containing 'portfolio_values_over_time' list.
    objective : str, default='profit'
        Either 'mse' (minimize prediction error) or 'profit' 
        (maximize profit from predicted directions).
    
    Returns
    -------
    float
        Loss value based on chosen objective.
    
    Raises
    ------
    ValueError
        If not enough data to create lagged features (needs at least 7 data points).
    """
    values = [d['value'] for d in backtest_results['portfolio_values_over_time']]
    if len(values) < 2:
        raise ValueError("Not enough data to compute returns for Elastic Net regression.")

    returns = np.diff(values) / values[:-1]
    min_length = len(returns) - 5
    if min_length <= 0:
        raise ValueError("Not enough data to create lagged features for Elastic Net regression.")
    
    # Create lagged features (5 lags)
    X = np.hstack([returns[i : i + min_length].reshape(-1, 1) for i in range(5)])
    y = returns[5 : 5 + min_length]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    en = ElasticNet(alpha=0.5, l1_ratio=0.5)
    en.fit(X_scaled, y)

    y_pred = en.predict(X_scaled)

    if objective == 'mse':
        mse = np.mean((y - y_pred) ** 2)
        return mse
    elif objective == 'profit':
        profit = np.sum(np.sign(y_pred) * y)
        return -profit
    else:
        raise ValueError(f"Unknown objective '{objective}'")


# Dictionary for easy access to all loss functions
LOSS_FUNCTIONS = {
    'simple': simple_loss_function,
    'sharpe': sharpe_ratio_loss_function,
    'sortino': sortino_ratio_loss_function,
    'ridge': ridge_regression_loss_function,
    'elastic_net': elastic_net_loss_function,
    'gt_score': gt_score,
}


def get_loss_function(name):
    """
    Get a loss function by name.
    
    Parameters
    ----------
    name : str
        Name of the loss function. One of: 'simple', 'sharpe', 'sortino',
        'ridge', 'elastic_net', 'gt_score'.
    
    Returns
    -------
    callable
        The loss function.
    
    Raises
    ------
    ValueError
        If the loss function name is unknown.
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_FUNCTIONS.keys())}")
    return LOSS_FUNCTIONS[name]
