"""
GT-Score (Golden-Ticket Score) Implementation

This module implements the GT-Score loss function for evaluating trading strategies
by measuring excess return, return consistency (R²), and downside risk (σd).

The GT-Score combines multiple desirable properties:
- Higher average returns (μ) are rewarded
- Statistically significant outperformance over buy-and-hold (z-score)
- Consistent equity curve growth (R²)
- Lower downside volatility (σd) is preferred

Mathematical Definition:
    GT-Score = (μ × ln(z) × R²) / σd

Where:
- μ = mean return per period
- z = (μ - μ_market) / (σ / √N) is the z-score for excess returns
- R² = coefficient of determination of returns over time
- σd = standard deviation of negative returns (downside deviation)

Edge Case Handling (Piecewise Definition):
    Case 1: z ≤ 0 (underperforms buy-and-hold)
        Score = 100 + 100 × (1 - exp(-|z - 1|))
        Returns a large penalty (>100) that increases with worse performance.
        
    Case 2: 0 < z ≤ 1 (marginal outperformance)  
        Score = 100 × (1 - exp(-|z - 1|))
        Returns a smooth transition score between 0 and 63.2.
        
    Case 3: z > 1 (statistically significant outperformance)
        Score = -(μ × ln(z) × R²) / σd
        Returns negative GT-Score (negated for minimization).

Smoothing Parameters:
- ε = 1e-6 for σd to prevent division by zero when no negative returns exist
- N_periods default = 50 (corresponding to ~6 trades per year over 8 years)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
import math


def find_stabilized_variance(data, min_period=20, max_period=100):
    """
    Find the number of periods where the variance of returns stabilizes naturally.
    
    This function implements a binary search to find the optimal number of periods
    for dividing the equity curve where variance becomes stable. This is used to
    set the N_periods parameter adaptively rather than using a fixed value.
    
    The stabilization criterion is:
        ΔσN ≤ 0.01 × mean(recent_variances)
    
    Where ΔσN is the average change in variance between consecutive period counts.
    
    Parameters
    ----------
    data : list of dict
        List of dictionaries with 'date_time' (Timestamp) and 'value' (float).
        Typically from backtest_results['portfolio_values_over_time'].
    min_period : int, default=20
        Minimum number of periods to consider.
    max_period : int, default=100
        Maximum number of periods to consider.
    
    Returns
    -------
    int
        Optimal number of periods where variance stabilizes.
        Returns 50 as default if no stabilization is found within the range.
    
    Notes
    -----
    The algorithm works by:
    1. Using binary search between min_period and max_period
    2. For each candidate period count, computing the variance of returns
    3. Checking if the last 4 variance values have stabilized (≤1% change)
    4. Returning the first period count where stabilization occurs
    """
    df = pd.DataFrame(data)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.sort_values('date_time', inplace=True)

    total_time = (df['date_time'].max() - df['date_time'].min()).days
    results = []
    
    # Binary search to optimize the number of periods
    low = min_period
    high = max_period
    while low <= high:
        num_periods = (low + high) // 2
        period_length = total_time / num_periods
        df['period'] = ((df['date_time'] - df['date_time'].min()).dt.days // period_length).astype(int)

        # Compute returns per period
        returns = df.groupby('period')['value'].apply(
            lambda g: (g.iloc[-1] - g.iloc[0]) / g.iloc[0] if len(g) > 1 else None
        ).dropna()

        if len(returns) < 2:
            low = num_periods + 1  # Not enough data for this split, go larger
            continue

        variance = np.var(returns)
        results.append((num_periods, variance))

        # Check stabilization by comparing recent variances
        if len(results) > 3:
            recent_variances = [v[1] for v in results[-4:]]  # Last 4 variances
            changes = [abs(recent_variances[i] - recent_variances[i - 1]) for i in range(1, len(recent_variances))]
            avg_change = np.mean(changes)
            
            # If the variance change has plateaued, return num_periods
            if avg_change <= np.mean(recent_variances) * 0.01:  # 1% of the average variance
                return num_periods

        # Adjust search range
        if len(results) > 1 and variance < results[-2][1]:
            high = num_periods - 1  # Variance decreasing, search smaller periods
        else:
            low = num_periods + 1  # Variance increasing, search larger periods

    # If stabilization is not found, return 50 as the default value
    return 50


def get_period_returns(data, num_periods):
    """
    Calculate portfolio and market returns for a given number of periods.
    
    Divides the equity curve into equal-length periods and computes the
    return for each period as (end_value - start_value) / start_value.
    
    Parameters
    ----------
    data : list of dict
        List of dictionaries with 'date_time', 'value', and 'stock_value' keys.
    num_periods : int
        Number of periods to divide the data into.
    
    Returns
    -------
    tuple of (list, list)
        - portfolio_returns: List of portfolio returns per period
        - market_returns: List of market (buy-and-hold) returns per period
    """
    df = pd.DataFrame(data)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.sort_values('date_time', inplace=True)

    # Calculate total days and period length in days
    total_time = (df['date_time'].max() - df['date_time'].min()).days
    period_length = total_time / num_periods
    
    # Assign each row to a period based on integer division
    df['period'] = ((df['date_time'] - df['date_time'].min()).dt.days // period_length).astype(int)

    # Compute portfolio returns
    portfolio_returns = (
        df.groupby('period')['value']
        .apply(lambda g: (g.iloc[-1] - g.iloc[0]) / g.iloc[0] if len(g) > 1 else None)
        .dropna()
        .tolist()
    )

    # Compute market (stock) returns
    market_returns = (
        df.groupby('period')['stock_value']
        .apply(lambda g: (g.iloc[-1] - g.iloc[0]) / g.iloc[0] if len(g) > 1 else None)
        .dropna()
        .tolist()
    )

    return portfolio_returns, market_returns


def gt_score(backtest_results, stabilize=False, mode="trades"):
    """
    Calculate the GT-Score (Golden-Ticket Score) for a backtest.
    
    The GT-Score is a composite loss function that rewards:
    - Higher average returns (μ)
    - Statistically significant excess returns over buy-and-hold (ln(z))
    - Consistent equity curve growth (R²)
    - Lower downside volatility (σd)
    
    Parameters
    ----------
    backtest_results : dict
        Dictionary containing backtest results with keys:
        - 'portfolio_values_over_time': List of {date_time, value, stock_value}
        - 'trades_history': List of trade records with 'profit_loss_percent'
    stabilize : bool, default=False
        If True, automatically find optimal N_periods using variance stabilization.
        If False, use the default N_periods=50.
    mode : str, default="trades"
        Either "trades" (use individual trade returns) or "portfolio_value" 
        (use period-based portfolio returns).
    
    Returns
    -------
    float
        The GT-Score value (lower is better for optimization):
        - Large positive (>100): Poor performance, underperforms buy-and-hold
        - Small positive (0-100): Marginal performance
        - Negative: Good performance (more negative = better)
    
    Notes
    -----
    Piecewise Definition for Edge Cases:
    
    1. z ≤ 0 (strategy underperforms buy-and-hold):
       Returns 100 + 100 × (1 - exp(-|z - 1|))
       This gives a penalty score >100 that increases with worse performance.
    
    2. 0 < z ≤ 1 (marginal outperformance, not statistically significant):
       Returns 100 × (1 - exp(-|z - 1|))
       Smooth transition score between 0 and ~63.2.
    
    3. z > 1 (statistically significant outperformance):
       Returns -(μ × ln(z) × R²) / σd
       The negative sign converts to minimization objective.
    
    The σd (downside deviation) uses ε = 1e-6 smoothing when no negative 
    returns exist, preventing division by zero.
    
    Default N_periods = 50: For an 8-year backtest, this corresponds to
    approximately 6 trades per year, which is a reasonable assumption for
    medium-frequency trading strategies.
    
    Examples
    --------
    >>> results = run_backtest(trading_signals)
    >>> score = gt_score(results)
    >>> print(f"GT-Score: {score:.4f}")
    
    >>> # With variance stabilization
    >>> score = gt_score(results, stabilize=True)
    
    References
    ----------
    - Kestner (1996): R² as equity curve smoothness measure
    - Sortino & van der Meer (1991): Downside deviation (σd)
    """
    # Determine the number of periods
    if stabilize:
        num_periods = find_stabilized_variance(backtest_results['portfolio_values_over_time'])
    else:
        num_periods = 50
    
    # Get the number of trades
    num_trades = len(backtest_results["trades_history"])
    
    # Edge case: not enough trades
    if num_trades <= num_periods:
        # Push the score towards more trades
        # Best score possible with under periods is 100
        interval = (999 - 100) / num_periods
        return 999 - (num_trades * interval)
    
    # Get the returns from each trade
    percentage_returns_by_trade = [
        trade["profit_loss_percent"] 
        for trade in backtest_results["trades_history"]
    ]
    
    if mode == "portfolio_value":
        period_percentage_returns, period_percentage_returns_market = get_period_returns(
            backtest_results['portfolio_values_over_time'], 
            num_periods
        )
    elif mode == "trades":
        period_percentage_returns = percentage_returns_by_trade
        period_percentage_returns_market = []
        
        # Calculate market mean return based on same number of trades
        starting_market_value = backtest_results["portfolio_values_over_time"][0]["stock_value"]
        ending_market_value = backtest_results["portfolio_values_over_time"][-1]["stock_value"]
        mean_market_return = (ending_market_value / starting_market_value) ** (1 / num_trades) - 1
        
        for _ in range(num_trades):
            period_percentage_returns_market.append(mean_market_return)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'trades' or 'portfolio_value'.")
    
    # Calculate necessary values
    mu = np.mean(period_percentage_returns)  # Mean return
    mum = np.mean(period_percentage_returns_market)  # Mean market return
    
    # R² (coefficient of determination) for equity curve smoothness
    r2 = linregress(range(len(percentage_returns_by_trade)), percentage_returns_by_trade).rvalue ** 2
    
    # Downside deviation (σd) with smoothing parameter ε = 1e-6
    negative_returns = [r for r in period_percentage_returns if r < 0]
    sigma_d = np.std(negative_returns) if negative_returns else 1e-6  # ε smoothing
    
    # Standard deviation of returns
    sigma = np.std(period_percentage_returns)
    
    # Calculate z-score for excess returns
    z = (mu - mum) / (sigma / np.sqrt(num_trades))
    
    # Piecewise GT-Score definition
    if z <= 0:
        # Case 1: Underperforms buy-and-hold (large penalty)
        score = 100 + (100 * (1 - math.exp(-abs(z - 1))))
        return score
    elif z <= 1:
        # Case 2: Marginal outperformance (smooth transition)
        score = 100 * (1 - math.exp(-abs(z - 1)))
        return score
    else:
        # Case 3: Statistically significant outperformance
        ln_z = math.log(z)
        gt = (mu * ln_z * r2) / sigma_d
        # Negate for minimization (more negative = better)
        return -gt


# Alias for backward compatibility
gt_function = gt_score
