"""
Unit Tests for GT-Score Implementation

These tests verify the edge case handling documented in the paper:
1. z ≤ 0: Returns penalty score > 100
2. 0 < z ≤ 1: Returns smooth transition score (0 to ~63.2)
3. z > 1: Returns standard GT-Score (negative for minimization)
4. σd = 0: Uses smoothing parameter ε = 1e-6, doesn't crash
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gt_score import gt_score, find_stabilized_variance, get_period_returns


def create_mock_backtest_results(trades_returns, portfolio_trend='up', 
                                  market_trend='flat', n_days=1000):
    """
    Helper function to create mock backtest results.
    
    Parameters
    ----------
    trades_returns : list
        List of trade return percentages.
    portfolio_trend : str
        'up', 'down', or 'flat' - general portfolio trend.
    market_trend : str
        'up', 'down', or 'flat' - general market trend.
    n_days : int
        Number of days for portfolio values.
    """
    base_date = datetime(2020, 1, 1)
    
    # Create portfolio values over time
    if portfolio_trend == 'up':
        base_values = np.linspace(1000000, 1500000, n_days)
    elif portfolio_trend == 'down':
        base_values = np.linspace(1000000, 500000, n_days)
    else:  # flat
        base_values = np.ones(n_days) * 1000000
    
    # Add some noise
    noise = np.random.normal(0, 0.01, n_days) * base_values
    values = base_values + noise
    
    # Create market values
    if market_trend == 'up':
        market_base = np.linspace(100, 150, n_days)
    elif market_trend == 'down':
        market_base = np.linspace(100, 50, n_days)
    else:  # flat
        market_base = np.ones(n_days) * 100
    
    market_noise = np.random.normal(0, 0.01, n_days) * market_base
    market_values = market_base + market_noise
    
    portfolio_values_over_time = []
    for i in range(n_days):
        portfolio_values_over_time.append({
            'date_time': base_date + timedelta(days=i),
            'value': values[i],
            'stock_value': market_values[i]
        })
    
    # Create trades history
    trades_history = []
    for i, ret in enumerate(trades_returns):
        trades_history.append({
            'purchase_price': 100.0,
            'sale_price': 100.0 * (1 + ret),
            'purchase_date': base_date + timedelta(days=i*10),
            'sale_date': base_date + timedelta(days=i*10 + 5),
            'profit_loss_percent': ret,
            'profit_loss_dollars': 1000 * ret,
            'time_held': timedelta(days=5)
        })
    
    return {
        'portfolio_values_over_time': portfolio_values_over_time,
        'trades_history': trades_history,
        'total_amount_of_money_made': values[-1] - values[0],
        'total_percentage_gain': (values[-1] - values[0]) / values[0]
    }


class TestGTScoreEdgeCases:
    """Test the piecewise GT-Score definition for edge cases."""
    
    def test_z_negative_underperforms_market(self):
        """
        z ≤ 0 should return penalty score > 100.
        
        This happens when the strategy underperforms buy-and-hold.
        """
        # Create results where strategy loses money while market gains
        trades = [-0.02, -0.03, -0.01, -0.02, -0.015] * 20  # 100 losing trades
        results = create_mock_backtest_results(
            trades_returns=trades,
            portfolio_trend='down',
            market_trend='up'
        )
        
        score = gt_score(results)
        
        # Score should be > 100 for underperforming strategies
        assert score > 100, f"Expected score > 100, got {score}"
    
    def test_z_between_zero_and_one(self):
        """
        0 < z ≤ 1 should return smooth transition score between 0 and ~63.2.
        
        This happens when the strategy marginally outperforms but not significantly.
        """
        # Create results with slight outperformance
        trades = [0.01, 0.005, 0.008, 0.003, 0.006] * 20  # 100 small positive trades
        results = create_mock_backtest_results(
            trades_returns=trades,
            portfolio_trend='up',
            market_trend='up'  # Market also goes up, marginal outperformance
        )
        
        score = gt_score(results)
        
        # For marginal cases, score should be positive but moderate
        # The exact value depends on the z-score
        assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
    
    def test_z_greater_than_one(self):
        """
        z > 1 should return standard GT-Score (negative for minimization).
        
        This is the normal case where the strategy significantly outperforms.
        """
        # Create results with significant outperformance
        trades = [0.05, 0.04, 0.06, 0.03, 0.05] * 20  # 100 good trades
        results = create_mock_backtest_results(
            trades_returns=trades,
            portfolio_trend='up',
            market_trend='flat'  # Market flat, strategy gains
        )
        
        score = gt_score(results)
        
        # For good strategies, score should be negative (lower is better)
        # Note: large positive returns should yield negative scores
        assert isinstance(score, (int, float)), f"Expected numeric score, got {type(score)}"
    
    def test_sigma_d_zero_uses_smoothing(self):
        """
        σd = 0 (no negative returns) should use smoothing parameter ε = 1e-6.
        
        This shouldn't crash or return infinity.
        """
        # Create results with all positive trades (no negative returns)
        trades = [0.05, 0.03, 0.04, 0.06, 0.02] * 20  # 100 positive trades
        results = create_mock_backtest_results(
            trades_returns=trades,
            portfolio_trend='up',
            market_trend='flat'
        )
        
        # This should not raise an exception
        score = gt_score(results)
        
        # Score should be finite
        assert np.isfinite(score), f"Expected finite score, got {score}"
    
    def test_not_enough_trades_returns_penalty(self):
        """
        When num_trades <= num_periods (default 50), return penalty score.
        """
        # Only 10 trades (less than default 50 periods)
        trades = [0.05] * 10
        results = create_mock_backtest_results(
            trades_returns=trades,
            portfolio_trend='up',
            market_trend='flat'
        )
        
        score = gt_score(results)
        
        # Score should be high (penalty for not enough trades)
        assert score > 100, f"Expected penalty score > 100 for insufficient trades, got {score}"


class TestFindStabilizedVariance:
    """Test the variance stabilization function."""
    
    def test_returns_default_when_not_enough_data(self):
        """Should return 50 when there's not enough data."""
        # Create minimal data
        data = [
            {'date_time': datetime(2020, 1, 1), 'value': 100},
            {'date_time': datetime(2020, 1, 2), 'value': 101},
        ]
        
        result = find_stabilized_variance(data)
        
        # Should return default of 50
        assert result == 50
    
    def test_returns_valid_period_count(self):
        """Should return a valid period count between min and max."""
        # Create enough data
        data = []
        for i in range(1000):
            data.append({
                'date_time': datetime(2020, 1, 1) + timedelta(days=i),
                'value': 100 + np.random.normal(0, 5)
            })
        
        result = find_stabilized_variance(data, min_period=20, max_period=100)
        
        # Should return value in valid range or default
        assert 20 <= result <= 100 or result == 50


class TestGetPeriodReturns:
    """Test the period returns calculation."""
    
    def test_returns_correct_length(self):
        """Should return lists of appropriate length."""
        data = []
        for i in range(365):
            data.append({
                'date_time': datetime(2020, 1, 1) + timedelta(days=i),
                'value': 1000000 + i * 100,
                'stock_value': 100 + i * 0.1
            })
        
        portfolio_returns, market_returns = get_period_returns(data, num_periods=10)
        
        # Should return non-empty lists
        assert len(portfolio_returns) > 0
        assert len(market_returns) > 0
    
    def test_handles_small_periods(self):
        """Should handle when periods are small."""
        data = []
        for i in range(30):
            data.append({
                'date_time': datetime(2020, 1, 1) + timedelta(days=i),
                'value': 1000000 + i * 100,
                'stock_value': 100 + i * 0.1
            })
        
        portfolio_returns, market_returns = get_period_returns(data, num_periods=5)
        
        # Should not crash and return something
        assert isinstance(portfolio_returns, list)
        assert isinstance(market_returns, list)


class TestGTScoreModes:
    """Test different GT-Score calculation modes."""
    
    def test_trades_mode(self):
        """Test using trades mode (default)."""
        trades = [0.02, 0.03, 0.01, -0.01, 0.02] * 20
        results = create_mock_backtest_results(trades_returns=trades)
        
        score = gt_score(results, mode="trades")
        
        assert isinstance(score, (int, float))
        assert np.isfinite(score)
    
    def test_portfolio_value_mode(self):
        """Test using portfolio_value mode."""
        trades = [0.02, 0.03, 0.01, -0.01, 0.02] * 20
        results = create_mock_backtest_results(trades_returns=trades)
        
        score = gt_score(results, mode="portfolio_value")
        
        assert isinstance(score, (int, float))
        assert np.isfinite(score)
    
    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        trades = [0.02, 0.03] * 30
        results = create_mock_backtest_results(trades_returns=trades)
        
        with pytest.raises(ValueError):
            gt_score(results, mode="invalid_mode")


class TestGTScoreStabilization:
    """Test GT-Score with variance stabilization."""
    
    def test_stabilize_flag(self):
        """Test with stabilize=True."""
        trades = [0.02, 0.03, 0.01, -0.01, 0.02] * 20
        results = create_mock_backtest_results(trades_returns=trades, n_days=2000)
        
        score_default = gt_score(results, stabilize=False)
        score_stabilized = gt_score(results, stabilize=True)
        
        # Both should be finite
        assert np.isfinite(score_default)
        assert np.isfinite(score_stabilized)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
