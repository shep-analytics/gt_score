"""
Walk-Forward Validation Framework

This module implements proper time-series cross-validation for trading strategies,
preventing data leakage through embargo periods between training and validation sets.

Walk-forward validation splits data into multiple train/validate pairs where
training always precedes validation chronologically.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Tuple, Generator


def generate_walkforward_splits(data, train_years=4, val_years=2, step_years=1, 
                                 embargo_days=30):
    """
    Generate walk-forward validation splits for time-series data.
    
    Creates multiple train/validation splits where:
    1. Training period always precedes validation
    2. An embargo period separates train and validate to prevent leakage
    3. Splits are stepped forward in time
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with a 'Date' column containing the full time series.
    train_years : int, default=4
        Number of years for training window.
    val_years : int, default=2  
        Number of years for validation window.
    step_years : int, default=1
        Number of years to step forward between splits.
    embargo_days : int, default=30
        Gap days between train end and validate start to prevent leakage.
    
    Yields
    ------
    tuple of (pd.DataFrame, pd.DataFrame, dict)
        - train_data: Training data subset
        - val_data: Validation data subset
        - split_info: Dictionary with split metadata
    
    Examples
    --------
    For 2010-2024 data with train_years=4, val_years=2, step_years=1:
    - Split 1: Train 2010-2014, Validate 2014-2016
    - Split 2: Train 2011-2015, Validate 2015-2017
    - Split 3: Train 2012-2016, Validate 2016-2018
    ... etc.
    
    >>> for train, val, info in generate_walkforward_splits(df):
    ...     print(f"Split {info['split_num']}: Train {info['train_start']} to {info['train_end']}")
    ...     print(f"  Validate {info['val_start']} to {info['val_end']}")
    ...     # Run optimization on train, evaluate on val
    
    Notes
    -----
    The embargo period is critical for financial time series to prevent
    information leakage from autocorrelated features.
    """
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    train_days = int(train_years * 365.25)
    val_days = int(val_years * 365.25)
    step_days = int(step_years * 365.25)
    
    split_num = 0
    current_train_start = start_date
    
    while True:
        # Calculate split boundaries
        train_start = current_train_start
        train_end = train_start + timedelta(days=train_days)
        
        val_start = train_end + timedelta(days=embargo_days)
        val_end = val_start + timedelta(days=val_days)
        
        # Check if we have enough data for this split
        if val_end > end_date:
            break
        
        # Extract train and validation data
        train_mask = (df['Date'] >= train_start) & (df['Date'] < train_end)
        val_mask = (df['Date'] >= val_start) & (df['Date'] < val_end)
        
        train_data = df[train_mask].copy().reset_index(drop=True)
        val_data = df[val_mask].copy().reset_index(drop=True)
        
        # Only yield if both splits have data
        if len(train_data) > 0 and len(val_data) > 0:
            split_num += 1
            split_info = {
                'split_num': split_num,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'embargo_days': embargo_days
            }
            yield train_data, val_data, split_info
        
        # Step forward
        current_train_start += timedelta(days=step_days)


def count_walkforward_splits(data, train_years=4, val_years=2, step_years=1, 
                              embargo_days=30):
    """
    Count the number of walk-forward splits for given parameters.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with a 'Date' column.
    train_years, val_years, step_years, embargo_days : 
        Same as generate_walkforward_splits
    
    Returns
    -------
    int
        Number of splits that will be generated.
    """
    count = 0
    for _ in generate_walkforward_splits(data, train_years, val_years, 
                                          step_years, embargo_days):
        count += 1
    return count


class WalkForwardValidator:
    """
    Walk-forward validation manager for trading strategy optimization.
    
    Handles the full walk-forward validation workflow:
    1. Generate train/val splits
    2. Optimize on training data
    3. Evaluate on validation data
    4. Aggregate results across splits
    
    Attributes
    ----------
    train_years : int
        Years of training data per split.
    val_years : int
        Years of validation data per split.
    step_years : int
        Years to step between splits.
    embargo_days : int
        Gap between train and validate.
    
    Examples
    --------
    >>> validator = WalkForwardValidator()
    >>> results = validator.validate(
    ...     data=df,
    ...     optimize_fn=lambda train: optimize(strategies, train, gt_score),
    ...     evaluate_fn=lambda val, params: backtest(val, params)
    ... )
    """
    
    def __init__(self, train_years=4, val_years=2, step_years=1, embargo_days=30):
        self.train_years = train_years
        self.val_years = val_years
        self.step_years = step_years
        self.embargo_days = embargo_days
        self.results = []
    
    def validate(self, data, optimize_fn, evaluate_fn, verbose=True):
        """
        Run full walk-forward validation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full dataset with 'Date' column.
        optimize_fn : callable
            Function that takes training data and returns optimized parameters.
            Signature: optimize_fn(train_data) -> dict with 'best_params', 'best_loss'
        evaluate_fn : callable
            Function that evaluates parameters on validation data.
            Signature: evaluate_fn(val_data, params) -> dict with 'loss', 'metrics'
        verbose : bool
            Whether to print progress.
        
        Returns
        -------
        dict
            Aggregated results with per-split and summary statistics.
        """
        from tqdm import tqdm
        
        splits = list(generate_walkforward_splits(
            data, self.train_years, self.val_years, 
            self.step_years, self.embargo_days
        ))
        
        self.results = []
        
        iterator = tqdm(splits, desc="Walk-Forward") if verbose else splits
        
        for train_data, val_data, split_info in iterator:
            # Optimize on training data
            opt_result = optimize_fn(train_data)
            
            # Evaluate on validation data
            val_result = evaluate_fn(val_data, opt_result['best_params'])
            
            self.results.append({
                'split_info': split_info,
                'train_loss': opt_result['best_loss'],
                'train_params': opt_result['best_params'],
                'val_loss': val_result.get('loss', None),
                'val_metrics': val_result.get('metrics', {}),
            })
        
        return self._aggregate_results()
    
    def _aggregate_results(self):
        """Aggregate results across all splits."""
        if not self.results:
            return {'splits': [], 'summary': {}}
        
        train_losses = [r['train_loss'] for r in self.results if r['train_loss'] is not None]
        val_losses = [r['val_loss'] for r in self.results if r['val_loss'] is not None]
        
        summary = {
            'n_splits': len(self.results),
            'train_loss_mean': np.mean(train_losses) if train_losses else None,
            'train_loss_std': np.std(train_losses) if train_losses else None,
            'val_loss_mean': np.mean(val_losses) if val_losses else None,
            'val_loss_std': np.std(val_losses) if val_losses else None,
        }
        
        # Overfitting ratio (validation / training)
        if train_losses and val_losses and len(train_losses) == len(val_losses):
            ratios = []
            for t, v in zip(train_losses, val_losses):
                if t != 0:
                    ratios.append(v / t)
            if ratios:
                summary['overfitting_ratio_mean'] = np.mean(ratios)
                summary['overfitting_ratio_std'] = np.std(ratios)
        
        return {
            'splits': self.results,
            'summary': summary
        }
