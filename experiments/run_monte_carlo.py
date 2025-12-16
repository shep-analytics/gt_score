"""
Monte Carlo Experiment Runner

Runs multiple optimization trials with different random seeds to assess
the stability and reliability of each loss function.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    optimize, gt_score, run_backtest,
    LOSS_FUNCTIONS, generate_walkforward_splits
)
from strategies import strategies
from data import (
    fetch_all, fetch_test_assets, fetch_multiple,
    SP500_TOP_100, FULL_ASSET_UNIVERSE, TEST_ASSETS
)


# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default seeds (30 for Monte Carlo)
DEFAULT_SEEDS = list(range(42, 72))  # Seeds 42-71 (30 total)

# Loss functions to compare
LOSS_FUNCTION_NAMES = ['gt_score', 'simple', 'sharpe', 'sortino', 'ridge', 'elastic_net']

# Optimizers to test
OPTIMIZERS = ['random', 'hyperopt', 'genetic']


def estimate_time_remaining(completed, total, elapsed_seconds):
    """Estimate time remaining based on progress."""
    if completed == 0:
        return "Calculating..."
    
    rate = elapsed_seconds / completed
    remaining = (total - completed) * rate
    
    if remaining < 60:
        return f"{remaining:.0f}s"
    elif remaining < 3600:
        return f"{remaining/60:.1f}m"
    else:
        hours = remaining / 3600
        if hours > 24:
            return f"{hours/24:.1f}d"
        return f"{hours:.1f}h"


def run_single_monte_carlo(asset_data, strategy, loss_function, optimizer,
                            seed, train_pct=0.7, max_evals=50):
    """
    Run a single Monte Carlo trial.
    
    Parameters
    ----------
    asset_data : pd.DataFrame
        Asset price data with Date, OHLC columns.
    strategy : dict
        Strategy configuration.
    loss_function : callable
        Loss function to optimize.
    optimizer : str
        Optimizer method.
    seed : int
        Random seed for reproducibility.
    train_pct : float
        Fraction of data for training.
    max_evals : int
        Optimizer iterations.
    
    Returns
    -------
    dict
        Trial results including train/test performance.
    """
    # Split data
    n = len(asset_data)
    train_end = int(n * train_pct)
    
    train_data = asset_data.iloc[:train_end].copy().reset_index(drop=True)
    test_data = asset_data.iloc[train_end:].copy().reset_index(drop=True)
    
    if len(train_data) < 200 or len(test_data) < 50:
        return {'error': 'Insufficient data for train/test split'}
    
    train_df = {'ohlc': train_data}
    test_df = {'ohlc': test_data}
    
    try:
        # Optimize on training data
        start_time = time.time()
        opt_result = optimize(
            strategies=[strategy],
            data_frames=[train_df],
            loss_function=loss_function,
            optimization_method=optimizer,
            max_evals=max_evals,
            random_seed=seed,
            verbose=False
        )
        opt_time = time.time() - start_time
        
        # Evaluate on training data
        train_signals = strategy['strategy'](train_data, opt_result['best_params'])
        train_backtest, _ = run_backtest(train_signals)
        train_loss = loss_function(train_backtest)
        
        # Evaluate on test data
        test_signals = strategy['strategy'](test_data, opt_result['best_params'])
        test_backtest, _ = run_backtest(test_signals)
        test_loss = loss_function(test_backtest)
        
        return {
            'seed': seed,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_return': train_backtest.get('total_percentage_gain', 0),
            'test_return': test_backtest.get('total_percentage_gain', 0),
            'train_trades': train_backtest.get('total_trades', 0),
            'test_trades': test_backtest.get('total_trades', 0),
            'best_params': opt_result['best_params'],
            'optimization_time': opt_time,
            'overfitting_ratio': (
                test_backtest.get('total_percentage_gain', 0) / 
                train_backtest.get('total_percentage_gain', 1)
            ) if train_backtest.get('total_percentage_gain', 0) != 0 else 0
        }
    
    except Exception as e:
        return {'seed': seed, 'error': str(e)}


def load_checkpoint(checkpoint_file):
    """Load checkpoint if exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_file, results, progress):
    """Save checkpoint for resuming."""
    checkpoint = {
        'results': results,
        'progress': progress,
        'timestamp': str(datetime.now())
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)


def run_monte_carlo_experiment(
    assets_to_run=None,
    loss_functions_to_run=None,
    optimizers_to_run=None,
    strategies_to_run=None,
    seeds=None,
    max_evals=50,
    output_file="monte_carlo_results.json",
    checkpoint_file="monte_carlo_checkpoint.json",
    resume=True
):
    """
    Run full Monte Carlo experiment across assets, loss functions, and seeds.
    
    Parameters
    ----------
    assets_to_run : list, optional
        Tickers to test. Default: top 100 S&P 500.
    loss_functions_to_run : list, optional
        Loss function names. Default: all 6.
    optimizers_to_run : list, optional
        Optimizer methods. Default: ['random'].
    strategies_to_run : list, optional
        Strategy indices. Default: all.
    seeds : list, optional
        Random seeds. Default: 42-71 (30 seeds).
    max_evals : int
        Optimizer iterations.
    output_file : str
        Output JSON filename.
    checkpoint_file : str
        Checkpoint filename for resuming.
    resume : bool
        Whether to resume from checkpoint.
    """
    # Defaults
    if assets_to_run is None:
        assets_to_run = SP500_TOP_100[:100]
    if loss_functions_to_run is None:
        loss_functions_to_run = ['gt_score', 'simple', 'sharpe', 'sortino']
    if optimizers_to_run is None:
        optimizers_to_run = ['random']
    if strategies_to_run is None:
        strategies_to_run = [0]  # Just first strategy by default
    if seeds is None:
        seeds = DEFAULT_SEEDS
    
    checkpoint_path = OUTPUT_DIR / checkpoint_file
    
    # Try to load checkpoint
    all_results = []
    completed_tasks = set()
    
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            all_results = checkpoint['results']
            completed_tasks = set(tuple(p) for p in checkpoint['progress'])
            print(f"Resuming from checkpoint: {len(completed_tasks)} tasks completed")
    
    # Fetch data
    print("Fetching asset data...")
    data = fetch_multiple(assets_to_run)
    assets_to_run = [a for a in assets_to_run if a in data]
    
    # Calculate total iterations
    total_tasks = (
        len(assets_to_run) * 
        len(loss_functions_to_run) * 
        len(optimizers_to_run) * 
        len(strategies_to_run) * 
        len(seeds)
    )
    
    remaining_tasks = total_tasks - len(completed_tasks)
    
    print(f"\n{'='*70}")
    print("MONTE CARLO EXPERIMENT")
    print(f"{'='*70}")
    print(f"Assets:           {len(assets_to_run)}")
    print(f"Loss functions:   {loss_functions_to_run}")
    print(f"Optimizers:       {optimizers_to_run}")
    print(f"Strategies:       {len(strategies_to_run)}")
    print(f"Seeds per config: {len(seeds)}")
    print(f"Max evaluations:  {max_evals}")
    print(f"{'='*70}")
    print(f"Total tasks:      {total_tasks:,}")
    print(f"Already done:     {len(completed_tasks):,}")
    print(f"Remaining:        {remaining_tasks:,}")
    
    # Estimate time
    est_time_per_task = 10  # seconds (rough estimate)
    est_total_seconds = remaining_tasks * est_time_per_task
    if est_total_seconds < 3600:
        print(f"Estimated time:   {est_total_seconds/60:.1f} minutes")
    else:
        print(f"Estimated time:   {est_total_seconds/3600:.1f} hours")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    newly_completed = 0
    
    # Progress bar
    pbar = tqdm(total=remaining_tasks, desc="Monte Carlo")
    
    for asset in assets_to_run:
        asset_data = data.get(asset)
        if asset_data is None:
            continue
        
        for loss_name in loss_functions_to_run:
            loss_fn = LOSS_FUNCTIONS.get(loss_name)
            if loss_fn is None:
                continue
            
            for optimizer in optimizers_to_run:
                for strat_idx in strategies_to_run:
                    strat = strategies[strat_idx]
                    strat_name = strat.get('name', strat['strategy'].__name__)
                    
                    for seed in seeds:
                        task_id = (asset, loss_name, optimizer, strat_name, seed)
                        
                        if task_id in completed_tasks:
                            continue
                        
                        # Run trial
                        try:
                            result = run_single_monte_carlo(
                                asset_data, strat, loss_fn, optimizer,
                                seed, max_evals=max_evals
                            )
                            
                            result.update({
                                'asset': asset,
                                'strategy': strat_name,
                                'loss_function': loss_name,
                                'optimizer': optimizer
                            })
                            
                            all_results.append(result)
                        
                        except Exception as e:
                            all_results.append({
                                'asset': asset,
                                'strategy': strat_name,
                                'loss_function': loss_name,
                                'optimizer': optimizer,
                                'seed': seed,
                                'error': str(e)
                            })
                        
                        completed_tasks.add(task_id)
                        newly_completed += 1
                        
                        elapsed = time.time() - start_time
                        eta = estimate_time_remaining(newly_completed, remaining_tasks, elapsed)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Asset': asset[:6],
                            'Loss': loss_name[:5],
                            'Seed': seed,
                            'ETA': eta
                        })
                        
                        # Checkpoint every 50 completions
                        if newly_completed % 50 == 0:
                            save_checkpoint(
                                checkpoint_path, 
                                all_results,
                                [list(t) for t in completed_tasks]
                            )
    
    pbar.close()
    
    # Save final results
    output_path = OUTPUT_DIR / output_file
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Remove checkpoint (completed successfully)
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total time:    {total_time/3600:.2f} hours")
    print(f"Tasks run:     {newly_completed:,}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}\n")
    
    return all_results


def main():
    """Main entry point with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Monte Carlo experiment")
    parser.add_argument('--assets', type=int, default=None,
                       help='Number of assets (default: 100)')
    parser.add_argument('--seeds', type=int, default=None,
                       help='Number of seeds (default: 30)')
    parser.add_argument('--strategies', type=int, default=None,
                       help='Number of strategies (default: 1)')
    parser.add_argument('--max-evals', type=int, default=50,
                       help='Optimizer evaluations')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, ignore checkpoint')
    parser.add_argument('--test', action='store_true',
                       help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.test:
        # Quick test mode
        run_monte_carlo_experiment(
            assets_to_run=['AAPL', 'MSFT'],
            loss_functions_to_run=['gt_score', 'sharpe'],
            optimizers_to_run=['random'],
            strategies_to_run=[0],
            seeds=[42, 43, 44, 45, 46],
            max_evals=10,
            output_file="monte_carlo_test.json",
            checkpoint_file="monte_carlo_test_checkpoint.json"
        )
    else:
        assets = SP500_TOP_100[:args.assets] if args.assets else None
        seeds = DEFAULT_SEEDS[:args.seeds] if args.seeds else None
        strats = list(range(args.strategies)) if args.strategies else None
        
        run_monte_carlo_experiment(
            assets_to_run=assets,
            seeds=seeds,
            strategies_to_run=strats,
            max_evals=args.max_evals,
            resume=not args.no_resume
        )


if __name__ == "__main__":
    main()
