"""
Walk-Forward Validation Experiment Runner

Runs walk-forward validation across all assets and loss functions
to evaluate generalization performance and detect overfitting.
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
    simple_loss_function, sharpe_ratio_loss_function,
    sortino_ratio_loss_function, ridge_regression_loss_function,
    elastic_net_loss_function, LOSS_FUNCTIONS,
    generate_walkforward_splits, count_walkforward_splits
)
from strategies import strategies
from data import fetch_all, fetch_test_assets, SP500_TOP_100, FULL_ASSET_UNIVERSE


# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_time_remaining(completed, total, elapsed_seconds):
    """Estimate time remaining based on progress."""
    if completed == 0:
        return "Calculating..."
    
    rate = elapsed_seconds / completed
    remaining = (total - completed) * rate
    
    if remaining < 60:
        return f"{remaining:.0f} seconds"
    elif remaining < 3600:
        return f"{remaining/60:.1f} minutes"
    else:
        return f"{remaining/3600:.1f} hours"


def run_single_walkforward(asset_data, strategy, loss_function, loss_name,
                            train_years=4, val_years=2, step_years=1,
                            max_evals=50, seed=42):
    """
    Run walk-forward validation for a single asset/strategy/loss combo.
    
    Returns
    -------
    dict
        Results including train/val performance for each split.
    """
    results = []
    
    splits = list(generate_walkforward_splits(
        asset_data, 
        train_years=train_years, 
        val_years=val_years, 
        step_years=step_years
    ))
    
    for train_data, val_data, split_info in splits:
        try:
            # Prepare data for optimizer
            train_df = {'ohlc': train_data}
            val_df = {'ohlc': val_data}
            
            # Optimize on training data
            opt_result = optimize(
                strategies=[strategy],
                data_frames=[train_df],
                loss_function=loss_function,
                optimization_method='random',
                max_evals=max_evals,
                random_seed=seed,
                verbose=False
            )
            
            # Evaluate on validation data
            trading_signals = strategy['strategy'](val_data, opt_result['best_params'])
            val_backtest, _ = run_backtest(trading_signals)
            val_loss = loss_function(val_backtest)
            
            results.append({
                'split_num': split_info['split_num'],
                'train_start': str(split_info['train_start']),
                'train_end': str(split_info['train_end']),
                'val_start': str(split_info['val_start']),
                'val_end': str(split_info['val_end']),
                'train_loss': opt_result['best_loss'],
                'val_loss': val_loss,
                'best_params': opt_result['best_params'],
                'val_total_return': val_backtest.get('total_percentage_gain', 0),
                'val_trades': val_backtest.get('total_trades', 0),
            })
        
        except Exception as e:
            results.append({
                'split_num': split_info['split_num'],
                'error': str(e)
            })
    
    return results


def run_walkforward_experiment(assets_to_run=None, loss_functions_to_run=None,
                                strategies_to_run=None, max_evals=50,
                                output_file="walkforward_results.json"):
    """
    Run full walk-forward validation experiment.
    
    Parameters
    ----------
    assets_to_run : list, optional
        List of tickers to run. Default: all assets.
    loss_functions_to_run : list, optional
        List of loss function names. Default: all.
    strategies_to_run : list, optional
        List of strategy indices. Default: all.
    max_evals : int
        Optimizer evaluations per split.
    output_file : str
        Output JSON filename.
    """
    # Defaults
    if loss_functions_to_run is None:
        loss_functions_to_run = ['gt_score', 'sharpe', 'sortino', 'simple']
    
    if strategies_to_run is None:
        strategies_to_run = list(range(len(strategies)))
    
    # Fetch data
    print("Fetching asset data...")
    if assets_to_run is None:
        data = fetch_all()
        assets_to_run = list(data.keys())
    else:
        from data import fetch_multiple
        data = fetch_multiple(assets_to_run)
    
    # Calculate total iterations
    total_tasks = (
        len(assets_to_run) * 
        len(loss_functions_to_run) * 
        len(strategies_to_run)
    )
    
    print(f"\n{'='*60}")
    print("WALK-FORWARD VALIDATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Assets: {len(assets_to_run)}")
    print(f"Loss functions: {loss_functions_to_run}")
    print(f"Strategies: {len(strategies_to_run)}")
    print(f"Total task combinations: {total_tasks}")
    print(f"{'='*60}\n")
    
    all_results = []
    start_time = time.time()
    completed = 0
    
    # Progress tracking
    pbar = tqdm(total=total_tasks, desc="Overall progress")
    
    for asset_ticker in assets_to_run:
        asset_data = data.get(asset_ticker)
        if asset_data is None:
            continue
        
        for loss_name in loss_functions_to_run:
            loss_fn = LOSS_FUNCTIONS.get(loss_name)
            if loss_fn is None:
                continue
            
            for strat_idx in strategies_to_run:
                strat = strategies[strat_idx]
                
                # Run walk-forward
                try:
                    wf_results = run_single_walkforward(
                        asset_data, strat, loss_fn, loss_name,
                        max_evals=max_evals
                    )
                    
                    all_results.append({
                        'asset': asset_ticker,
                        'strategy': strat.get('name', strat['strategy'].__name__),
                        'loss_function': loss_name,
                        'splits': wf_results
                    })
                
                except Exception as e:
                    all_results.append({
                        'asset': asset_ticker,
                        'strategy': strat.get('name', strat['strategy'].__name__),
                        'loss_function': loss_name,
                        'error': str(e)
                    })
                
                completed += 1
                elapsed = time.time() - start_time
                eta = estimate_time_remaining(completed, total_tasks, elapsed)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Asset': asset_ticker[:6],
                    'ETA': eta
                })
                
                # Checkpoint every 10 completions
                if completed % 10 == 0:
                    with open(OUTPUT_DIR / output_file, 'w') as f:
                        json.dump(all_results, f, indent=2, default=str)
    
    pbar.close()
    
    # Save final results
    with open(OUTPUT_DIR / output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {OUTPUT_DIR / output_file}")
    print(f"{'='*60}\n")
    
    return all_results


def main():
    """Main entry point with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument('--assets', type=int, default=None,
                       help='Number of assets to run (default: all)')
    parser.add_argument('--strategies', type=int, default=None,
                       help='Number of strategies (default: all)')
    parser.add_argument('--max-evals', type=int, default=50,
                       help='Optimizer evaluations per split')
    parser.add_argument('--test', action='store_true',
                       help='Quick test with minimal settings')
    
    args = parser.parse_args()
    
    if args.test:
        # Quick test mode
        run_walkforward_experiment(
            assets_to_run=['AAPL', 'MSFT'],
            loss_functions_to_run=['gt_score', 'sharpe'],
            strategies_to_run=[0],
            max_evals=10,
            output_file="walkforward_test.json"
        )
    else:
        assets = SP500_TOP_100[:args.assets] if args.assets else None
        strats = list(range(args.strategies)) if args.strategies else None
        
        run_walkforward_experiment(
            assets_to_run=assets,
            strategies_to_run=strats,
            max_evals=args.max_evals
        )


if __name__ == "__main__":
    main()
