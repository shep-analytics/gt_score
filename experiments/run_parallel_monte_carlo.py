"""
Parallel Monte Carlo Experiment Runner

Optimized for multi-core execution with configurable parameters.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import optimize, run_backtest, LOSS_FUNCTIONS
from strategies import strategies
from data import fetch_multiple, SP500_TOP_100

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_trial(args):
    """
    Run a single Monte Carlo trial. Designed for parallel execution.
    
    Returns dict with results or error.
    """
    asset, asset_data_dict, strategy_config, loss_name, seed, max_evals, train_pct = args
    
    try:
        # Reconstruct DataFrame from dict (for pickling)
        asset_data = pd.DataFrame(asset_data_dict)
        asset_data['Date'] = pd.to_datetime(asset_data['Date'])
        
        # Get loss function
        loss_fn = LOSS_FUNCTIONS.get(loss_name)
        if loss_fn is None:
            return {'asset': asset, 'error': f'Unknown loss function: {loss_name}'}
        
        # Split data
        n = len(asset_data)
        train_end = int(n * train_pct)
        
        train_data = asset_data.iloc[:train_end].copy().reset_index(drop=True)
        test_data = asset_data.iloc[train_end:].copy().reset_index(drop=True)
        
        if len(train_data) < 200 or len(test_data) < 50:
            return {'asset': asset, 'error': 'Insufficient data'}
        
        train_df = {'ohlc': train_data}
        
        # Optimize on training data
        opt_result = optimize(
            strategies=[strategy_config],
            data_frames=[train_df],
            loss_function=loss_fn,
            optimization_method='random',
            max_evals=max_evals,
            random_seed=seed,
            verbose=False
        )
        
        # Evaluate on training data
        train_signals = strategy_config['strategy'](train_data, opt_result['best_params'])
        train_backtest, _ = run_backtest(train_signals)
        train_loss = loss_fn(train_backtest)
        
        # Evaluate on test data
        test_signals = strategy_config['strategy'](test_data, opt_result['best_params'])
        test_backtest, _ = run_backtest(test_signals)
        test_loss = loss_fn(test_backtest)
        
        return {
            'asset': asset,
            'strategy': strategy_config.get('name', 'Unknown'),
            'loss_function': loss_name,
            'seed': seed,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_return': train_backtest.get('total_percentage_gain', 0),
            'test_return': test_backtest.get('total_percentage_gain', 0),
            'train_trades': train_backtest.get('total_trades', 0),
            'test_trades': test_backtest.get('total_trades', 0),
            'best_params': opt_result['best_params'],
        }
    
    except Exception as e:
        return {
            'asset': asset,
            'strategy': strategy_config.get('name', 'Unknown') if strategy_config else 'Unknown',
            'loss_function': loss_name,
            'seed': seed,
            'error': str(e)
        }


def run_parallel_monte_carlo(
    n_assets=50,
    loss_functions=None,
    strategy_indices=None,
    n_seeds=15,
    max_evals=25,
    n_workers=None,
    output_file="monte_carlo_balanced.json"
):
    """
    Run parallelized Monte Carlo experiment.
    
    Parameters
    ----------
    n_assets : int
        Number of top S&P 500 assets to use.
    loss_functions : list
        Loss function names. Default: ['gt_score', 'sharpe', 'sortino', 'simple']
    strategy_indices : list
        Strategy indices. Default: [0, 1, 2] (RSI, MACD, Bollinger)
    n_seeds : int
        Number of random seeds.
    max_evals : int
        Optimizer evaluations.
    n_workers : int
        Number of parallel workers. Default: CPU count - 2
    output_file : str
        Output filename.
    """
    if loss_functions is None:
        loss_functions = ['gt_score', 'sharpe', 'sortino', 'simple']
    
    if strategy_indices is None:
        strategy_indices = [0, 1, 2]  # RSI, MACD, Bollinger
    
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    
    seeds = list(range(42, 42 + n_seeds))
    assets_to_use = SP500_TOP_100[:n_assets]
    strategies_to_use = [strategies[i] for i in strategy_indices]
    
    print("="*70)
    print("PARALLEL MONTE CARLO EXPERIMENT")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {n_workers}")
    print(f"Assets: {n_assets}")
    print(f"Loss functions: {loss_functions}")
    print(f"Strategies: {[s.get('name', s['strategy'].__name__) for s in strategies_to_use]}")
    print(f"Seeds: {n_seeds}")
    print(f"Max evals: {max_evals}")
    print("="*70)
    
    # Fetch data
    print("\nFetching data...")
    data = fetch_multiple(assets_to_use, use_cache=True, verbose=True)
    assets_with_data = [a for a in assets_to_use if a in data]
    
    # Prepare tasks
    print("\nPreparing tasks...")
    tasks = []
    
    for asset in assets_with_data:
        # Convert DataFrame to dict for pickling
        asset_data_dict = data[asset].to_dict('list')
        
        for strat in strategies_to_use:
            for loss_name in loss_functions:
                for seed in seeds:
                    tasks.append((
                        asset,
                        asset_data_dict,
                        strat,
                        loss_name,
                        seed,
                        max_evals,
                        0.7  # train_pct
                    ))
    
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks:,}")
    
    # Estimate time
    time_per_task = 7.5  # seconds (with 25 evals)
    serial_time = total_tasks * time_per_task
    parallel_time = serial_time / n_workers
    
    print(f"Estimated time: {parallel_time/3600:.1f} hours with {n_workers} workers")
    print("="*70)
    print()
    
    # Run in parallel
    results = []
    start_time = time.time()
    completed = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_trial, task): task for task in tasks}
        
        pbar = tqdm(total=total_tasks, desc="Monte Carlo")
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            completed += 1
            if 'error' in result:
                errors += 1
            
            # Update progress
            elapsed = time.time() - start_time
            rate = completed / elapsed
            remaining = (total_tasks - completed) / rate if rate > 0 else 0
            
            pbar.update(1)
            pbar.set_postfix({
                'Done': f'{completed}/{total_tasks}',
                'Errors': errors,
                'ETA': f'{remaining/60:.0f}m'
            })
            
            # Checkpoint every 500 completions
            if completed % 500 == 0:
                with open(OUTPUT_DIR / output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
        
        pbar.close()
    
    # Save final results
    with open(OUTPUT_DIR / output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    total_time = time.time() - start_time
    
    print()
    print("="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.0f} minutes)")
    print(f"Tasks completed: {completed:,}")
    print(f"Errors: {errors}")
    print(f"Results saved: {OUTPUT_DIR / output_file}")
    print("="*70)
    
    # Quick summary
    df = pd.DataFrame([r for r in results if 'error' not in r])
    if len(df) > 0:
        print("\nResults Summary:")
        summary = df.groupby('loss_function')['test_return'].agg(['mean', 'std', 'count'])
        print(summary.round(4))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parallel Monte Carlo")
    parser.add_argument('--assets', type=int, default=50)
    parser.add_argument('--seeds', type=int, default=15)
    parser.add_argument('--strategies', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--max-evals', type=int, default=25)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--output', type=str, default='monte_carlo_balanced.json')
    
    args = parser.parse_args()
    
    run_parallel_monte_carlo(
        n_assets=args.assets,
        strategy_indices=args.strategies,
        n_seeds=args.seeds,
        max_evals=args.max_evals,
        n_workers=args.workers,
        output_file=args.output
    )
