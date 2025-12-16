"""
Parallel Walk-Forward Validation Runner

Runs walk-forward validation with multiple time splits to test
out-of-sample performance across different market regimes.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import optimize, run_backtest, LOSS_FUNCTIONS
from src.walkforward import generate_walkforward_splits
from strategies import strategies
from data import fetch_multiple, SP500_TOP_100

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_wf_split(args):
    """
    Run a single walk-forward split. Designed for parallel execution.
    """
    (asset, train_data_dict, val_data_dict, split_info, 
     strategy_config, loss_name, max_evals) = args
    
    try:
        # Reconstruct DataFrames
        train_data = pd.DataFrame(train_data_dict)
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        val_data = pd.DataFrame(val_data_dict)
        val_data['Date'] = pd.to_datetime(val_data['Date'])
        
        loss_fn = LOSS_FUNCTIONS.get(loss_name)
        if loss_fn is None:
            return {'error': f'Unknown loss function: {loss_name}'}
        
        train_df = {'ohlc': train_data}
        
        # Optimize on training data
        opt_result = optimize(
            strategies=[strategy_config],
            data_frames=[train_df],
            loss_function=loss_fn,
            optimization_method='random',
            max_evals=max_evals,
            random_seed=42,  # Fixed seed for walk-forward
            verbose=False
        )
        
        # Evaluate on training data
        train_signals = strategy_config['strategy'](train_data, opt_result['best_params'])
        train_backtest, _ = run_backtest(train_signals)
        train_loss = loss_fn(train_backtest)
        
        # Evaluate on validation data
        val_signals = strategy_config['strategy'](val_data, opt_result['best_params'])
        val_backtest, _ = run_backtest(val_signals)
        val_loss = loss_fn(val_backtest)
        
        return {
            'asset': asset,
            'strategy': strategy_config.get('name', 'Unknown'),
            'loss_function': loss_name,
            'split_num': split_info['split_num'],
            'train_start': str(split_info['train_start']),
            'train_end': str(split_info['train_end']),
            'val_start': str(split_info['val_start']),
            'val_end': str(split_info['val_end']),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_return': train_backtest.get('total_percentage_gain', 0),
            'val_return': val_backtest.get('total_percentage_gain', 0),
            'train_trades': train_backtest.get('total_trades', 0),
            'val_trades': val_backtest.get('total_trades', 0),
            'best_params': opt_result['best_params'],
        }
    
    except Exception as e:
        return {
            'asset': asset,
            'strategy': strategy_config.get('name', 'Unknown') if strategy_config else 'Unknown',
            'loss_function': loss_name,
            'split_num': split_info.get('split_num', 0) if split_info else 0,
            'error': str(e)
        }


def run_parallel_walkforward(
    n_assets=50,
    loss_functions=None,
    strategy_indices=None,
    train_years=4,
    val_years=2,
    step_years=1,
    max_evals=25,
    n_workers=None,
    output_file="walkforward_balanced.json"
):
    """
    Run parallelized walk-forward validation.
    """
    if loss_functions is None:
        loss_functions = ['gt_score', 'sharpe', 'sortino', 'simple']
    
    if strategy_indices is None:
        strategy_indices = [0, 1, 2]  # RSI, MACD, Bollinger
    
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    
    assets_to_use = SP500_TOP_100[:n_assets]
    strategies_to_use = [strategies[i] for i in strategy_indices]
    
    print("="*70)
    print("PARALLEL WALK-FORWARD VALIDATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {n_workers}")
    print(f"Assets: {n_assets}")
    print(f"Loss functions: {loss_functions}")
    print(f"Strategies: {[s.get('name', s['strategy'].__name__) for s in strategies_to_use]}")
    print(f"Train/Val/Step: {train_years}/{val_years}/{step_years} years")
    print(f"Max evals: {max_evals}")
    print("="*70)
    
    # Fetch data
    print("\nFetching data...")
    data = fetch_multiple(assets_to_use, use_cache=True, verbose=True)
    assets_with_data = [a for a in assets_to_use if a in data]
    
    # Prepare tasks
    print("\nPreparing walk-forward splits...")
    tasks = []
    
    for asset in assets_with_data:
        asset_data = data[asset]
        
        # Generate walk-forward splits
        splits = list(generate_walkforward_splits(
            asset_data,
            train_years=train_years,
            val_years=val_years,
            step_years=step_years
        ))
        
        for train_data, val_data, split_info in splits:
            train_dict = train_data.to_dict('list')
            val_dict = val_data.to_dict('list')
            
            for strat in strategies_to_use:
                for loss_name in loss_functions:
                    tasks.append((
                        asset,
                        train_dict,
                        val_dict,
                        split_info,
                        strat,
                        loss_name,
                        max_evals
                    ))
    
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks:,}")
    
    # Estimate time
    time_per_task = 7.5  # seconds
    parallel_time = (total_tasks * time_per_task) / n_workers
    
    print(f"Estimated time: {parallel_time/60:.1f} minutes with {n_workers} workers")
    print("="*70)
    print()
    
    # Run in parallel
    results = []
    start_time = time.time()
    completed = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_wf_split, task): task for task in tasks}
        
        pbar = tqdm(total=total_tasks, desc="Walk-Forward")
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            completed += 1
            if 'error' in result:
                errors += 1
            
            elapsed = time.time() - start_time
            rate = completed / elapsed
            remaining = (total_tasks - completed) / rate if rate > 0 else 0
            
            pbar.update(1)
            pbar.set_postfix({
                'Done': f'{completed}/{total_tasks}',
                'Errors': errors,
                'ETA': f'{remaining/60:.0f}m'
            })
            
            # Checkpoint every 200 completions
            if completed % 200 == 0:
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
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Tasks completed: {completed:,}")
    print(f"Errors: {errors}")
    print(f"Results saved: {OUTPUT_DIR / output_file}")
    print("="*70)
    
    # Quick summary
    df = pd.DataFrame([r for r in results if 'error' not in r])
    if len(df) > 0:
        print("\nResults Summary:")
        summary = df.groupby('loss_function')['val_return'].agg(['mean', 'std', 'count'])
        print(summary.round(4))
        
        # Overfitting analysis
        print("\nOverfitting Analysis (Val/Train ratio):")
        for loss in df['loss_function'].unique():
            subset = df[df['loss_function'] == loss]
            train_mean = subset['train_return'].mean()
            val_mean = subset['val_return'].mean()
            ratio = val_mean / train_mean if train_mean > 0 else 0
            print(f"  {loss}: {ratio:.3f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parallel walk-forward validation")
    parser.add_argument('--assets', type=int, default=50)
    parser.add_argument('--strategies', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--max-evals', type=int, default=25)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--output', type=str, default='walkforward_balanced.json')
    
    args = parser.parse_args()
    
    run_parallel_walkforward(
        n_assets=args.assets,
        strategy_indices=args.strategies,
        max_evals=args.max_evals,
        n_workers=args.workers,
        output_file=args.output
    )
