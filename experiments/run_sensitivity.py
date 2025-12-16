"""
Sensitivity Analysis Experiment

Tests how GT-Score performance varies with:
- N_periods parameter: [20, 30, 50, 75, 100]
- Train/validation ratios: [60/40, 70/30, 80/20]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gt_score import gt_score, find_stabilized_variance, get_period_returns
from src import optimize, run_backtest
from strategies import strategies
from data import fetch_multiple, TEST_ASSETS

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_gt_score_with_n_periods(n_periods):
    """Create a GT-Score variant with fixed N_periods."""
    def custom_gt_score(backtest_results, stabilize=False, mode="trades"):
        # Force the specific n_periods
        import math
        from scipy.stats import linregress
        
        num_periods = n_periods  # Use the fixed value
        num_trades = len(backtest_results["trades_history"])
        
        if num_trades <= num_periods:
            interval = (999 - 100) / num_periods
            return 999 - (num_trades * interval)
        
        percentage_returns_by_trade = [
            trade["profit_loss_percent"] 
            for trade in backtest_results["trades_history"]
        ]
        
        if mode == "portfolio_value":
            period_percentage_returns, period_percentage_returns_market = get_period_returns(
                backtest_results['portfolio_values_over_time'], num_periods
            )
        else:
            period_percentage_returns = percentage_returns_by_trade
            starting_market_value = backtest_results["portfolio_values_over_time"][0]["stock_value"]
            ending_market_value = backtest_results["portfolio_values_over_time"][-1]["stock_value"]
            mean_market_return = (ending_market_value / starting_market_value) ** (1 / num_trades) - 1
            period_percentage_returns_market = [mean_market_return] * num_trades
        
        mu = np.mean(period_percentage_returns)
        mum = np.mean(period_percentage_returns_market)
        r2 = linregress(range(len(percentage_returns_by_trade)), percentage_returns_by_trade).rvalue ** 2
        
        negative_returns = [r for r in period_percentage_returns if r < 0]
        sigma_d = np.std(negative_returns) if negative_returns else 1e-6
        sigma = np.std(period_percentage_returns)
        
        z = (mu - mum) / (sigma / np.sqrt(num_trades))
        
        if z <= 0:
            return 100 + (100 * (1 - math.exp(-abs(z - 1))))
        elif z <= 1:
            return 100 * (1 - math.exp(-abs(z - 1)))
        
        ln_z = math.log(z)
        gt = (mu * ln_z * r2) / sigma_d
        return -gt
    
    return custom_gt_score


def run_sensitivity_analysis(assets=None, strategies_to_run=None, seeds=None, max_evals=50):
    """
    Run sensitivity analysis on N_periods and train/val splits.
    """
    if assets is None:
        assets = TEST_ASSETS[:10]
    if strategies_to_run is None:
        strategies_to_run = [0]
    if seeds is None:
        seeds = list(range(42, 52))
    
    # Parameters to test
    n_periods_values = [20, 30, 50, 75, 100]
    train_ratios = [0.60, 0.70, 0.80]
    
    print("Fetching data...")
    data = fetch_multiple(assets)
    
    results = []
    
    # ================================
    # Part 1: N_periods sensitivity
    # ================================
    print("\n=== N_periods Sensitivity ===")
    
    total_1 = len(assets) * len(n_periods_values) * len(strategies_to_run) * len(seeds)
    pbar = tqdm(total=total_1, desc="N_periods")
    
    for asset in assets:
        asset_data = data.get(asset)
        if asset_data is None:
            continue
        
        n = len(asset_data)
        train_end = int(n * 0.7)
        train_data = asset_data.iloc[:train_end].copy().reset_index(drop=True)
        test_data = asset_data.iloc[train_end:].copy().reset_index(drop=True)
        train_df = {'ohlc': train_data}
        
        for n_periods in n_periods_values:
            loss_fn = create_gt_score_with_n_periods(n_periods)
            
            for strat_idx in strategies_to_run:
                strat = strategies[strat_idx]
                strat_name = strat.get('name', strat['strategy'].__name__)
                
                for seed in seeds:
                    try:
                        opt_result = optimize(
                            strategies=[strat],
                            data_frames=[train_df],
                            loss_function=loss_fn,
                            optimization_method='random',
                            max_evals=max_evals,
                            random_seed=seed,
                            verbose=False
                        )
                        
                        test_signals = strat['strategy'](test_data, opt_result['best_params'])
                        test_backtest, _ = run_backtest(test_signals)
                        
                        results.append({
                            'analysis': 'n_periods',
                            'n_periods': n_periods,
                            'train_ratio': 0.7,
                            'asset': asset,
                            'strategy': strat_name,
                            'seed': seed,
                            'train_loss': opt_result['best_loss'],
                            'test_return': test_backtest.get('total_percentage_gain', 0),
                            'test_trades': test_backtest.get('total_trades', 0),
                        })
                    
                    except Exception as e:
                        results.append({
                            'analysis': 'n_periods',
                            'n_periods': n_periods,
                            'asset': asset,
                            'strategy': strat_name,
                            'seed': seed,
                            'error': str(e)
                        })
                    
                    pbar.update(1)
    
    pbar.close()
    
    # ================================
    # Part 2: Train/Val ratio sensitivity
    # ================================
    print("\n=== Train/Val Ratio Sensitivity ===")
    
    total_2 = len(assets) * len(train_ratios) * len(strategies_to_run) * len(seeds)
    pbar = tqdm(total=total_2, desc="Train Ratio")
    
    for asset in assets:
        asset_data = data.get(asset)
        if asset_data is None:
            continue
        
        for train_ratio in train_ratios:
            n = len(asset_data)
            train_end = int(n * train_ratio)
            train_data = asset_data.iloc[:train_end].copy().reset_index(drop=True)
            test_data = asset_data.iloc[train_end:].copy().reset_index(drop=True)
            train_df = {'ohlc': train_data}
            
            for strat_idx in strategies_to_run:
                strat = strategies[strat_idx]
                strat_name = strat.get('name', strat['strategy'].__name__)
                
                for seed in seeds:
                    try:
                        opt_result = optimize(
                            strategies=[strat],
                            data_frames=[train_df],
                            loss_function=gt_score,
                            optimization_method='random',
                            max_evals=max_evals,
                            random_seed=seed,
                            verbose=False
                        )
                        
                        test_signals = strat['strategy'](test_data, opt_result['best_params'])
                        test_backtest, _ = run_backtest(test_signals)
                        
                        results.append({
                            'analysis': 'train_ratio',
                            'n_periods': 50,
                            'train_ratio': train_ratio,
                            'asset': asset,
                            'strategy': strat_name,
                            'seed': seed,
                            'train_loss': opt_result['best_loss'],
                            'test_return': test_backtest.get('total_percentage_gain', 0),
                            'test_trades': test_backtest.get('total_trades', 0),
                        })
                    
                    except Exception as e:
                        results.append({
                            'analysis': 'train_ratio',
                            'train_ratio': train_ratio,
                            'asset': asset,
                            'strategy': strat_name,
                            'seed': seed,
                            'error': str(e)
                        })
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Save results
    output_path = OUTPUT_DIR / "sensitivity_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    df = pd.DataFrame([r for r in results if 'error' not in r])
    
    if len(df) > 0:
        print("\n=== N_periods Sensitivity Summary ===")
        n_periods_df = df[df['analysis'] == 'n_periods']
        if len(n_periods_df) > 0:
            summary = n_periods_df.groupby('n_periods')['test_return'].agg(['mean', 'std', 'count'])
            print(summary)
        
        print("\n=== Train Ratio Sensitivity Summary ===")
        ratio_df = df[df['analysis'] == 'train_ratio']
        if len(ratio_df) > 0:
            summary = ratio_df.groupby('train_ratio')['test_return'].agg(['mean', 'std', 'count'])
            print(summary)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sensitivity analysis")
    parser.add_argument('--assets', type=int, default=10)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--max-evals', type=int, default=50)
    
    args = parser.parse_args()
    
    run_sensitivity_analysis(
        assets=TEST_ASSETS[:args.assets],
        seeds=list(range(42, 42 + args.seeds)),
        max_evals=args.max_evals
    )
