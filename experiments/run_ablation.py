"""
Ablation Study Experiment

Tests the contribution of each GT-Score component by removing it:
- GT-Score full (baseline)
- GT-Score without ln(z)
- GT-Score without R²
- GT-Score without σd
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gt_score import get_period_returns, find_stabilized_variance
from src import optimize, run_backtest
from strategies import strategies
from data import fetch_multiple, TEST_ASSETS

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def gt_score_without_ln_z(backtest_results, stabilize=False, mode="trades"):
    """GT-Score variant without ln(z) component."""
    if stabilize:
        num_periods = find_stabilized_variance(backtest_results['portfolio_values_over_time'])
    else:
        num_periods = 50
    
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
    
    # Without ln(z) - use z directly
    gt = (mu * z * r2) / sigma_d
    return -gt


def gt_score_without_r2(backtest_results, stabilize=False, mode="trades"):
    """GT-Score variant without R² component."""
    if stabilize:
        num_periods = find_stabilized_variance(backtest_results['portfolio_values_over_time'])
    else:
        num_periods = 50
    
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
    
    negative_returns = [r for r in period_percentage_returns if r < 0]
    sigma_d = np.std(negative_returns) if negative_returns else 1e-6
    sigma = np.std(period_percentage_returns)
    
    z = (mu - mum) / (sigma / np.sqrt(num_trades))
    
    if z <= 0:
        return 100 + (100 * (1 - math.exp(-abs(z - 1))))
    elif z <= 1:
        return 100 * (1 - math.exp(-abs(z - 1)))
    
    ln_z = math.log(z)
    # Without R² - just use mu and ln(z)
    gt = (mu * ln_z) / sigma_d
    return -gt


def gt_score_without_sigma_d(backtest_results, stabilize=False, mode="trades"):
    """GT-Score variant without σd component (uses 1 instead)."""
    if stabilize:
        num_periods = find_stabilized_variance(backtest_results['portfolio_values_over_time'])
    else:
        num_periods = 50
    
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
    sigma = np.std(period_percentage_returns)
    
    z = (mu - mum) / (sigma / np.sqrt(num_trades))
    
    if z <= 0:
        return 100 + (100 * (1 - math.exp(-abs(z - 1))))
    elif z <= 1:
        return 100 * (1 - math.exp(-abs(z - 1)))
    
    ln_z = math.log(z)
    # Without σd - use 1 as denominator
    gt = mu * ln_z * r2
    return -gt


# Ablation variants
ABLATION_VARIANTS = {
    'full': None,  # Will use standard gt_score
    'no_ln_z': gt_score_without_ln_z,
    'no_r2': gt_score_without_r2,
    'no_sigma_d': gt_score_without_sigma_d,
}


def run_ablation_study(assets=None, strategies_to_run=None, seeds=None, max_evals=50):
    """
    Run ablation study comparing GT-Score variants.
    
    Parameters
    ----------
    assets : list
        Asset tickers to test.
    strategies_to_run : list
        Strategy indices.
    seeds : list
        Random seeds.
    max_evals : int
        Optimizer iterations.
    """
    from src import gt_score
    
    if assets is None:
        assets = TEST_ASSETS[:10]
    if strategies_to_run is None:
        strategies_to_run = [0]
    if seeds is None:
        seeds = list(range(42, 52))  # 10 seeds
    
    # Use full gt_score from the module
    variants = {
        'full': gt_score,
        'no_ln_z': gt_score_without_ln_z,
        'no_r2': gt_score_without_r2,
        'no_sigma_d': gt_score_without_sigma_d,
    }
    
    print("Fetching data...")
    data = fetch_multiple(assets)
    
    total = len(assets) * len(variants) * len(strategies_to_run) * len(seeds)
    print(f"Running ablation study: {total} trials")
    
    results = []
    
    pbar = tqdm(total=total, desc="Ablation Study")
    
    for asset in assets:
        asset_data = data.get(asset)
        if asset_data is None:
            continue
        
        # Split data
        n = len(asset_data)
        train_end = int(n * 0.7)
        train_data = asset_data.iloc[:train_end].copy().reset_index(drop=True)
        test_data = asset_data.iloc[train_end:].copy().reset_index(drop=True)
        
        train_df = {'ohlc': train_data}
        
        for variant_name, loss_fn in variants.items():
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
                        
                        # Evaluate on test
                        test_signals = strat['strategy'](test_data, opt_result['best_params'])
                        test_backtest, _ = run_backtest(test_signals)
                        
                        results.append({
                            'asset': asset,
                            'variant': variant_name,
                            'strategy': strat_name,
                            'seed': seed,
                            'train_loss': opt_result['best_loss'],
                            'test_return': test_backtest.get('total_percentage_gain', 0),
                            'test_trades': test_backtest.get('total_trades', 0),
                        })
                    
                    except Exception as e:
                        results.append({
                            'asset': asset,
                            'variant': variant_name,
                            'strategy': strat_name,
                            'seed': seed,
                            'error': str(e)
                        })
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Save results
    output_path = OUTPUT_DIR / "ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    import pandas as pd
    df = pd.DataFrame([r for r in results if 'error' not in r])
    if len(df) > 0:
        summary = df.groupby('variant')['test_return'].agg(['mean', 'std', 'count'])
        print("\nAblation Study Summary:")
        print(summary)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument('--assets', type=int, default=10)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--max-evals', type=int, default=50)
    
    args = parser.parse_args()
    
    run_ablation_study(
        assets=TEST_ASSETS[:args.assets],
        seeds=list(range(42, 42 + args.seeds)),
        max_evals=args.max_evals
    )
