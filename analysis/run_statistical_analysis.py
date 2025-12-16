"""
Statistical Analysis Runner

Loads Monte Carlo results and performs formal statistical tests:
- Paired t-tests comparing GT-Score vs baselines
- Effect size calculations (Cohen's d)
- Bootstrap confidence intervals
- Multiple testing corrections
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.statistics import (
    paired_ttest,
    wilcoxon_signed_rank,
    bootstrap_ci,
    cohens_d,
    overfitting_ratio,
    probability_of_backtest_overfitting,
    multiple_testing_adjustment,
    summary_statistics
)

OUTPUT_DIR = Path(__file__).parent.parent / "output"
RESULTS_DIR = OUTPUT_DIR / "results"


def load_results(filename="monte_carlo_results.json"):
    """Load experiment results from JSON file."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def run_pairwise_comparisons(df, metric='test_return', gt_name='gt_score'):
    """
    Run pairwise statistical comparisons between GT-Score and baselines.
    
    Returns dict with test results for each comparison.
    """
    results = {}
    
    baseline_names = [n for n in df['loss_function'].unique() if n != gt_name]
    
    for baseline in baseline_names:
        # Get matched pairs (same asset, strategy, seed)
        gt_df = df[df['loss_function'] == gt_name].copy()
        bl_df = df[df['loss_function'] == baseline].copy()
        
        # Merge on matching keys
        merged = pd.merge(
            gt_df[['asset', 'strategy', 'seed', metric]],
            bl_df[['asset', 'strategy', 'seed', metric]],
            on=['asset', 'strategy', 'seed'],
            suffixes=('_gt', '_baseline')
        )
        
        if len(merged) < 5:
            results[baseline] = {'error': 'Insufficient matched pairs'}
            continue
        
        gt_values = merged[f'{metric}_gt'].values
        bl_values = merged[f'{metric}_baseline'].values
        
        # Paired t-test
        ttest_result = paired_ttest(gt_values.tolist(), bl_values.tolist())

        # Paired non-parametric test (robust to non-normal differences)
        wilcoxon_result = wilcoxon_signed_rank(gt_values.tolist(), bl_values.tolist())
        
        # Effect size
        effect_size = cohens_d(gt_values.tolist(), bl_values.tolist())
        
        # Bootstrap CI for mean difference
        differences = gt_values - bl_values
        mean_diff, ci_lower, ci_upper = bootstrap_ci(differences.tolist())
        
        results[baseline] = {
            't_statistic': ttest_result['t_statistic'],
            'p_value': ttest_result['p_value'],
            'significant_05': ttest_result['significant_05'],
            'significant_01': ttest_result['significant_01'],
            'wilcoxon_statistic': wilcoxon_result['statistic'],
            'wilcoxon_p_value': wilcoxon_result['p_value'],
            'wilcoxon_significant_05': wilcoxon_result['significant_05'],
            'wilcoxon_significant_01': wilcoxon_result['significant_01'],
            'mean_gt': np.mean(gt_values),
            'mean_baseline': np.mean(bl_values),
            'mean_difference': mean_diff,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'cohens_d': effect_size,
            'effect_interpretation': (
                'large' if abs(effect_size) > 0.8 else
                'medium' if abs(effect_size) > 0.5 else
                'small'
            ),
            'n_pairs': len(merged)
        }
    
    return results


def run_overfitting_analysis(df):
    """Analyze overfitting metrics across loss functions."""
    results = {}
    
    for loss_name in df['loss_function'].unique():
        loss_df = df[df['loss_function'] == loss_name]
        
        # Get train/test returns
        train_returns = loss_df['train_return'].dropna().values if 'train_return' in loss_df else []
        test_returns = loss_df['test_return'].dropna().values
        
        if len(train_returns) > 0 and len(test_returns) > 0:
            # Calculate overfitting ratios
            ratios = []
            for t, v in zip(train_returns, test_returns):
                if t != 0:
                    ratios.append(overfitting_ratio(t, v))
            
            if ratios:
                results[loss_name] = {
                    'mean_overfitting_ratio': np.mean(ratios),
                    'std_overfitting_ratio': np.std(ratios),
                    'pct_overfitting': np.mean([r < 1 for r in ratios]) * 100,
                    'n_samples': len(ratios)
                }
        
        # Summary stats for test returns
        test_stats = summary_statistics(test_returns.tolist())
        results[loss_name] = results.get(loss_name, {})
        results[loss_name].update({
            'test_return_mean': test_stats['mean'],
            'test_return_std': test_stats['std'],
            'test_return_ci_lower': test_stats['ci_95_lower'],
            'test_return_ci_upper': test_stats['ci_95_upper']
        })
    
    return results


def apply_multiple_testing_corrections(pairwise_results):
    """Apply multiple testing corrections to pairwise p-values."""
    p_values = []
    baselines = []
    
    for baseline, result in pairwise_results.items():
        if 'p_value' in result:
            p_values.append(result['p_value'])
            baselines.append(baseline)
    
    if not p_values:
        return pairwise_results
    
    # Apply corrections
    bonf = multiple_testing_adjustment(p_values, method='bonferroni')
    holm = multiple_testing_adjustment(p_values, method='holm')
    fdr = multiple_testing_adjustment(p_values, method='fdr_bh')
    
    # Add corrected p-values back
    for i, baseline in enumerate(baselines):
        pairwise_results[baseline]['p_bonferroni'] = bonf[i]
        pairwise_results[baseline]['p_holm'] = holm[i]
        pairwise_results[baseline]['p_fdr'] = fdr[i]
        pairwise_results[baseline]['significant_holm_05'] = holm[i] < 0.05
        pairwise_results[baseline]['significant_fdr_05'] = fdr[i] < 0.05
    
    return pairwise_results


def run_statistical_analysis(results_file="monte_carlo_results.json"):
    """Run complete statistical analysis on experiment results."""
    print("Loading results...")
    try:
        raw_results = load_results(results_file)
    except FileNotFoundError:
        print(f"Results file not found. Run experiments first.")
        return None
    
    # Filter out errors and convert to DataFrame
    valid_results = [r for r in raw_results if 'error' not in r]
    df = pd.DataFrame(valid_results)
    
    if len(df) == 0:
        print("No valid results to analyze")
        return None
    
    print(f"Analyzing {len(df)} valid results...")
    
    analysis = {
        'summary': {
            'n_total_results': len(raw_results),
            'n_valid_results': len(df),
            'n_errors': len(raw_results) - len(df),
            'loss_functions': df['loss_function'].unique().tolist() if 'loss_function' in df else [],
            'assets': df['asset'].nunique() if 'asset' in df else 0,
        }
    }
    
    # Pairwise comparisons
    print("Running pairwise comparisons...")
    if 'loss_function' in df.columns and 'test_return' in df.columns:
        pairwise = run_pairwise_comparisons(df)
        pairwise = apply_multiple_testing_corrections(pairwise)
        analysis['pairwise_comparisons'] = pairwise
    
    # Overfitting analysis
    print("Running overfitting analysis...")
    if 'loss_function' in df.columns:
        overfitting = run_overfitting_analysis(df)
        analysis['overfitting_analysis'] = overfitting
    
    # Per-asset breakdown
    print("Computing per-asset breakdown...")
    if 'asset' in df.columns and 'loss_function' in df.columns:
        per_asset = {}
        for asset in df['asset'].unique():
            asset_df = df[df['asset'] == asset]
            per_asset[asset] = {}
            for loss in asset_df['loss_function'].unique():
                loss_df = asset_df[asset_df['loss_function'] == loss]
                if 'test_return' in loss_df:
                    per_asset[asset][loss] = {
                        'mean_return': loss_df['test_return'].mean(),
                        'std_return': loss_df['test_return'].std(),
                        'n_trials': len(loss_df)
                    }
        analysis['per_asset'] = per_asset
    
    # Save results
    output_path = RESULTS_DIR / "statistical_tests.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    
    if 'pairwise_comparisons' in analysis:
        print("\nPairwise Comparisons (GT-Score vs Baseline):")
        print("-"*60)
        for baseline, result in analysis['pairwise_comparisons'].items():
            if 'p_value' in result:
                sig = "***" if result['significant_01'] else ("*" if result['significant_05'] else "")
                print(f"  vs {baseline:12s}: p={result['p_value']:.4f}{sig}, "
                      f"d={result['cohens_d']:.3f} ({result['effect_interpretation']})")
    
    if 'overfitting_analysis' in analysis:
        print("\nOverfitting Analysis:")
        print("-"*60)
        for loss, stats in analysis['overfitting_analysis'].items():
            if 'test_return_mean' in stats:
                print(f"  {loss:12s}: mean={stats['test_return_mean']:.4f} "
                      f"(95% CI: [{stats['test_return_ci_lower']:.4f}, {stats['test_return_ci_upper']:.4f}])")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run statistical analysis")
    parser.add_argument('--results-file', default='monte_carlo_results.json')
    
    args = parser.parse_args()
    run_statistical_analysis(args.results_file)
