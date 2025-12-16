"""
Statistical Tests for Trading Strategy Evaluation

This module provides formal statistical tests for comparing trading strategy
performance, including:
- Paired t-tests for comparing GT-Score vs baselines
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d)
- Probability of Backtest Overfitting (PBO)
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional


def paired_ttest(gt_results: List[float], baseline_results: List[float]) -> dict:
    """
    Perform a paired t-test comparing GT-Score optimized results vs baseline.
    
    Used to determine if GT-Score produces significantly different (better)
    results compared to an alternative loss function.
    
    Parameters
    ----------
    gt_results : list of float
        Performance metrics (e.g., validation returns) from GT-Score optimization.
    baseline_results : list of float
        Corresponding metrics from baseline optimization.
    
    Returns
    -------
    dict
        - 't_statistic': t-test statistic
        - 'p_value': two-tailed p-value
        - 'p_value_one_tailed': one-tailed p-value (GT > baseline)
        - 'significant_05': True if p < 0.05
        - 'significant_01': True if p < 0.01
        - 'mean_difference': mean(GT) - mean(baseline)
        - 'ci_95': 95% confidence interval for the mean difference
    
    Examples
    --------
    >>> gt_returns = [0.12, 0.15, 0.10, 0.18, 0.14]
    >>> sharpe_returns = [0.08, 0.11, 0.07, 0.12, 0.09]
    >>> result = paired_ttest(gt_returns, sharpe_returns)
    >>> print(f"p-value: {result['p_value']:.4f}")
    >>> print(f"Significant at 0.05: {result['significant_05']}")
    """
    gt_arr = np.array(gt_results)
    baseline_arr = np.array(baseline_results)
    
    if len(gt_arr) != len(baseline_arr):
        raise ValueError("Arrays must be the same length for paired t-test")
    
    n = len(gt_arr)
    differences = gt_arr - baseline_arr
    
    t_stat, p_val = stats.ttest_rel(gt_arr, baseline_arr)
    
    # One-tailed p-value (GT > baseline)
    p_one_tailed = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    
    # 95% CI for the mean difference
    mean_diff = np.mean(differences)
    se_diff = stats.sem(differences)
    ci_95 = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se_diff)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'p_value_one_tailed': float(p_one_tailed),
        'significant_05': bool(p_val < 0.05),
        'significant_01': bool(p_val < 0.01),
        'mean_difference': float(mean_diff),
        'ci_95': (float(ci_95[0]), float(ci_95[1])),
        'n': int(n)
    }


def wilcoxon_signed_rank(
    gt_results: List[float],
    baseline_results: List[float],
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> dict:
    """
    Perform a paired non-parametric Wilcoxon signed-rank test.

    This test is a robust alternative to the paired t-test when the
    distribution of paired differences is non-normal (common with
    heavy-tailed return distributions).

    Parameters
    ----------
    gt_results : list of float
        Paired performance metrics from GT-Score optimization.
    baseline_results : list of float
        Paired performance metrics from baseline optimization.
    alternative : str, default='two-sided'
        {'two-sided', 'greater', 'less'}.
    zero_method : str, default='wilcox'
        How to handle zero-differences: {'wilcox', 'pratt', 'zsplit'}.

    Returns
    -------
    dict
        - 'statistic': Wilcoxon test statistic
        - 'p_value': p-value
        - 'significant_05': True if p < 0.05
        - 'significant_01': True if p < 0.01
        - 'n': number of paired observations used
    """
    gt_arr = np.asarray(gt_results, dtype=float)
    baseline_arr = np.asarray(baseline_results, dtype=float)

    if len(gt_arr) != len(baseline_arr):
        raise ValueError("Arrays must be the same length for Wilcoxon signed-rank test")

    # Drop NaNs (if any) in paired fashion.
    valid_mask = ~np.isnan(gt_arr) & ~np.isnan(baseline_arr)
    gt_arr = gt_arr[valid_mask]
    baseline_arr = baseline_arr[valid_mask]

    if gt_arr.size == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant_05': False,
            'significant_01': False,
            'n': 0
        }

    try:
        stat, p_val = stats.wilcoxon(
            gt_arr,
            baseline_arr,
            alternative=alternative,
            zero_method=zero_method,
            method="auto",
        )
    except TypeError:
        # Backwards compatibility for older SciPy without `method=`.
        stat, p_val = stats.wilcoxon(
            gt_arr,
            baseline_arr,
            alternative=alternative,
            zero_method=zero_method,
        )
    except ValueError:
        # Can occur when all paired differences are zero.
        stat, p_val = 0.0, 1.0

    return {
        'statistic': float(stat),
        'p_value': float(p_val),
        'significant_05': float(p_val) < 0.05,
        'significant_01': float(p_val) < 0.01,
        'n': int(gt_arr.size)
    }


def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, 
                 ci: float = 0.95, statistic: str = 'mean') -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : list of float
        Sample data.
    n_bootstrap : int, default=10000
        Number of bootstrap resamples.
    ci : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI).
    statistic : str, default='mean'
        Statistic to compute: 'mean', 'median', 'std'.
    
    Returns
    -------
    tuple
        (point_estimate, lower_bound, upper_bound)
    
    Examples
    --------
    >>> returns = [0.05, 0.08, 0.03, 0.12, 0.07, 0.09]
    >>> mean, lower, upper = bootstrap_ci(returns)
    >>> print(f"Mean: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    data = np.array(data)
    n = len(data)
    
    stat_funcs = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std
    }
    
    if statistic not in stat_funcs:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    stat_func = stat_funcs[statistic]
    point_estimate = stat_func(data)
    
    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(resample))
    
    # Calculate percentile CI
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, lower, upper


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Cohen's d measures the standardized difference between two means.
    Interpretation guidelines:
    - Small: d ≈ 0.2
    - Medium: d ≈ 0.5
    - Large: d ≈ 0.8
    
    Parameters
    ----------
    group1 : list of float
        First group of observations (e.g., GT-Score results).
    group2 : list of float
        Second group of observations (e.g., baseline results).
    
    Returns
    -------
    float
        Cohen's d effect size. Positive if group1 > group2.
    
    Examples
    --------
    >>> gt_returns = [0.12, 0.15, 0.10, 0.18, 0.14]
    >>> baseline_returns = [0.08, 0.11, 0.07, 0.12, 0.09]
    >>> d = cohens_d(gt_returns, baseline_returns)
    >>> print(f"Effect size: {d:.2f}")  # Large positive = GT much better
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return float(d)


def overfitting_ratio(train_return: float, val_return: float) -> float:
    """
    Calculate the overfitting ratio.
    
    A ratio < 1 indicates the strategy performs worse out-of-sample,
    suggesting overfitting. Values close to 1 suggest good generalization.
    
    Parameters
    ----------
    train_return : float
        In-sample (training) return.
    val_return : float
        Out-of-sample (validation) return.
    
    Returns
    -------
    float
        Validation Return / Training Return.
        Returns 0 if train_return is 0.
    
    Notes
    -----
    Interpretation:
    - Ratio > 1: Strategy improves OOS (rare, often lucky)
    - Ratio ≈ 1: Good generalization
    - Ratio < 1: Overfitting (performance degrades OOS)
    - Ratio < 0: Strategy loses money OOS when it made money IS
    """
    if train_return == 0:
        return 0.0
    return val_return / train_return


def probability_of_backtest_overfitting(train_results: List[float], 
                                         val_results: List[float],
                                         n_permutations: int = 10000) -> dict:
    """
    Calculate the Probability of Backtest Overfitting (PBO).
    
    PBO estimates the probability that the best in-sample strategy
    underperforms the median out-of-sample. High PBO (> 0.5) suggests
    significant overfitting risk.
    
    Parameters
    ----------
    train_results : list of float
        In-sample performance for each configuration tested.
    val_results : list of float
        Corresponding out-of-sample performance.
    n_permutations : int, default=10000
        Number of permutations for significance testing.
    
    Returns
    -------
    dict
        - 'pbo': Probability of backtest overfitting
        - 'best_is_rank_oos': OOS rank of best IS performer
        - 'correlation': Spearman correlation between IS and OOS rankings
        - 'n_configs': Number of configurations tested
    
    References
    ----------
    Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
    Journal of Portfolio Management.
    """
    train_arr = np.array(train_results)
    val_arr = np.array(val_results)
    
    if len(train_arr) != len(val_arr):
        raise ValueError("Training and validation results must be same length")
    
    n = len(train_arr)
    
    # Find best in-sample performer
    best_is_idx = np.argmax(train_arr)
    best_is_oos_performance = val_arr[best_is_idx]
    
    # Rank of best IS performer in OOS
    oos_ranks = stats.rankdata(-val_arr)  # Negative for descending rank
    best_is_rank_oos = oos_ranks[best_is_idx]
    
    # Spearman correlation between IS and OOS rankings
    is_ranks = stats.rankdata(-train_arr)
    correlation, corr_pval = stats.spearmanr(is_ranks, oos_ranks)
    
    # PBO: proportion of times best IS underperforms median OOS
    oos_median = np.median(val_arr)
    
    # Permutation test for PBO
    underperform_count = 0
    for _ in range(n_permutations):
        # Randomly assign IS/OOS labels
        perm = np.random.permutation(n)
        half = n // 2
        perm_is = train_arr[perm[:half]]
        perm_oos = val_arr[perm[half:]]
        
        best_perm_is_idx = np.argmax(perm_is)
        best_perm_is_oos = perm_oos[best_perm_is_idx] if best_perm_is_idx < len(perm_oos) else 0
        
        if best_perm_is_oos < np.median(perm_oos):
            underperform_count += 1
    
    pbo = underperform_count / n_permutations
    
    return {
        'pbo': pbo,
        'best_is_rank_oos': best_is_rank_oos,
        'correlation': correlation,
        'correlation_pval': corr_pval,
        'n_configs': n,
        'interpretation': 'High overfitting risk' if pbo > 0.5 else 'Acceptable'
    }


def multiple_testing_adjustment(p_values: List[float], method: str = 'holm') -> List[float]:
    """
    Adjust p-values for multiple hypothesis testing.
    
    When testing many hypotheses simultaneously (e.g., comparing GT-Score
    to multiple baselines), raw p-values need adjustment to control
    the family-wise error rate or false discovery rate.
    
    Parameters
    ----------
    p_values : list of float
        Raw p-values from individual tests.
    method : str, default='holm'
        Adjustment method:
        - 'bonferroni': Bonferroni correction (conservative)
        - 'holm': Holm-Bonferroni (step-down, less conservative)
        - 'fdr_bh': Benjamini-Hochberg FDR (controls false discovery rate)
    
    Returns
    -------
    list of float
        Adjusted p-values.
    
    References
    ----------
    Harvey, C., Liu, Y., & Zhu, H. (2016). "... and the Cross-Section 
    of Expected Returns." Review of Financial Studies.
    """
    p_arr = np.array(p_values)
    n = len(p_arr)
    
    if method == 'bonferroni':
        adjusted = np.minimum(p_arr * n, 1.0)
    
    elif method == 'holm':
        # Holm-Bonferroni step-down procedure
        sorted_idx = np.argsort(p_arr)
        adjusted = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(p_arr[idx] * (n - i), 1.0)
        # Ensure monotonicity
        for i in range(1, n):
            idx = sorted_idx[i]
            prev_idx = sorted_idx[i-1]
            adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_arr)
        adjusted = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(p_arr[idx] * n / (i + 1), 1.0)
        # Ensure monotonicity (from largest to smallest)
        for i in range(n - 2, -1, -1):
            idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]
            adjusted[idx] = min(adjusted[idx], adjusted[next_idx])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return adjusted.tolist()


def summary_statistics(results: List[float]) -> dict:
    """
    Calculate comprehensive summary statistics for results.
    
    Parameters
    ----------
    results : list of float
        Performance metrics to summarize.
    
    Returns
    -------
    dict
        Comprehensive statistics including mean, std, median, IQR,
        min/max, skewness, kurtosis, and CI.
    """
    arr = np.array(results)
    n = len(arr)
    
    if n == 0:
        return {'n': 0}
    
    mean, lower, upper = bootstrap_ci(results, ci=0.95)
    
    return {
        'n': n,
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1) if n > 1 else 0,
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
        'iqr': np.percentile(arr, 75) - np.percentile(arr, 25),
        'min': np.min(arr),
        'max': np.max(arr),
        'skewness': stats.skew(arr) if n > 2 else 0,
        'kurtosis': stats.kurtosis(arr) if n > 3 else 0,
        'ci_95_lower': lower,
        'ci_95_upper': upper
    }
