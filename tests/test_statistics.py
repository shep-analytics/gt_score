"""
Unit Tests for Statistical Functions

Tests for paired t-tests, bootstrap confidence intervals, effect sizes,
and probability of backtest overfitting calculations.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statistics import (
    paired_ttest,
    bootstrap_ci,
    cohens_d,
    overfitting_ratio,
    probability_of_backtest_overfitting,
    multiple_testing_adjustment,
    summary_statistics
)


class TestPairedTTest:
    """Test paired t-test implementation."""
    
    def test_significant_difference(self):
        """Two clearly different groups should have low p-value."""
        gt_results = [0.15, 0.16, 0.14, 0.17, 0.15, 0.18, 0.16, 0.14]
        baseline = [0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.08, 0.07]
        
        result = paired_ttest(gt_results, baseline)
        
        assert result['p_value'] < 0.05
        assert result['significant_05'] is True
        assert result['mean_difference'] > 0
    
    def test_no_significant_difference(self):
        """Similar groups should have high p-value."""
        group1 = [0.10, 0.11, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10]
        group2 = [0.10, 0.10, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11]
        
        result = paired_ttest(group1, group2)
        
        assert result['p_value'] > 0.05
        assert result['significant_05'] is False
    
    def test_returns_required_keys(self):
        """Result should contain all required keys."""
        gt_results = [0.15, 0.16, 0.14, 0.17, 0.15]
        baseline = [0.08, 0.09, 0.07, 0.10, 0.08]
        
        result = paired_ttest(gt_results, baseline)
        
        required_keys = ['t_statistic', 'p_value', 'p_value_one_tailed',
                         'significant_05', 'significant_01', 'mean_difference',
                         'ci_95', 'n']
        for key in required_keys:
            assert key in result
    
    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError):
            paired_ttest([1, 2, 3], [1, 2])


class TestBootstrapCI:
    """Test bootstrap confidence interval calculation."""
    
    def test_mean_ci(self):
        """Test bootstrap CI for mean."""
        data = [0.10, 0.12, 0.11, 0.09, 0.13, 0.11, 0.10, 0.12]
        
        mean_val, lower, upper = bootstrap_ci(data, n_bootstrap=1000)
        
        # Mean should be between bounds
        assert lower <= mean_val <= upper
        # CI should contain the sample mean
        assert 0.08 < lower < mean_val
        assert mean_val < upper < 0.15
    
    def test_median_ci(self):
        """Test bootstrap CI for median."""
        data = [0.10, 0.12, 0.11, 0.09, 0.13, 0.11, 0.10, 0.12]
        
        median_val, lower, upper = bootstrap_ci(data, statistic='median')
        
        assert lower <= median_val <= upper
    
    def test_confidence_level(self):
        """Higher confidence should give wider CI."""
        data = [0.10, 0.12, 0.11, 0.09, 0.13, 0.11, 0.10, 0.12]
        
        _, lower_95, upper_95 = bootstrap_ci(data, ci=0.95, n_bootstrap=5000)
        _, lower_99, upper_99 = bootstrap_ci(data, ci=0.99, n_bootstrap=5000)
        
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        
        # 99% CI should be wider than 95% CI
        assert width_99 >= width_95 * 0.9  # Allow some randomness


class TestCohensD:
    """Test Cohen's d effect size calculation."""
    
    def test_large_effect(self):
        """Clearly different groups should have d > 0.8."""
        group1 = [0.20, 0.22, 0.18, 0.21, 0.19]
        group2 = [0.08, 0.09, 0.07, 0.10, 0.08]
        
        d = cohens_d(group1, group2)
        
        assert d > 0.8  # Large effect
    
    def test_small_effect(self):
        """Similar groups should have small d."""
        group1 = [0.10, 0.11, 0.09, 0.10, 0.11]
        group2 = [0.09, 0.10, 0.12, 0.08, 0.11]
        
        d = cohens_d(group1, group2)
        
        assert abs(d) < 0.5  # Small effect
    
    def test_direction(self):
        """Sign should indicate which group is larger."""
        group1 = [0.20, 0.22, 0.18]
        group2 = [0.08, 0.09, 0.07]
        
        d_pos = cohens_d(group1, group2)  # group1 > group2
        d_neg = cohens_d(group2, group1)  # group2 < group1
        
        assert d_pos > 0
        assert d_neg < 0
        assert abs(d_pos - (-d_neg)) < 0.01  # Should be opposite


class TestOverfittingRatio:
    """Test overfitting ratio calculation."""
    
    def test_perfect_generalization(self):
        """Same train and val return should give ratio = 1."""
        ratio = overfitting_ratio(0.10, 0.10)
        assert ratio == pytest.approx(1.0)
    
    def test_overfitting(self):
        """Lower val than train should give ratio < 1."""
        ratio = overfitting_ratio(0.15, 0.05)
        assert ratio < 1.0
    
    def test_negative_train(self):
        """Should handle negative training returns."""
        ratio = overfitting_ratio(-0.10, -0.05)
        assert ratio == pytest.approx(0.5)
    
    def test_zero_train(self):
        """Zero training return should return 0."""
        ratio = overfitting_ratio(0.0, 0.05)
        assert ratio == 0.0


class TestPBO:
    """Test Probability of Backtest Overfitting."""
    
    def test_returns_required_keys(self):
        """Result should contain all required keys."""
        train = [0.15, 0.12, 0.18, 0.10, 0.14]
        val = [0.08, 0.06, 0.09, 0.05, 0.07]
        
        result = probability_of_backtest_overfitting(train, val, n_permutations=100)
        
        required_keys = ['pbo', 'best_is_rank_oos', 'correlation', 'n_configs']
        for key in required_keys:
            assert key in result
    
    def test_pbo_range(self):
        """PBO should be between 0 and 1."""
        train = [0.15, 0.12, 0.18, 0.10, 0.14]
        val = [0.08, 0.06, 0.09, 0.05, 0.07]
        
        result = probability_of_backtest_overfitting(train, val, n_permutations=100)
        
        assert 0.0 <= result['pbo'] <= 1.0


class TestMultipleTestingAdjustment:
    """Test multiple testing correction methods."""
    
    def test_bonferroni(self):
        """Bonferroni should multiply by n."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        adjusted = multiple_testing_adjustment(p_values, method='bonferroni')
        
        assert adjusted[0] == pytest.approx(0.05)  # 0.01 * 5
        assert adjusted[-1] == pytest.approx(0.25)  # 0.05 * 5
    
    def test_holm(self):
        """Holm should be less conservative than Bonferroni."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        bonf = multiple_testing_adjustment(p_values, method='bonferroni')
        holm = multiple_testing_adjustment(p_values, method='holm')
        
        # Holm should be less conservative or equal
        for b, h in zip(bonf, holm):
            assert h <= b + 0.001  # Allow small numerical error
    
    def test_fdr(self):
        """FDR should be less conservative than Bonferroni."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        bonf = multiple_testing_adjustment(p_values, method='bonferroni')
        fdr = multiple_testing_adjustment(p_values, method='fdr_bh')
        
        # FDR should generally be less conservative
        assert sum(fdr) <= sum(bonf) + 0.001
    
    def test_caps_at_one(self):
        """Adjusted p-values should not exceed 1."""
        p_values = [0.30, 0.40, 0.50]
        
        adjusted = multiple_testing_adjustment(p_values, method='bonferroni')
        
        for p in adjusted:
            assert p <= 1.0


class TestSummaryStatistics:
    """Test summary statistics calculation."""
    
    def test_returns_all_stats(self):
        """Should return all expected statistics."""
        data = [0.10, 0.12, 0.11, 0.09, 0.13, 0.11, 0.10, 0.12]
        
        stats = summary_statistics(data)
        
        expected_keys = ['n', 'mean', 'std', 'median', 'q25', 'q75', 
                        'iqr', 'min', 'max', 'ci_95_lower', 'ci_95_upper']
        for key in expected_keys:
            assert key in stats
    
    def test_correct_values(self):
        """Values should be calculated correctly."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        stats = summary_statistics(data)
        
        assert stats['n'] == 5
        assert stats['mean'] == pytest.approx(3.0)
        assert stats['median'] == pytest.approx(3.0)
        assert stats['min'] == pytest.approx(1.0)
        assert stats['max'] == pytest.approx(5.0)
    
    def test_empty_data(self):
        """Should handle empty data gracefully."""
        stats = summary_statistics([])
        
        assert stats['n'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
