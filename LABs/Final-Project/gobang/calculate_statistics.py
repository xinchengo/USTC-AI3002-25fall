#!/usr/bin/env python3
"""
Script to calculate statistical results for the Gobang AI experiments.
This ensures all calculations are done by executing Python scripts rather than internally in the LLM.

[TODO(AGENT)]: output format from .4f --> exponential for very small p-values
    [LINKUPDATE log.md]

"""

import numpy as np
from scipy import stats
import math

def calculate_binomial_test(x, n, p_null=0.5):
    """
    Calculate binomial test statistics
    :param x: number of successes
    :param n: number of trials
    :param p_null: null hypothesis probability (default 0.5)
    :return: test statistic, p-value, rejection_threshold
    """
    # Calculate mean and std under null hypothesis
    mean_null = n * p_null
    var_null = n * p_null * (1 - p_null)
    std_null = math.sqrt(var_null)
    
    # Calculate z-score
    z_score = (x - mean_null) / std_null
    
    # For two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Rejection threshold for alpha=0.01 (two-tailed)
    z_critical = stats.norm.ppf(1 - 0.01/2)  # Approximately 2.576
    rejection_threshold = z_critical * std_null
    
    return z_score, p_value, rejection_threshold, mean_null, std_null

def calculate_two_proportion_z_test(x1, x2, n1, n2):
    """
    Calculate two proportion z-test
    :param x1: successes for group 1
    :param x2: successes for group 2
    :param n1: trials for group 1
    :param n2: trials for group 2
    :return: z-statistic, p-value
    """
    p1_hat = x1 / n1
    p2_hat = x2 / n2
    
    # Pooled proportion
    p_pooled = (x1 + x2) / (n1 + n2)
    
    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Z-statistic
    z_stat = (p1_hat - p2_hat) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value, p1_hat, p2_hat

def main():
    print("Gobang AI Experiment Statistical Calculations")
    print("="*50)
    
    # Regular CNN results
    print("\nRegular CNN (1000 epoch vs 800 epoch):")
    print("H0: p = 0.5 vs H1: p ≠ 0.5")
    x_regular = 245  # 1000 epoch model wins
    n_regular = 500
    z_reg, p_reg, thresh_reg, mean_reg, std_reg = calculate_binomial_test(x_regular, n_regular)
    print(f"X = {x_regular}")
    print(f"Mean under H0: {mean_reg}")
    print(f"Std under H0: {std_reg:.2f}")
    print(f"Rejection threshold (α=0.01): ±{thresh_reg:.2f}")
    print(f"Observed difference from mean: {abs(x_regular - mean_reg)}")
    print(f"Is |observed - mean| > threshold? {abs(x_regular - mean_reg) > thresh_reg}")
    print(f"p-value: {p_reg:.3f}")
    
    # Deep CNN results
    print("\nDeep CNN (1000 epoch vs 800 epoch):")
    print("H0: p = 0.5 vs H1: p ≠ 0.5")
    x_deep = 254  # 1000 epoch model wins
    n_deep = 500
    z_deep, p_deep, thresh_deep, mean_deep, std_deep = calculate_binomial_test(x_deep, n_deep)
    print(f"X = {x_deep}")
    print(f"Mean under H0: {mean_deep}")
    print(f"Std under H0: {std_deep:.2f}")
    print(f"Rejection threshold (α=0.01): ±{thresh_deep:.2f}")
    print(f"Observed difference from mean: {abs(x_deep - mean_deep)}")
    print(f"Is |observed - mean| > threshold? {abs(x_deep - mean_deep) > thresh_deep}")
    print(f"p-value: {p_deep:.3f}")
    
    # Comparison between regular and deep CNN
    print("\nComparison between Regular and Deep CNN:")
    print("H0: p_regular = p_deep vs H1: p_regular ≠ p_deep")
    wins_regular = 6   # Regular CNN wins
    wins_deep = 94     # Deep CNN wins
    n_comparison = 100
    
    z_comp, p_comp, p1_hat, p2_hat = calculate_two_proportion_z_test(
        wins_regular, wins_deep, n_comparison, n_comparison
    )
    
    print(f"Regular CNN win rate: {p1_hat:.3f} ({wins_regular}/{n_comparison})")
    print(f"Deep CNN win rate: {p2_hat:.3f} ({wins_deep}/{n_comparison})")
    print(f"Z-statistic: {z_comp:.2f}")
    print(f"p-value: {p_comp:.4f}")
    
    print("\nSummary:")
    print(f"- Regular CNN convergence test p-value: {p_reg:.3f}")
    print(f"- Deep CNN convergence test p-value: {p_deep:.3f}")
    print(f"- CNN depth comparison p-value: {p_comp:.4f}")
    
    if p_reg > 0.01:
        print("- Regular CNN has converged (fail to reject H0 at α=0.01)")
    else:
        print("- Regular CNN has NOT converged (reject H0 at α=0.01)")
        
    if p_deep > 0.01:
        print("- Deep CNN has converged (fail to reject H0 at α=0.01)")
    else:
        print("- Deep CNN has NOT converged (reject H0 at α=0.01)")
        
    if p_comp < 0.01:
        print("- Significant difference between regular and deep CNN (p < 0.01)")
    else:
        print("- No significant difference between regular and deep CNN (p ≥ 0.01)")

if __name__ == "__main__":
    main()