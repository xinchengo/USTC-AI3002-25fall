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
import argparse

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

def calculate_one_proportion_z_test(x, n, p_null=0.5):
    """
    Calculate one proportion z-test (for one-tailed test)
    :param x: number of successes
    :param n: number of trials
    :param p_null: null hypothesis probability (default 0.5)
    :return: z-statistic, p-value
    """
    p_hat = x / n

    # Standard error
    se = math.sqrt(p_null * (1 - p_null) / n)

    # Z-statistic
    z_stat = (p_hat - p_null) / se

    # One-tailed p-value (testing if p > p_null)
    p_value = 1 - stats.norm.cdf(z_stat)

    return z_stat, p_value, p_hat

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
    parser = argparse.ArgumentParser(description='Calculate statistical results for Gobang AI experiments')
    parser.add_argument('--experiment', type=str, choices=[
        'convergence', 'comparison', 'vs-random', 'tournament'
    ], help='Type of statistical test to perform')
    parser.add_argument('--x1', type=int, help='Number of successes for group 1')
    parser.add_argument('--x2', type=int, help='Number of successes for group 2')
    parser.add_argument('--n1', type=int, help='Number of trials for group 1')
    parser.add_argument('--n2', type=int, help='Number of trials for group 2')
    parser.add_argument('--p_null', type=float, default=0.5, help='Null hypothesis probability')
    parser.add_argument('--one_tailed', action='store_true', help='Perform one-tailed test')

    args = parser.parse_args()

    print("Gobang AI Experiment Statistical Calculations")
    print("="*50)

    if args.experiment == 'convergence':
        # Binomial test for convergence
        if args.x1 is not None and args.n1 is not None:
            z_score, p_value, _, mean_null, std_null = calculate_binomial_test(
                args.x1, args.n1, args.p_null
            )
            print(f"\nConvergence Test Results:")
            print(f"H0: p = {args.p_null} vs H1: p ≠ {args.p_null}")
            print(f"X = {args.x1}")
            print(f"n = {args.n1}")
            print(f"Mean under H0: {mean_null}")
            print(f"Std under H0: {std_null:.2f}")
            print(f"p-value: {p_value:.4f}")

            alpha = 0.01
            z_critical = stats.norm.ppf(1 - alpha/2)
            rejection_threshold = z_critical * std_null
            print(f"Rejection threshold (α={alpha}): ±{rejection_threshold:.2f}")
            print(f"Observed difference from mean: {abs(args.x1 - mean_null)}")
            print(f"Is |observed - mean| > threshold? {abs(args.x1 - mean_null) > rejection_threshold}")

            if p_value < alpha:
                print(f"Conclusion: Reject H0, model has not converged (p < {alpha})")
            else:
                print(f"Conclusion: Fail to reject H0, model has converged (p ≥ {alpha})")

    elif args.experiment == 'vs-random':
        # One-proportion test against random policy
        if args.x1 is not None and args.n1 is not None:
            z_stat, p_value, p_hat = calculate_one_proportion_z_test(
                args.x1, args.n1, args.p_null
            )
            print(f"\nPerformance vs Random Policy Test:")
            print(f"H0: p = {args.p_null} vs H1: p > {args.p_null}")
            print(f"Successes: {args.x1}")
            print(f"Trials: {args.n1}")
            print(f"Observed proportion: {p_hat:.3f}")
            print(f"p-value: {p_value:.2e}")

            alpha = 0.01
            if p_value < alpha:
                print(f"Conclusion: Strong evidence that model significantly outperforms random policy (p < {alpha})")
            else:
                print(f"Conclusion: Insufficient evidence that model outperforms random policy (p ≥ {alpha})")

    elif args.experiment == 'comparison':
        # Two-proportion test
        if args.x1 is not None and args.x2 is not None and args.n1 is not None and args.n2 is not None:
            z_stat, p_value, p1_hat, p2_hat = calculate_two_proportion_z_test(
                args.x1, args.x2, args.n1, args.n2
            )
            print(f"\nTwo-Model Comparison Test:")
            print(f"H0: p1 = p2 vs H1: p1 ≠ p2")
            print(f"Group 1: {args.x1}/{args.n1} = {p1_hat:.3f}")
            print(f"Group 2: {args.x2}/{args.n2} = {p2_hat:.3f}")
            print(f"Z-statistic: {z_stat:.2f}")
            print(f"p-value: {p_value:.2e}")

            alpha = 0.01
            if p_value < alpha:
                print(f"Conclusion: Strong evidence of significant difference between models (p < {alpha})")
            else:
                print(f"Conclusion: No significant difference between models (p ≥ {alpha})")

    elif args.experiment == 'tournament':
        # Placeholder for tournament statistics
        print("\nTournament Statistics Placeholder")
        print("This would analyze results from multiple model comparisons")

    else:
        # Default: Show example calculations
        print("\nExample calculations:")

        # Regular CNN results
        print("\nRegular CNN (1000 epoch vs 800 epoch):")
        print("H0: p = 0.5 vs H1: p ≠ 0.5")
        x_regular = 265  # Updated example value
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
        print("\nDeep CNN (1500 epoch vs 1300 epoch):")
        print("H0: p = 0.5 vs H1: p ≠ 0.5")
        x_deep = 278  # Updated example value
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
        print("\nComparison between Baseline and Deep CNN:")
        print("H0: p_baseline = p_deep vs H1: p_baseline ≠ p_deep")
        wins_baseline = 160   # Updated example value
        wins_deep = 340       # Updated example value
        n_comparison = 500

        z_comp, p_comp, p1_hat, p2_hat = calculate_two_proportion_z_test(
            wins_baseline, wins_deep, n_comparison, n_comparison
        )

        print(f"Baseline CNN win rate: {p1_hat:.3f} ({wins_baseline}/{n_comparison})")
        print(f"Deep CNN win rate: {p2_hat:.3f} ({wins_deep}/{n_comparison})")
        print(f"Z-statistic: {z_comp:.2f}")
        print(f"p-value: {p_comp:.2e}")

        print("\nSummary:")
        print(f"- Baseline CNN convergence test p-value: {p_reg:.3f}")
        print(f"- Deep CNN convergence test p-value: {p_deep:.3f}")
        print(f"- CNN depth comparison p-value: {p_comp:.2e}")

        if p_reg > 0.01:
            print("- Baseline CNN has converged (fail to reject H0 at α=0.01)")
        else:
            print("- Baseline CNN has NOT converged (reject H0 at α=0.01)")

        if p_deep > 0.01:
            print("- Deep CNN has converged (fail to reject H0 at α=0.01)")
        else:
            print("- Deep CNN has NOT converged (reject H0 at α=0.01)")

        if p_comp < 0.01:
            print("- Significant difference between baseline and deep CNN (p < 0.01)")
        else:
            print("- No significant difference between baseline and deep CNN (p ≥ 0.01)")

if __name__ == "__main__":
    main()