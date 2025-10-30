#!/usr/bin/env python3
"""
Example: Calculating Effective Sample Size for Income Change and Tree Cover Study

This script demonstrates how to calculate the effective sample size when you:
- Do not have repeated measurements over time
- Have missing data in your variables

Research Question: What proportion of areas show both positive income change
and increased tree cover?
"""

import numpy as np
from effective_sample_size import (
    calculate_effective_sample_size,
    calculate_effective_sample_size_multivariate,
    calculate_proportion_estimate_ci
)


def main():
    print("=" * 70)
    print("EFFECTIVE SAMPLE SIZE CALCULATION EXAMPLE")
    print("=" * 70)
    print()
    
    # Simulated data: Income change (in dollars) and Tree cover (proportion)
    # Some measurements are missing (represented as np.nan)
    print("Scenario: Income Change and Tree Cover Study")
    print("-" * 70)
    
    income_change = [
        150, 200, np.nan, 400, -50, 600, np.nan, 800, 
        100, np.nan, 250, 300, 350, np.nan, 450
    ]
    
    tree_cover_change = [
        0.05, np.nan, 0.10, 0.08, 0.02, np.nan, 0.12, 0.15,
        np.nan, 0.06, 0.09, np.nan, 0.11, 0.07, 0.13
    ]
    
    print(f"\nTotal observations collected: {len(income_change)}")
    print()
    
    # Step 1: Calculate effective sample size for each variable individually
    print("Step 1: Individual Variable Analysis")
    print("-" * 70)
    
    income_result = calculate_effective_sample_size(income_change)
    print(f"\nIncome Change Variable:")
    print(f"  Total observations: {income_result['n_total']}")
    print(f"  Missing values: {income_result['n_missing']}")
    print(f"  Effective sample size: {income_result['n_effective']}")
    print(f"  Completeness: {income_result['proportion_complete']:.1%}")
    
    tree_result = calculate_effective_sample_size(tree_cover_change)
    print(f"\nTree Cover Change Variable:")
    print(f"  Total observations: {tree_result['n_total']}")
    print(f"  Missing values: {tree_result['n_missing']}")
    print(f"  Effective sample size: {tree_result['n_effective']}")
    print(f"  Completeness: {tree_result['proportion_complete']:.1%}")
    
    # Step 2: Calculate effective sample size for multivariate analysis
    print("\n\nStep 2: Multivariate Analysis (Both Variables)")
    print("-" * 70)
    
    multi_result = calculate_effective_sample_size_multivariate(
        income_change, tree_cover_change
    )
    
    print(f"\nComplete cases (no missing in either variable):")
    print(f"  Effective sample size: {multi_result['n_effective']}")
    print(f"  Cases with any missing: {multi_result['n_missing']}")
    print(f"  Completeness rate: {multi_result['proportion_complete']:.1%}")
    print(f"\nMissing by variable:")
    print(f"  Income change: {multi_result['missing_by_variable'][0]} missing")
    print(f"  Tree cover change: {multi_result['missing_by_variable'][1]} missing")
    
    # Step 3: Analyze the complete cases
    print("\n\nStep 3: Proportion Estimate Using Effective Sample Size")
    print("-" * 70)
    
    # Get indices of complete cases
    complete_indices = []
    for i in range(len(income_change)):
        income_val = income_change[i]
        tree_val = tree_cover_change[i]
        
        income_missing = isinstance(income_val, float) and np.isnan(income_val)
        tree_missing = isinstance(tree_val, float) and np.isnan(tree_val)
        
        if not income_missing and not tree_missing:
            complete_indices.append(i)
    
    # Count "successes": cases where both income and tree cover increased
    successes = 0
    for idx in complete_indices:
        if income_change[idx] > 0 and tree_cover_change[idx] > 0:
            successes += 1
    
    print(f"\nAnalyzing {multi_result['n_effective']} complete cases...")
    print(f"Cases with positive income AND tree cover increase: {successes}")
    
    # Calculate proportion estimate and confidence interval
    prop_result = calculate_proportion_estimate_ci(
        successes=successes,
        n_effective=multi_result['n_effective'],
        confidence_level=0.95
    )
    
    print(f"\nProportion Estimate (p̂):")
    print(f"  p̂ = {prop_result['p_hat']:.3f}")
    print(f"  Standard Error = {prop_result['standard_error']:.3f}")
    print(f"  95% Confidence Interval: [{prop_result['ci_lower']:.3f}, {prop_result['ci_upper']:.3f}]")
    
    # Step 4: Interpretation
    print("\n\nStep 4: Interpretation")
    print("-" * 70)
    print(f"""
Answer to the question: "What is your effective sample size for estimating p̂?"

For this study with:
- No repeated measurements over time
- Missing data in both variables
- Multivariate analysis (income change AND tree cover)

The effective sample size is: {multi_result['n_effective']} observations

This represents {multi_result['proportion_complete']:.1%} of the total {multi_result['n_total']} 
observations collected.

The proportion estimate p̂ = {prop_result['p_hat']:.3f} means that approximately 
{prop_result['p_hat']*100:.1f}% of the areas with complete data show both positive 
income change and increased tree cover.

The 95% confidence interval [{prop_result['ci_lower']:.3f}, {prop_result['ci_upper']:.3f}] 
indicates the range where we expect the true proportion to lie with 95% confidence.

Note: This analysis uses listwise deletion (complete case analysis), which assumes 
data is Missing Completely At Random (MCAR) or Missing At Random (MAR).
    """)
    
    print("\n" + "=" * 70)
    print("CALCULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
