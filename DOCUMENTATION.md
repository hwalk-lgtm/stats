# Effective Sample Size Calculator

This module provides functions to calculate the effective sample size when dealing with missing data and no repeated measurements, particularly for estimating proportions (p-hat).

## Problem Statement

Given that you do not have repeated measurements of income change or tree cover over time and some missing data, what is your effective sample size for estimating p̂ (p-hat)?

## Answer

The **effective sample size (n_effective)** is the number of complete cases (observations with no missing data).

When you have:
- No repeated measurements over time
- Missing data in one or more variables

The effective sample size for estimating p̂ is simply the count of observations where all required variables have non-missing values.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single Variable Analysis

```python
import numpy as np
from effective_sample_size import calculate_effective_sample_size

# Example: Survey data with missing responses
data = [1, 2, np.nan, 4, 5, np.nan, 7]
result = calculate_effective_sample_size(data)

print(f"Effective sample size: {result['n_effective']}")
print(f"Total observations: {result['n_total']}")
print(f"Missing observations: {result['n_missing']}")
print(f"Proportion complete: {result['proportion_complete']:.2%}")
```

Output:
```
Effective sample size: 5
Total observations: 7
Missing observations: 2
Proportion complete: 71.43%
```

### Multivariate Analysis (Income Change and Tree Cover)

```python
from effective_sample_size import calculate_effective_sample_size_multivariate

# Example: Income change and tree cover measurements
income_change = [100, 200, np.nan, 400, 500, 600, np.nan, 800]
tree_cover = [0.5, np.nan, 0.7, 0.8, np.nan, 0.9, 0.95, 1.0]

result = calculate_effective_sample_size_multivariate(income_change, tree_cover)

print(f"Effective sample size: {result['n_effective']}")
print(f"Total observations: {result['n_total']}")
print(f"Complete cases: {result['proportion_complete']:.2%}")
```

Output:
```
Effective sample size: 4
Total observations: 8
Complete cases: 50.00%
```

### Calculating Proportion Estimate (p̂) with Confidence Interval

```python
from effective_sample_size import (
    calculate_effective_sample_size,
    calculate_proportion_estimate_ci
)

# Example: Binary outcome data (e.g., income increased = 1, decreased = 0)
outcomes = [1, 1, 0, np.nan, 1, 0, 0, np.nan, 1, 1]

# Step 1: Calculate effective sample size
size_result = calculate_effective_sample_size(outcomes)
n_eff = size_result['n_effective']

# Step 2: Count successes in complete cases
complete_outcomes = [x for x in outcomes if not (isinstance(x, float) and np.isnan(x))]
successes = sum(complete_outcomes)

# Step 3: Calculate p̂ and confidence interval
prop_result = calculate_proportion_estimate_ci(
    successes=successes,
    n_effective=n_eff,
    confidence_level=0.95
)

print(f"p̂ (proportion estimate): {prop_result['p_hat']:.3f}")
print(f"Standard error: {prop_result['standard_error']:.3f}")
print(f"95% CI: [{prop_result['ci_lower']:.3f}, {prop_result['ci_upper']:.3f}]")
```

Output:
```
p̂ (proportion estimate): 0.625
Standard error: 0.171
95% CI: [0.290, 0.960]
```

## Custom Missing Value Indicators

If your data uses a specific value to indicate missing data (e.g., -999):

```python
data = [1, 2, -999, 4, 5, -999, 7]
result = calculate_effective_sample_size(data, missing_indicator=-999)
print(f"Effective sample size: {result['n_effective']}")
```

## Key Concepts

### What is Effective Sample Size?

The effective sample size is the number of observations that can actually be used in your analysis. When you have missing data:

- **Total sample size (n_total)**: Total number of observations collected
- **Effective sample size (n_effective)**: Number of complete cases (no missing values)
- **Missing observations (n_missing)**: Number of cases with at least one missing value

### Why Does This Matter?

1. **Statistical Power**: Your effective sample size determines the precision of your estimates
2. **Confidence Intervals**: Smaller n_effective leads to wider confidence intervals
3. **Standard Error**: SE = √[p̂(1-p̂)/n_effective], so missing data increases uncertainty

### Example Scenario

Suppose you're studying the relationship between income change and tree cover:
- You collected data from 100 households
- 15 households didn't report income change
- 20 households have missing tree cover measurements
- 10 households are missing both

Your effective sample size for multivariate analysis is:
- n_total = 100
- n_effective = 100 - 15 - 20 - 10 = 55

(Note: If some households are missing both variables, they're only counted once in the missing total)

## Functions

### `calculate_effective_sample_size(data, missing_indicator=None)`

Calculate effective sample size for a single variable.

**Parameters:**
- `data`: Array-like of observations
- `missing_indicator`: Optional custom missing value indicator

**Returns:** Dictionary with:
- `n_effective`: Effective sample size
- `n_total`: Total observations
- `n_missing`: Missing observations
- `proportion_complete`: Proportion of complete cases
- `proportion_missing`: Proportion of missing cases

### `calculate_effective_sample_size_multivariate(*data_arrays, missing_indicator=None)`

Calculate effective sample size for multiple variables (complete cases across all variables).

**Parameters:**
- `*data_arrays`: Multiple data arrays (one per variable)
- `missing_indicator`: Optional custom missing value indicator

**Returns:** Dictionary including:
- `n_effective`: Complete cases across all variables
- `missing_by_variable`: List of missing counts per variable

### `calculate_proportion_estimate_ci(successes, n_effective, confidence_level=0.95)`

Calculate proportion estimate (p̂) and confidence interval using effective sample size.

**Parameters:**
- `successes`: Number of successes
- `n_effective`: Effective sample size
- `confidence_level`: Confidence level (default: 0.95)

**Returns:** Dictionary with:
- `p_hat`: Proportion estimate
- `standard_error`: Standard error
- `ci_lower`: Lower confidence bound
- `ci_upper`: Upper confidence bound

## Running Tests

```bash
python -m unittest test_effective_sample_size -v
```

## Statistical Notes

1. **Complete Case Analysis**: This approach uses listwise deletion (only complete cases are analyzed)
2. **Assumptions**: Assumes data is Missing Completely At Random (MCAR) or Missing At Random (MAR)
3. **Alternatives**: For more sophisticated handling of missing data, consider:
   - Multiple imputation
   - Maximum likelihood estimation
   - Inverse probability weighting

## References

- Standard formula for proportion confidence interval: Wald method
- Standard error: SE(p̂) = √[p̂(1-p̂)/n]
- For small samples or extreme proportions, consider Wilson score interval or exact binomial methods
