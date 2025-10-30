# Quick Reference: Effective Sample Size

## The Question
**Given that you do not have repeated measurements of income change or tree cover over time and some missing data, what is your effective sample size for estimating p̂?**

## The Answer
**n_effective = number of complete cases**

Complete cases are observations where all required variables have non-missing values.

## Simple Example

```python
import numpy as np
from effective_sample_size import calculate_effective_sample_size

# Your data with some missing values (np.nan)
data = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9]

result = calculate_effective_sample_size(data)
print(f"Effective sample size: {result['n_effective']}")
# Output: Effective sample size: 7
```

## Multivariate Example (Income & Tree Cover)

```python
from effective_sample_size import calculate_effective_sample_size_multivariate

income = [100, 200, np.nan, 400, 500]
tree_cover = [0.5, 0.6, 0.7, np.nan, 0.9]

result = calculate_effective_sample_size_multivariate(income, tree_cover)
print(f"Effective sample size: {result['n_effective']}")
# Output: Effective sample size: 3
# (only indices 0, 1, and 4 have complete data in both variables)
```

## Complete Workflow: Calculate p̂

```python
from effective_sample_size import (
    calculate_effective_sample_size,
    calculate_proportion_estimate_ci
)

# 1. Your data with missing values
outcomes = [1, 1, 0, np.nan, 1, 0, 0, np.nan, 1, 1]

# 2. Calculate effective sample size
result = calculate_effective_sample_size(outcomes)
n_eff = result['n_effective']  # Will be 8

# 3. Count successes (non-missing 1s)
complete_outcomes = [x for x in outcomes if not (isinstance(x, float) and np.isnan(x))]
successes = sum(complete_outcomes)  # Will be 5

# 4. Calculate p̂ with confidence interval
prop = calculate_proportion_estimate_ci(successes=successes, n_effective=n_eff)
print(f"p̂ = {prop['p_hat']:.3f}")
print(f"95% CI: [{prop['ci_lower']:.3f}, {prop['ci_upper']:.3f}]")
# Output:
# p̂ = 0.625
# 95% CI: [0.290, 0.960]
```

## Key Points

1. **No repeated measurements** → Each observation is independent
2. **Missing data** → Only count complete cases
3. **Single variable**: n_eff = observations with non-missing value
4. **Multiple variables**: n_eff = observations with non-missing values in ALL variables
5. **Standard error depends on n_eff**: SE(p̂) = √[p̂(1-p̂)/n_eff]

## Files to Use

- `effective_sample_size.py` - Main functions
- `example.py` - Complete working example
- `DOCUMENTATION.md` - Full documentation
- `test_effective_sample_size.py` - Test suite

## Run Example

```bash
python example.py
```

## Run Tests

```bash
python -m unittest test_effective_sample_size -v
```
