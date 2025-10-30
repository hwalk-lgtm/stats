# Effective Sample Size Calculator

## Problem Statement

**Given that you do not have repeated measurements of income change or tree cover over time and some missing data, what is your effective sample size for estimating p̂?**

## Answer

The **effective sample size (n_effective)** for estimating p̂ is the **number of complete cases** - that is, the count of observations where all required variables have non-missing values.

When you have:
- No repeated measurements over time
- Missing data in your variables

Your effective sample size is simply: **n_effective = total observations - missing observations**

For multivariate analysis (e.g., income change AND tree cover), n_effective is the count of cases with complete data across **all** variables.

## Quick Start

```python
import numpy as np
from effective_sample_size import calculate_effective_sample_size

# Example with missing data
data = [1, 2, np.nan, 4, 5, np.nan, 7]
result = calculate_effective_sample_size(data)

print(f"Effective sample size: {result['n_effective']}")  # Output: 5
```

## Features

- ✅ Calculate effective sample size for single variables
- ✅ Calculate effective sample size for multivariate analysis
- ✅ Calculate proportion estimates (p̂) with confidence intervals
- ✅ Handle NaN, None, and custom missing value indicators
- ✅ Comprehensive test suite
- ✅ Detailed documentation and examples

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed usage instructions and examples.

Run the example script:
```bash
python example.py
```

Run tests:
```bash
python -m unittest test_effective_sample_size -v
```

## Files

- `effective_sample_size.py` - Main module with calculation functions
- `test_effective_sample_size.py` - Comprehensive test suite
- `example.py` - Example demonstrating income change and tree cover analysis
- `DOCUMENTATION.md` - Detailed documentation
- `requirements.txt` - Python dependencies

## Key Concepts

**Effective Sample Size**: The number of observations that can actually be used in analysis after accounting for missing data.

**Example**: If you collected data from 100 subjects but 15 have missing income data and 20 have missing tree cover data (with 5 missing both), your effective sample size for multivariate analysis is: 100 - 15 - 20 + 5 = 70 complete cases.

## Statistical Foundation

This implementation follows standard statistical practice for complete case analysis (listwise deletion). The effective sample size determines:
- Statistical power
- Precision of estimates
- Width of confidence intervals

Standard error for proportions: SE(p̂) = √[p̂(1-p̂)/n_effective]