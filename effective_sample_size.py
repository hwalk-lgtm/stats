"""
Effective Sample Size Calculator

This module provides functions to calculate the effective sample size
when dealing with missing data and no repeated measurements.
"""

import numpy as np
from typing import Union, Optional


def calculate_effective_sample_size(
    data: Union[np.ndarray, list],
    missing_indicator: Optional[Union[float, str]] = None
) -> dict:
    """
    Calculate the effective sample size for estimating p-hat (proportion estimate).
    
    When there are no repeated measurements of income change or tree cover over time
    and some missing data, the effective sample size is the number of complete cases
    (non-missing observations).
    
    Parameters
    ----------
    data : array-like
        Input data array or list containing the observations.
        Can contain missing values (NaN, None, or custom missing indicator).
    missing_indicator : float, str, or None, optional
        Custom missing value indicator. If None, will treat NaN and None as missing.
        Default is None.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'n_effective': Effective sample size (number of complete cases)
        - 'n_total': Total number of observations (including missing)
        - 'n_missing': Number of missing observations
        - 'proportion_complete': Proportion of complete cases
        - 'proportion_missing': Proportion of missing data
    
    Examples
    --------
    >>> data = [1, 2, np.nan, 4, 5, np.nan, 7]
    >>> result = calculate_effective_sample_size(data)
    >>> result['n_effective']
    5
    
    >>> data_with_custom_missing = [1, 2, -999, 4, 5, -999, 7]
    >>> result = calculate_effective_sample_size(data_with_custom_missing, missing_indicator=-999)
    >>> result['n_effective']
    5
    """
    # Convert to numpy array for easier handling
    data_array = np.array(data, dtype=object)
    n_total = len(data_array)
    
    if n_total == 0:
        return {
            'n_effective': 0,
            'n_total': 0,
            'n_missing': 0,
            'proportion_complete': 0.0,
            'proportion_missing': 0.0
        }
    
    # Identify missing values
    if missing_indicator is not None:
        # Use custom missing indicator
        missing_mask = data_array == missing_indicator
    else:
        # Default: treat NaN and None as missing
        missing_mask = np.array([
            (x is None) or (isinstance(x, float) and np.isnan(x))
            for x in data_array
        ])
    
    n_missing = np.sum(missing_mask)
    n_effective = n_total - n_missing
    
    proportion_complete = n_effective / n_total if n_total > 0 else 0.0
    proportion_missing = n_missing / n_total if n_total > 0 else 0.0
    
    return {
        'n_effective': int(n_effective),
        'n_total': int(n_total),
        'n_missing': int(n_missing),
        'proportion_complete': float(proportion_complete),
        'proportion_missing': float(proportion_missing)
    }


def calculate_effective_sample_size_multivariate(
    *data_arrays: Union[np.ndarray, list],
    missing_indicator: Optional[Union[float, str]] = None
) -> dict:
    """
    Calculate the effective sample size when multiple variables are involved.
    
    For multivariate analysis (e.g., income change AND tree cover), the effective
    sample size is the number of cases with complete data across all variables.
    
    Parameters
    ----------
    *data_arrays : array-like
        Multiple data arrays, one for each variable.
        Each array should have the same length.
    missing_indicator : float, str, or None, optional
        Custom missing value indicator. If None, will treat NaN and None as missing.
        Default is None.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'n_effective': Effective sample size (complete cases across all variables)
        - 'n_total': Total number of observations
        - 'n_missing': Number of cases with at least one missing value
        - 'proportion_complete': Proportion of complete cases
        - 'proportion_missing': Proportion with missing data
        - 'missing_by_variable': List of missing counts per variable
    
    Examples
    --------
    >>> income = [100, 200, np.nan, 400, 500]
    >>> tree_cover = [0.5, 0.6, 0.7, np.nan, 0.9]
    >>> result = calculate_effective_sample_size_multivariate(income, tree_cover)
    >>> result['n_effective']
    3
    """
    if len(data_arrays) == 0:
        raise ValueError("At least one data array must be provided")
    
    # Convert all arrays to numpy arrays
    arrays = [np.array(arr, dtype=object) for arr in data_arrays]
    
    # Check that all arrays have the same length
    n_total = len(arrays[0])
    if not all(len(arr) == n_total for arr in arrays):
        raise ValueError("All data arrays must have the same length")
    
    if n_total == 0:
        return {
            'n_effective': 0,
            'n_total': 0,
            'n_missing': 0,
            'proportion_complete': 0.0,
            'proportion_missing': 0.0,
            'missing_by_variable': [0] * len(arrays)
        }
    
    # Create a combined missing mask (True if ANY variable is missing)
    combined_missing_mask = np.zeros(n_total, dtype=bool)
    missing_by_variable = []
    
    for arr in arrays:
        if missing_indicator is not None:
            missing_mask = arr == missing_indicator
        else:
            missing_mask = np.array([
                (x is None) or (isinstance(x, float) and np.isnan(x))
                for x in arr
            ])
        
        missing_by_variable.append(int(np.sum(missing_mask)))
        combined_missing_mask = combined_missing_mask | missing_mask
    
    n_missing = np.sum(combined_missing_mask)
    n_effective = n_total - n_missing
    
    proportion_complete = n_effective / n_total if n_total > 0 else 0.0
    proportion_missing = n_missing / n_total if n_total > 0 else 0.0
    
    return {
        'n_effective': int(n_effective),
        'n_total': int(n_total),
        'n_missing': int(n_missing),
        'proportion_complete': float(proportion_complete),
        'proportion_missing': float(proportion_missing),
        'missing_by_variable': missing_by_variable
    }


def calculate_proportion_estimate_ci(
    successes: int,
    n_effective: int,
    confidence_level: float = 0.95
) -> dict:
    """
    Calculate the proportion estimate (p-hat) and its confidence interval
    using the effective sample size.
    
    Parameters
    ----------
    successes : int
        Number of successes (e.g., positive income changes, tree cover increases)
    n_effective : int
        Effective sample size (number of complete cases)
    confidence_level : float, optional
        Confidence level for the interval (default: 0.95 for 95% CI)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'p_hat': Estimated proportion
        - 'standard_error': Standard error of the proportion
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        - 'confidence_level': Confidence level used
    
    Examples
    --------
    >>> result = calculate_proportion_estimate_ci(successes=30, n_effective=50)
    >>> result['p_hat']
    0.6
    """
    if n_effective <= 0:
        raise ValueError("Effective sample size must be positive")
    
    if successes < 0 or successes > n_effective:
        raise ValueError("Number of successes must be between 0 and n_effective")
    
    # Calculate proportion estimate
    p_hat = successes / n_effective
    
    # Calculate standard error using the formula: sqrt(p_hat * (1 - p_hat) / n_effective)
    standard_error = np.sqrt(p_hat * (1 - p_hat) / n_effective)
    
    # Calculate z-score for the confidence level
    from scipy import stats
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    # Calculate confidence interval
    margin_of_error = z_score * standard_error
    ci_lower = max(0.0, p_hat - margin_of_error)
    ci_upper = min(1.0, p_hat + margin_of_error)
    
    return {
        'p_hat': float(p_hat),
        'standard_error': float(standard_error),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence_level': float(confidence_level)
    }
