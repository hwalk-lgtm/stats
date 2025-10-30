"""
Tests for the effective sample size calculator
"""

import unittest
import numpy as np
from effective_sample_size import (
    calculate_effective_sample_size,
    calculate_effective_sample_size_multivariate,
    calculate_proportion_estimate_ci
)


class TestCalculateEffectiveSampleSize(unittest.TestCase):
    """Tests for calculate_effective_sample_size function"""
    
    def test_no_missing_data(self):
        """Test with complete data (no missing values)"""
        data = [1, 2, 3, 4, 5]
        result = calculate_effective_sample_size(data)
        
        self.assertEqual(result['n_effective'], 5)
        self.assertEqual(result['n_total'], 5)
        self.assertEqual(result['n_missing'], 0)
        self.assertEqual(result['proportion_complete'], 1.0)
        self.assertEqual(result['proportion_missing'], 0.0)
    
    def test_with_nan_missing(self):
        """Test with NaN values as missing data"""
        data = [1, 2, np.nan, 4, 5, np.nan, 7]
        result = calculate_effective_sample_size(data)
        
        self.assertEqual(result['n_effective'], 5)
        self.assertEqual(result['n_total'], 7)
        self.assertEqual(result['n_missing'], 2)
        self.assertAlmostEqual(result['proportion_complete'], 5/7)
        self.assertAlmostEqual(result['proportion_missing'], 2/7)
    
    def test_with_none_missing(self):
        """Test with None values as missing data"""
        data = [1, 2, None, 4, 5, None, 7]
        result = calculate_effective_sample_size(data)
        
        self.assertEqual(result['n_effective'], 5)
        self.assertEqual(result['n_total'], 7)
        self.assertEqual(result['n_missing'], 2)
    
    def test_with_custom_missing_indicator(self):
        """Test with custom missing value indicator"""
        data = [1, 2, -999, 4, 5, -999, 7]
        result = calculate_effective_sample_size(data, missing_indicator=-999)
        
        self.assertEqual(result['n_effective'], 5)
        self.assertEqual(result['n_total'], 7)
        self.assertEqual(result['n_missing'], 2)
    
    def test_all_missing(self):
        """Test when all data is missing"""
        data = [np.nan, np.nan, np.nan]
        result = calculate_effective_sample_size(data)
        
        self.assertEqual(result['n_effective'], 0)
        self.assertEqual(result['n_total'], 3)
        self.assertEqual(result['n_missing'], 3)
        self.assertEqual(result['proportion_complete'], 0.0)
        self.assertEqual(result['proportion_missing'], 1.0)
    
    def test_empty_data(self):
        """Test with empty data array"""
        data = []
        result = calculate_effective_sample_size(data)
        
        self.assertEqual(result['n_effective'], 0)
        self.assertEqual(result['n_total'], 0)
        self.assertEqual(result['n_missing'], 0)
    
    def test_mixed_missing_types(self):
        """Test with both NaN and None as missing values"""
        data = [1, 2, None, 4, np.nan, 6, 7]
        result = calculate_effective_sample_size(data)
        
        self.assertEqual(result['n_effective'], 5)
        self.assertEqual(result['n_total'], 7)
        self.assertEqual(result['n_missing'], 2)


class TestCalculateEffectiveSampleSizeMultivariate(unittest.TestCase):
    """Tests for calculate_effective_sample_size_multivariate function"""
    
    def test_two_variables_no_missing(self):
        """Test with two complete variables"""
        income = [100, 200, 300, 400, 500]
        tree_cover = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = calculate_effective_sample_size_multivariate(income, tree_cover)
        
        self.assertEqual(result['n_effective'], 5)
        self.assertEqual(result['n_total'], 5)
        self.assertEqual(result['n_missing'], 0)
        self.assertEqual(result['missing_by_variable'], [0, 0])
    
    def test_two_variables_with_missing(self):
        """Test with missing data in both variables"""
        income = [100, 200, np.nan, 400, 500]
        tree_cover = [0.5, 0.6, 0.7, np.nan, 0.9]
        result = calculate_effective_sample_size_multivariate(income, tree_cover)
        
        self.assertEqual(result['n_effective'], 3)
        self.assertEqual(result['n_total'], 5)
        self.assertEqual(result['n_missing'], 2)
        self.assertEqual(result['missing_by_variable'], [1, 1])
    
    def test_three_variables(self):
        """Test with three variables"""
        var1 = [1, 2, np.nan, 4, 5]
        var2 = [10, np.nan, 30, 40, 50]
        var3 = [100, 200, 300, 400, np.nan]
        result = calculate_effective_sample_size_multivariate(var1, var2, var3)
        
        # Only index 0 and 3 are complete across all variables
        self.assertEqual(result['n_effective'], 2)
        self.assertEqual(result['n_total'], 5)
        self.assertEqual(result['n_missing'], 3)
        self.assertEqual(result['missing_by_variable'], [1, 1, 1])
    
    def test_same_missing_pattern(self):
        """Test when missing values are in the same positions"""
        var1 = [1, np.nan, 3, 4, np.nan]
        var2 = [10, np.nan, 30, 40, np.nan]
        result = calculate_effective_sample_size_multivariate(var1, var2)
        
        self.assertEqual(result['n_effective'], 3)
        self.assertEqual(result['n_missing'], 2)
    
    def test_custom_missing_indicator_multivariate(self):
        """Test multivariate with custom missing indicator"""
        var1 = [1, -999, 3, 4, -999]
        var2 = [10, 20, -999, 40, 50]
        result = calculate_effective_sample_size_multivariate(
            var1, var2, missing_indicator=-999
        )
        
        self.assertEqual(result['n_effective'], 2)
        self.assertEqual(result['missing_by_variable'], [2, 1])
    
    def test_unequal_length_arrays(self):
        """Test that unequal length arrays raise an error"""
        var1 = [1, 2, 3]
        var2 = [10, 20, 30, 40]
        
        with self.assertRaises(ValueError):
            calculate_effective_sample_size_multivariate(var1, var2)
    
    def test_no_arrays_provided(self):
        """Test that providing no arrays raises an error"""
        with self.assertRaises(ValueError):
            calculate_effective_sample_size_multivariate()
    
    def test_empty_arrays(self):
        """Test with empty arrays"""
        result = calculate_effective_sample_size_multivariate([], [])
        
        self.assertEqual(result['n_effective'], 0)
        self.assertEqual(result['n_total'], 0)


class TestCalculateProportionEstimateCI(unittest.TestCase):
    """Tests for calculate_proportion_estimate_ci function"""
    
    def test_basic_proportion_estimate(self):
        """Test basic proportion estimate calculation"""
        result = calculate_proportion_estimate_ci(successes=30, n_effective=50)
        
        self.assertAlmostEqual(result['p_hat'], 0.6)
        self.assertGreater(result['standard_error'], 0)
        self.assertLess(result['ci_lower'], result['p_hat'])
        self.assertGreater(result['ci_upper'], result['p_hat'])
        self.assertEqual(result['confidence_level'], 0.95)
    
    def test_proportion_zero(self):
        """Test when proportion is 0"""
        result = calculate_proportion_estimate_ci(successes=0, n_effective=50)
        
        self.assertEqual(result['p_hat'], 0.0)
        self.assertEqual(result['ci_lower'], 0.0)
        # When p=0, standard error is 0, so CI bounds collapse to point estimate
        self.assertEqual(result['ci_upper'], 0.0)
    
    def test_proportion_one(self):
        """Test when proportion is 1"""
        result = calculate_proportion_estimate_ci(successes=50, n_effective=50)
        
        self.assertEqual(result['p_hat'], 1.0)
        # When p=1, standard error is 0, so CI bounds collapse to point estimate
        self.assertEqual(result['ci_lower'], 1.0)
        self.assertEqual(result['ci_upper'], 1.0)
    
    def test_different_confidence_level(self):
        """Test with different confidence level"""
        result_95 = calculate_proportion_estimate_ci(
            successes=30, n_effective=50, confidence_level=0.95
        )
        result_99 = calculate_proportion_estimate_ci(
            successes=30, n_effective=50, confidence_level=0.99
        )
        
        # 99% CI should be wider than 95% CI
        width_95 = result_95['ci_upper'] - result_95['ci_lower']
        width_99 = result_99['ci_upper'] - result_99['ci_lower']
        self.assertGreater(width_99, width_95)
    
    def test_invalid_n_effective(self):
        """Test with invalid effective sample size"""
        with self.assertRaises(ValueError):
            calculate_proportion_estimate_ci(successes=5, n_effective=0)
        
        with self.assertRaises(ValueError):
            calculate_proportion_estimate_ci(successes=5, n_effective=-10)
    
    def test_invalid_successes(self):
        """Test with invalid number of successes"""
        with self.assertRaises(ValueError):
            calculate_proportion_estimate_ci(successes=-1, n_effective=50)
        
        with self.assertRaises(ValueError):
            calculate_proportion_estimate_ci(successes=60, n_effective=50)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic scenarios"""
    
    def test_income_tree_cover_scenario(self):
        """Test realistic scenario with income change and tree cover data"""
        # Simulate data: income change and tree cover measurements
        income_change = [100, 200, np.nan, 400, 500, 600, np.nan, 800]
        tree_cover = [0.5, np.nan, 0.7, 0.8, np.nan, 0.9, 0.95, 1.0]
        
        # Calculate effective sample size for multivariate analysis
        result = calculate_effective_sample_size_multivariate(
            income_change, tree_cover
        )
        
        # Indices 0, 3, 5, and 7 have complete data in both variables
        self.assertEqual(result['n_effective'], 4)
        self.assertEqual(result['n_total'], 8)
        
        # Assume 3 out of 4 complete cases show positive outcomes
        prop_result = calculate_proportion_estimate_ci(
            successes=3, n_effective=result['n_effective']
        )
        
        self.assertAlmostEqual(prop_result['p_hat'], 0.75)
        self.assertGreater(prop_result['standard_error'], 0)
    
    def test_single_variable_workflow(self):
        """Test complete workflow for single variable"""
        # Survey data with some missing responses
        survey_responses = [1, 1, 0, np.nan, 1, 0, 0, np.nan, 1, 1]
        
        # Calculate effective sample size
        result = calculate_effective_sample_size(survey_responses)
        
        self.assertEqual(result['n_effective'], 8)
        self.assertEqual(result['n_missing'], 2)
        
        # Count successes (1s) in non-missing data
        valid_data = [x for x in survey_responses if x is not None and not (isinstance(x, float) and np.isnan(x))]
        successes = sum(valid_data)
        
        # Calculate proportion estimate
        prop_result = calculate_proportion_estimate_ci(
            successes=successes,
            n_effective=result['n_effective']
        )
        
        self.assertAlmostEqual(prop_result['p_hat'], 5/8)


if __name__ == '__main__':
    unittest.main()
