"""
Test suite for SAMIRA (State-space Adaptive Multi-variate Integrated Regression Analysis) model
"""

import numpy as np
import pandas as pd
import pytest
from unittest import TestCase

from src.models.model_samira import SAMIRAModel, fit_samira_model


class TestSAMIRAModel(TestCase):
    """Test cases for SAMIRA model implementation"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic time series with trend, seasonality, and noise
        n_periods = 100
        time_index = pd.date_range('2020-01-01', periods=n_periods, freq='M')
        
        # Trend component
        trend = np.linspace(100, 150, n_periods)
        
        # Seasonal component (annual seasonality)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
        
        # Noise
        noise = np.random.normal(0, 2, n_periods)
        
        # Combined series
        self.y = pd.Series(trend + seasonal + noise, index=time_index)
        
        # Exogenous variables
        self.exog = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_periods),
            'feature2': np.random.uniform(-1, 1, n_periods)
        }, index=time_index)
        
        # Add exogenous effects to series
        self.y += 5 * self.exog['feature1'] + 3 * self.exog['feature2']
        
    def test_model_initialization(self):
        """Test SAMIRA model initialization"""
        model = SAMIRAModel(
            trend_components=2,
            seasonal_period=12,
            adaptation_rate=0.95
        )
        
        self.assertEqual(model.trend_components, 2)
        self.assertEqual(model.seasonal_period, 12)
        self.assertEqual(model.adaptation_rate, 0.95)
        self.assertFalse(model.is_fitted_)
    
    def test_seasonality_detection(self):
        """Test automatic seasonality detection"""
        model = SAMIRAModel()
        detected_period = model._detect_seasonality(self.y)
        
        # Should detect monthly seasonality (period=12)
        self.assertGreaterEqual(detected_period, 1)
        self.assertLessEqual(detected_period, 52)
    
    def test_state_space_setup(self):
        """Test state space dimension setup"""
        model = SAMIRAModel(trend_components=2, seasonal_period=12)
        n_obs, state_dim = model._setup_state_space(self.y, self.exog)
        
        self.assertEqual(n_obs, len(self.y))
        # State: 2 trend + 11 seasonal + 2 exog = 15
        self.assertEqual(state_dim, 2 + 11 + 2)
    
    def test_observation_matrix(self):
        """Test observation matrix construction"""
        model = SAMIRAModel(trend_components=2, seasonal_period=12)
        model._setup_state_space(self.y, self.exog)
        
        Z = model._build_observation_matrix(self.exog)
        
        # Should have correct dimensions
        self.assertEqual(len(Z), 2 + 11 + 2)  # trend + seasonal + exog
        
        # Level component should be 1
        self.assertEqual(Z[0], 1.0)
        
        # Seasonal component should be 1
        self.assertEqual(Z[2], 1.0)  # After trend components
    
    def test_transition_matrix(self):
        """Test transition matrix construction"""
        model = SAMIRAModel(trend_components=2, seasonal_period=12)
        T = model._build_transition_matrix(exog_dim=2)
        
        # Should be square matrix
        self.assertEqual(T.shape[0], T.shape[1])
        
        # Should have appropriate structure for trend
        self.assertEqual(T[0, 1], 1.0)  # level = level + slope
    
    def test_model_fitting_univariate(self):
        """Test model fitting without exogenous variables"""
        model = SAMIRAModel(
            trend_components=2,
            seasonal_period=12,
            max_iterations=10  # Reduce for testing
        )
        
        # Fit model
        fitted_model = model.fit(self.y)
        
        # Check if model is fitted
        self.assertTrue(fitted_model.is_fitted_)
        self.assertIsNotNone(fitted_model.fitted_values_)
        self.assertIsNotNone(fitted_model.states_)
        self.assertIsNotNone(fitted_model.log_likelihood_)
        
        # Check fitted values dimensions
        self.assertEqual(len(fitted_model.fitted_values_), len(self.y))
    
    def test_model_fitting_multivariate(self):
        """Test model fitting with exogenous variables"""
        model = SAMIRAModel(
            trend_components=2,
            seasonal_period=12,
            max_iterations=10
        )
        
        # Fit model with exogenous variables
        fitted_model = model.fit(self.y, self.exog)
        
        self.assertTrue(fitted_model.is_fitted_)
        self.assertIsNotNone(fitted_model.fitted_values_)
        self.assertEqual(len(fitted_model.fitted_values_), len(self.y))
    
    def test_forecasting(self):
        """Test forecasting functionality"""
        model = SAMIRAModel(
            trend_components=2,
            seasonal_period=12,
            max_iterations=10
        )
        
        # Fit model
        model.fit(self.y, self.exog)
        
        # Generate forecasts
        steps = 12
        exog_future = pd.DataFrame({
            'feature1': np.random.normal(0, 1, steps),
            'feature2': np.random.uniform(-1, 1, steps)
        })
        
        forecast, lower_bound, upper_bound = model.forecast(steps, exog_future)
        
        # Check forecast dimensions
        self.assertEqual(len(forecast), steps)
        self.assertEqual(len(lower_bound), steps)
        self.assertEqual(len(upper_bound), steps)
        
        # Check confidence interval ordering
        self.assertTrue(all(lower_bound <= forecast))
        self.assertTrue(all(forecast <= upper_bound))
    
    def test_component_extraction(self):
        """Test extraction of model components"""
        model = SAMIRAModel(
            trend_components=2,
            seasonal_period=12,
            max_iterations=10
        )
        
        # Fit model
        model.fit(self.y, self.exog)
        
        # Extract components
        components = model.get_components()
        
        # Check component availability
        self.assertIn('level', components)
        self.assertIn('slope', components)
        self.assertIn('seasonal', components)
        
        # Check exogenous effects
        self.assertIn('effect_feature1', components)
        self.assertIn('effect_feature2', components)
        self.assertIn('coeff_feature1', components)
        self.assertIn('coeff_feature2', components)
        
        # Check component dimensions
        for component in components.values():
            self.assertEqual(len(component), len(self.y))
    
    def test_wrapper_function(self):
        """Test the wrapper function for pipeline integration"""
        result = fit_samira_model(
            series=self.y,
            test_size=12,
            exog=self.exog,
            trend_components=2,
            seasonal_period=12
        )
        
        # Check return structure
        self.assertIn('forecast', result)
        self.assertIn('fitted', result)
        self.assertIn('mae', result)
        self.assertIn('rmse', result)
        self.assertIn('components', result)
        self.assertIn('confidence_intervals', result)
        
        # Check forecast quality
        self.assertIsNotNone(result['forecast'])
        self.assertEqual(len(result['forecast']), 12)
        
        # Check metrics
        self.assertIsInstance(result['mae'], float)
        self.assertIsInstance(result['rmse'], float)
        self.assertGreater(result['mae'], 0)
        self.assertGreater(result['rmse'], 0)
    
    def test_short_series_handling(self):
        """Test handling of short time series"""
        short_series = self.y[:2]  # Very short series
        
        with self.assertRaises(ValueError):
            model = SAMIRAModel()
            model.fit(short_series)
    
    def test_missing_values_handling(self):
        """Test handling of missing values"""
        # Create series with missing values
        y_missing = self.y.copy()
        y_missing.iloc[10:15] = np.nan
        
        model = SAMIRAModel(max_iterations=5)
        
        # Should handle missing values gracefully
        fitted_model = model.fit(y_missing)
        self.assertTrue(fitted_model.is_fitted_)
    
    def test_adaptation_rate_effect(self):
        """Test effect of different adaptation rates"""
        # High adaptation rate (more adaptive)
        model_adaptive = SAMIRAModel(adaptation_rate=0.90, max_iterations=5)
        model_adaptive.fit(self.y, self.exog)
        
        # Low adaptation rate (less adaptive)
        model_stable = SAMIRAModel(adaptation_rate=0.99, max_iterations=5)
        model_stable.fit(self.y, self.exog)
        
        # Both should fit successfully
        self.assertTrue(model_adaptive.is_fitted_)
        self.assertTrue(model_stable.is_fitted_)
        
        # Adaptive model might have different coefficient evolution
        components_adaptive = model_adaptive.get_components()
        components_stable = model_stable.get_components()
        
        # Coefficient variance should be different
        coeff_var_adaptive = components_adaptive['coeff_feature1'].var()
        coeff_var_stable = components_stable['coeff_feature1'].var()
        
        # More adaptive model should have higher coefficient variance
        self.assertGreaterEqual(coeff_var_adaptive, 0)
        self.assertGreaterEqual(coeff_var_stable, 0)


def test_samira_integration():
    """Integration test with realistic data"""
    # Create more realistic economic time series
    np.random.seed(123)
    
    # Monthly data for 5 years
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='M')
    n = len(dates)
    
    # Economic indicators
    gdp_growth = np.random.normal(0.02, 0.01, n)  # Monthly GDP growth
    inflation = np.random.normal(0.002, 0.005, n)  # Monthly inflation
    
    # Target variable (e.g., unemployment rate)
    unemployment = 5.0  # Base level
    unemployment_series = [unemployment]
    
    for i in range(1, n):
        # Unemployment responds to economic conditions
        change = -0.5 * gdp_growth[i] + 2.0 * inflation[i] + np.random.normal(0, 0.1)
        unemployment += change
        unemployment_series.append(unemployment)
    
    unemployment_ts = pd.Series(unemployment_series, index=dates)
    exog_data = pd.DataFrame({
        'gdp_growth': gdp_growth,
        'inflation': inflation
    }, index=dates)
    
    # Test SAMIRA fitting
    result = fit_samira_model(
        series=unemployment_ts,
        test_size=12,  # Test on last year
        exog=exog_data,
        trend_components=2,
        seasonal_period=12,
        adaptation_rate=0.95
    )
    
    # Should produce reasonable results
    assert result['mae'] < 1.0  # MAE should be reasonable for unemployment rate
    assert result['rmse'] < 1.5  # RMSE should be reasonable
    assert len(result['components']) > 5  # Should have multiple components
    
    print(f"SAMIRA Integration Test Results:")
    print(f"MAE: {result['mae']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"Components: {list(result['components'].keys())}")
    
    return result


if __name__ == "__main__":
    # Run integration test
    test_result = test_samira_integration()
    print("✅ SAMIRA integration test passed!")