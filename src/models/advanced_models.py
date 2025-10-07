"""Advanced time series models for CrystalBall.

This module implements several sophisticated time series models:
1. Prophet (Facebook) - Handles seasonality and holidays
2. LightGBM - Gradient boosting for time series
3. VAR (Vector Autoregression) - For multivariate analysis
4. TBATS - Complex seasonality handling
5. Neural Prophet - Deep learning extension of Prophet
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def fit_prophet_model(series: pd.Series,
                     test_size: int,
                     yearly_seasonality: Union[bool, int] = 'auto',
                     weekly_seasonality: Union[bool, int] = 'auto',
                     daily_seasonality: Union[bool, int] = 'auto',
                     holidays: Optional[pd.DataFrame] = None) -> Dict:
    """
    Fit Facebook Prophet model with automatic seasonality detection.
    
    Args:
        series: Input time series
        test_size: Number of periods to forecast
        yearly_seasonality: Yearly seasonality setting
        weekly_seasonality: Weekly seasonality setting
        daily_seasonality: Daily seasonality setting
        holidays: Optional holiday DataFrame
        
    Returns:
        Dict with forecast, fitted values and metrics
    """
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        train_df = df.iloc[:-test_size]
        
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        if holidays is not None:
            model.add_country_holidays(country_name='US')
            
        model.fit(train_df)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=test_size, 
                                           freq=pd.infer_freq(series.index))
        forecast = model.predict(future)
        
        # Extract results
        fitted_values = pd.Series(
            forecast['yhat'].iloc[:-test_size].values,
            index=series.index[:-test_size]
        )
        forecast_values = pd.Series(
            forecast['yhat'].iloc[-test_size:].values,
            index=series.index[-test_size:]
        )
        
        # Compute metrics
        actuals = series.iloc[-test_size:]
        mae = np.mean(np.abs(actuals - forecast_values))
        rmse = np.sqrt(np.mean((actuals - forecast_values) ** 2))
        
        return {
            'forecast': forecast_values,
            'fitted': fitted_values,
            'mae': mae,
            'rmse': rmse,
            'model': model
        }
    
    except Exception as e:
        logger.exception("Prophet model fitting failed")
        return {'error': str(e)}

def fit_lightgbm_model(series: pd.Series,
                      test_size: int,
                      num_lags: int = 12,
                      num_features: int = 5) -> Dict:
    """
    Fit LightGBM model for time series forecasting.
    
    Args:
        series: Input time series
        test_size: Number of periods to forecast
        num_lags: Number of lag features to create
        num_features: Number of additional engineered features
        
    Returns:
        Dict with forecast, fitted values and metrics
    """
    try:
        import lightgbm as lgb
        
        # Create features
        data = pd.DataFrame({'y': series})
        
        # Add lag features
        for i in range(1, num_lags + 1):
            data[f'lag_{i}'] = data['y'].shift(i)
            
        # Add rolling features
        for window in [3, 6, 12]:
            data[f'rolling_mean_{window}'] = data['y'].rolling(window).mean()
            data[f'rolling_std_{window}'] = data['y'].rolling(window).std()
            
        # Add time features if index is datetime
        if isinstance(series.index, pd.DatetimeIndex):
            data['month'] = series.index.month
            data['quarter'] = series.index.quarter
            if series.index.freq == 'D':
                data['dayofweek'] = series.index.dayofweek
                
        # Split data
        data = data.dropna()
        train_data = data.iloc[:-test_size]
        
        # Prepare training data
        X_train = train_data.drop('y', axis=1)
        y_train = train_data['y']
        
        # Train model
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31
        )
        model.fit(X_train, y_train)
        
        # Generate forecasts
        forecasts = []
        current_data = data.iloc[:-test_size].copy()
        
        for i in range(test_size):
            # Prepare features for next prediction
            next_features = current_data.iloc[-1:].drop('y', axis=1)
            
            # Make prediction
            pred = model.predict(next_features)[0]
            forecasts.append(pred)
            
            # Update data for next iteration
            new_row = pd.DataFrame({'y': pred}, index=[series.index[-test_size + i]])
            current_data = pd.concat([current_data, new_row])
            
            # Update features
            for lag in range(1, num_lags + 1):
                current_data.loc[new_row.index, f'lag_{lag}'] = current_data['y'].shift(lag).iloc[-1]
            
            for window in [3, 6, 12]:
                current_data.loc[new_row.index, f'rolling_mean_{window}'] = current_data['y'].rolling(window).mean().iloc[-1]
                current_data.loc[new_row.index, f'rolling_std_{window}'] = current_data['y'].rolling(window).std().iloc[-1]
                
        # Create forecast series
        forecast_values = pd.Series(forecasts, index=series.index[-test_size:])
        fitted_values = pd.Series(
            model.predict(X_train),
            index=series.index[num_lags:-test_size]
        )
        
        # Compute metrics
        actuals = series.iloc[-test_size:]
        mae = np.mean(np.abs(actuals - forecast_values))
        rmse = np.sqrt(np.mean((actuals - forecast_values) ** 2))
        
        return {
            'forecast': forecast_values,
            'fitted': fitted_values,
            'mae': mae,
            'rmse': rmse,
            'model': model
        }
    
    except Exception as e:
        logger.exception("LightGBM model fitting failed")
        return {'error': str(e)}

def fit_var_model(series: pd.Series,
                 exog: Optional[pd.DataFrame],
                 test_size: int,
                 maxlags: int = 15) -> Dict:
    """
    Fit Vector Autoregression (VAR) model.
    
    Args:
        series: Target time series
        exog: Exogenous variables DataFrame
        test_size: Number of periods to forecast
        maxlags: Maximum number of lags to consider
        
    Returns:
        Dict with forecast, fitted values and metrics
    """
    try:
        from statsmodels.tsa.api import VAR
        
        if exog is None:
            logger.warning("VAR model requires exogenous variables, using differenced lags instead")
            # Create features from the series itself
            data = pd.DataFrame({'y': series})
            data['diff1'] = series.diff()
            data['diff2'] = series.diff().diff()
            data = data.dropna()
        else:
            # Combine target and exogenous variables
            data = pd.concat([series, exog], axis=1).dropna()
            
        # Split data
        train_data = data.iloc[:-test_size]
        
        # Fit VAR model
        model = VAR(train_data)
        results = model.select_order(maxlags=maxlags)
        var_model = model.fit(results.aic)
        
        # Generate forecasts
        forecast = var_model.forecast(train_data.values, steps=test_size)
        forecast_values = pd.Series(
            forecast[:, 0],  # First column contains target variable forecast
            index=series.index[-test_size:]
        )
        
        fitted_values = pd.Series(
            var_model.fittedvalues[:, 0],
            index=series.index[maxlags:-test_size]
        )
        
        # Compute metrics
        actuals = series.iloc[-test_size:]
        mae = np.mean(np.abs(actuals - forecast_values))
        rmse = np.sqrt(np.mean((actuals - forecast_values) ** 2))
        
        return {
            'forecast': forecast_values,
            'fitted': fitted_values,
            'mae': mae,
            'rmse': rmse,
            'model': var_model
        }
    
    except Exception as e:
        logger.exception("VAR model fitting failed")
        return {'error': str(e)}

def fit_tbats_model(series: pd.Series,
                   test_size: int,
                   seasonal_periods: Optional[List[int]] = None) -> Dict:
    """
    Fit TBATS model for complex seasonality.
    
    Args:
        series: Input time series
        test_size: Number of periods to forecast
        seasonal_periods: List of seasonal periods to consider
        
    Returns:
        Dict with forecast, fitted values and metrics
    """
    try:
        from tbats import TBATS
        
        # Detect seasonal periods if not provided
        if seasonal_periods is None:
            from statsmodels.tsa.stattools import acf
            r = acf(series, nlags=min(len(series) - 1, 730))  # Max 2 years of daily data
            potential_periods = [i for i, x in enumerate(r) if x > 0.3]
            seasonal_periods = [p for p in potential_periods if p > 1][:3]  # Take top 3
            
        # Initialize and fit model
        estimator = TBATS(
            seasonal_periods=seasonal_periods,
            use_box_cox=True,
            use_trend=True,
            use_damped_trend=False
        )
        
        model = estimator.fit(series.iloc[:-test_size])
        
        # Generate forecasts
        forecast_values = pd.Series(
            model.forecast(steps=test_size),
            index=series.index[-test_size:]
        )
        
        fitted_values = pd.Series(
            model.y_hat,
            index=series.index[:-test_size]
        )
        
        # Compute metrics
        actuals = series.iloc[-test_size:]
        mae = np.mean(np.abs(actuals - forecast_values))
        rmse = np.sqrt(np.mean((actuals - forecast_values) ** 2))
        
        return {
            'forecast': forecast_values,
            'fitted': fitted_values,
            'mae': mae,
            'rmse': rmse,
            'model': model,
            'seasonal_periods': seasonal_periods
        }
    
    except Exception as e:
        logger.exception("TBATS model fitting failed")
        return {'error': str(e)}

def fit_neuralprophet_model(series: pd.Series,
                          test_size: int,
                          n_changepoints: int = 10,
                          yearly_seasonality: bool = True,
                          weekly_seasonality: bool = True,
                          daily_seasonality: bool = True) -> Dict:
    """
    Fit NeuralProphet model (deep learning extension of Prophet).
    
    Args:
        series: Input time series
        test_size: Number of periods to forecast
        n_changepoints: Number of potential trend changepoints
        yearly_seasonality: Whether to model yearly seasonality
        weekly_seasonality: Whether to model weekly seasonality
        daily_seasonality: Whether to model daily seasonality
        
    Returns:
        Dict with forecast, fitted values and metrics
    """
    try:
        from neuralprophet import NeuralProphet
        
        # Prepare data
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        train_df = df.iloc[:-test_size]
        
        # Initialize and fit model
        model = NeuralProphet(
            n_changepoints=n_changepoints,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            batch_size=64,
            epochs=100
        )
        
        metrics = model.fit(train_df, freq=pd.infer_freq(series.index))
        
        # Generate forecast
        future = model.make_future_dataframe(df=train_df, periods=test_size)
        forecast = model.predict(future)
        
        # Extract results
        forecast_values = pd.Series(
            forecast['yhat1'].iloc[-test_size:].values,
            index=series.index[-test_size:]
        )
        
        fitted_values = pd.Series(
            forecast['yhat1'].iloc[:-test_size].values,
            index=series.index[:-test_size]
        )
        
        # Compute metrics
        actuals = series.iloc[-test_size:]
        mae = np.mean(np.abs(actuals - forecast_values))
        rmse = np.sqrt(np.mean((actuals - forecast_values) ** 2))
        
        return {
            'forecast': forecast_values,
            'fitted': fitted_values,
            'mae': mae,
            'rmse': rmse,
            'model': model
        }
    
    except Exception as e:
        logger.exception("NeuralProphet model fitting failed")
        return {'error': str(e)}