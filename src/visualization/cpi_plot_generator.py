# ruff: noqa: E501
"""
CPI Forecasting Plot Generator

Interactive plot generator that allows users to choose different forecasting models
for CPI data visualization and comparison.
"""

import os
import warnings
from typing import Any, cast

import matplotlib.dates as mdates
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.evaluation import compute_model_cv_metrics
from src.core import utils
from src.core.logutil import get_logger
from src.models import advanced_models
from src.models.model_samira import fit_samira_model

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CPIPlotGenerator:
    """
    A comprehensive plot generator for CPI forecasting with multiple model options.
    """
    
    def __init__(
        self, 
        data_path: str | None = None, 
        output_dir: str = "data/processed/visualizations"
    ):
        """
        Initialize the CPI Plot Generator.
        
        Args:
            data_path: Path to CPI data file. If None, will try to auto-detect from raw data.
            output_dir: Directory to save generated plots.
        """
        # SECURE PATH HANDLING
        if data_path:
            # Validate and normalize the path
            data_path = os.path.normpath(data_path)
            # Allow paths within data/ directory or current working directory
            allowed_prefixes = ('data/', './data/', 'data\\', '.\\data\\')
            if not any(data_path.startswith(prefix) for prefix in allowed_prefixes):
                cwd = os.getcwd()
                if not data_path.startswith(cwd):
                    raise ValueError(
                        "Data path must be within allowed directories "
                        "(data/ or current working directory)"
                    )
        
        # Secure output directory
        output_dir = os.path.normpath(output_dir)
        if '..' in output_dir or output_dir.startswith(('/', '\\')):
            raise ValueError("Invalid output directory path")
            
        self.data_path = data_path
        self.output_dir = output_dir
        self.data: pd.DataFrame | None = None
        self.cpi_series: pd.Series | None = None
        self.available_models = self._get_available_models()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"CPI Plot Generator initialized. Output directory: {self.output_dir}")
    
    def _get_available_models(self) -> dict[str, str]:
        """Get available forecasting models with descriptions."""
        return {
            'arima': 'ARIMA - AutoRegressive Integrated Moving Average',
            'ets': 'ETS - Exponential Smoothing',
            'prophet': 'Prophet - Facebook\'s Time Series Forecasting',
            'samira': 'SAMIRA - Self-Adaptive Model for Interval Regression Analysis',
            'neural_prophet': 'Neural Prophet - Neural Network based Prophet',
            'croston': 'Croston - Intermittent Demand Forecasting',
            'ses': 'SES - Simple Exponential Smoothing',
            'holt': 'Holt - Double Exponential Smoothing',
            'naive': 'Naive - Last Value Persistence',
            'seasonal_naive': 'Seasonal Naive - Last Seasonal Value',
            'drift': 'Drift - Linear Trend Extrapolation'
        }
    
    def load_cpi_data(self, data_path: str | None = None) -> pd.Series:
        """
        Load CPI data from file or auto-detect from raw data directory.
        
        Args:
            data_path: Optional path to specific data file
            
        Returns:
            CPI time series data
        """
        if data_path:
            self.data_path = data_path
        
        # Try to auto-detect CPI data if no path provided
        if not self.data_path:
            self.data_path = self._auto_detect_cpi_file()
        
        if not self.data_path or not os.path.exists(self.data_path):
            return self._create_synthetic_cpi_data()
        
        try:
            return self._load_and_process_data()
        except Exception as e:
            logger.error(f"Error loading CPI data: {e}")
            logger.info("Creating synthetic CPI data for demonstration.")
            return self._create_synthetic_cpi_data()
    
    def _auto_detect_cpi_file(self) -> str | None:
        """Auto-detect CPI data file from raw data directory."""
        raw_data_dir = "data/raw"
        cpi_files = [
            "consumerpriceinflationdetailedreferencetables.xlsx",
            "series-041025.csv",
            "average_csv_2025-3.csv"
        ]
        
        for filename in cpi_files:
            potential_path = os.path.join(raw_data_dir, filename)
            if os.path.exists(potential_path):
                logger.info(f"Auto-detected CPI data file: {filename}")
                return potential_path
        
        logger.warning("No CPI data file found. Creating synthetic CPI data for demonstration.")
        return None
    
    def _load_and_process_data(self) -> pd.Series:
        """Load and process the CPI data file."""
        if not self.data_path:
            raise ValueError("No data path specified")
            
        # Load data based on file extension
        if self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path, sheet_name=0)
        elif self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        # Clean and prepare data
        self.data = utils.clean_df(self.data)
        
        # Extract CPI series
        self.cpi_series = self._extract_cpi_series()
        
        logger.info(
            f"Loaded CPI data: {len(self.cpi_series)} observations "
            f"from {self.cpi_series.index[0]} to {self.cpi_series.index[-1]}"
        )
        return self.cpi_series
    
    def _extract_cpi_series(self) -> pd.Series:
        """Extract CPI time series from the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Detect CPI column
        cpi_col = utils.detect_cpi_column(self.data)
        if not cpi_col:
            cpi_col = self._find_fallback_numeric_column()
        
        # Extract time series
        time_col = utils.detect_time_column(self.data)
        if time_col:
            return self._create_time_indexed_series(cpi_col, time_col)
        else:
            return self._create_default_indexed_series(cpi_col)
    
    def _find_fallback_numeric_column(self) -> str:
        """Find a fallback numeric column if no CPI column is detected."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cpi_col = str(numeric_cols[0])
            logger.warning(f"No CPI column detected. Using first numeric column: {cpi_col}")
            return cpi_col
        else:
            raise ValueError("No suitable CPI column found in the data")
    
    def _create_time_indexed_series(self, cpi_col: str, time_col: str) -> pd.Series:
        """Create time-indexed CPI series."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        self.data[time_col] = pd.to_datetime(self.data[time_col])
        return pd.Series(
            self.data[cpi_col].values,
            index=self.data[time_col],
            name='CPI'
        ).dropna().sort_index()
    
    def _create_default_indexed_series(self, cpi_col: str) -> pd.Series:
        """Create default-indexed CPI series when no time column is found."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        return pd.Series(
            self.data[cpi_col].values,
            index=pd.date_range(start='2010-01-01', periods=len(self.data), freq='M'),
            name='CPI'
        ).dropna()
    
    def _create_synthetic_cpi_data(
        self, n_periods: int = 120, start_date: str = '2015-01-01'
    ) -> pd.Series:
        """Create synthetic CPI data for demonstration purposes."""
        np.random.seed(42)
        
        # Create date index
        date_index = pd.date_range(start=start_date, periods=n_periods, freq='M')
        
        # Generate realistic CPI data
        base_level = 100.0
        trend = 0.2 * np.arange(n_periods)  # Gradual inflation
        seasonal = 1.5 * np.sin(2 * np.pi * np.arange(n_periods) / 12)  # Annual seasonality
        noise = np.random.normal(0, 0.8, n_periods)
        
        # Add some realistic volatility periods
        volatility_periods = np.random.choice(n_periods, size=int(n_periods * 0.1), replace=False)
        for period in volatility_periods:
            if period < n_periods - 5:
                noise[period:period+5] += np.random.normal(0, 2, 5)
        
        cpi_values = base_level + trend + seasonal + noise
        
        # Ensure CPI doesn't go negative
        cpi_values = np.maximum(cpi_values, 50.0)
        
        self.cpi_series = pd.Series(cpi_values, index=date_index, name='CPI')
        logger.info(f"Created synthetic CPI data: {len(self.cpi_series)} observations")
        
        return self.cpi_series
    
    def fit_model(self, model_name: str, test_size: int = 12, **kwargs: Any) -> dict[str, Any]:
        """
        Fit a specific forecasting model to the CPI data.
        
        Args:
            model_name: Name of the model to fit
            test_size: Number of periods to hold out for testing
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing model results
        """
        if self.cpi_series is None:
            raise RuntimeError("CPI data not loaded. Call load_cpi_data() first.")
        
        # Validate test_size
        if test_size < 0:
            raise ValueError("test_size must be non-negative")
        if test_size >= len(self.cpi_series):
            raise ValueError(
                f"test_size ({test_size}) must be less than data length ({len(self.cpi_series)})"
            )
        
        logger.info(f"Fitting {model_name} model...")
        try:
            result = self._dispatch_model_fitting(model_name, test_size, **kwargs)
            return self._validate_model_result(result, model_name)
        except Exception as e:  # noqa: BLE001 - broad to surface model errors
            logger.error(f"Error fitting {model_name} model: {e}")
            return {'error': str(e), 'model': model_name, 'forecast': None}
    
    def _dispatch_model_fitting(
        self, model_name: str, test_size: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Dispatch model fitting based on model name."""
        # ADD INPUT VALIDATION
        allowed_models = set(self.available_models.keys())
        if model_name not in allowed_models:
            raise ValueError(f"Model '{model_name}' not in allowed models: {allowed_models}")
        
        # Standard model mappings with proper type annotations
        standard_models: dict[str, Any] = {
            'arima': lambda: utils.fit_arima_series(
                self.cpi_series, test_size=test_size, **kwargs
            ),
            'ets': lambda: utils.fit_exponential_smoothing(
                self.cpi_series, test_size=test_size, **kwargs
            ),
            'prophet': lambda: utils.fit_prophet_series(
                self.cpi_series, test_size=test_size, **kwargs
            ),
            'samira': lambda: fit_samira_model(
                self.cpi_series, test_size=test_size, **kwargs
            ),
            'croston': lambda: utils.fit_croston(
                self.cpi_series, test_size=test_size, **kwargs
            ),
            'naive': lambda: self._fit_naive_model(self.cpi_series, test_size=test_size),
            'seasonal_naive': lambda: self._fit_seasonal_naive_model(
                self.cpi_series, test_size=test_size
            ),
            'drift': lambda: self._fit_drift_model(self.cpi_series, test_size=test_size),
        }
        
        # Check standard models first
        if model_name in standard_models:
            return cast(dict[str, Any], standard_models[model_name]())
        
        # SECURE: Replace dynamic attribute access with explicit mapping
        advanced_model_mapping: dict[str, Any] = {
            'neural_prophet': getattr(advanced_models, 'fit_neural_prophet_model', None),
            'ses': getattr(advanced_models, 'fit_ses_model', None),
            'holt': getattr(advanced_models, 'fit_holt_model', None),
            'tbats': getattr(advanced_models, 'fit_tbats_model', None),
            'darts_arima': getattr(advanced_models, 'fit_darts_arima_model', None),
            'darts_ets': getattr(advanced_models, 'fit_darts_ets_model', None),
        }
        
        if model_name in advanced_model_mapping:
            model_func = advanced_model_mapping[model_name]
            if model_func is not None:
                return cast(
                    dict[str, Any],
                    model_func(self.cpi_series, test_size=test_size, **kwargs),
                )
            else:
                raise ValueError(
                    f"Model function for '{model_name}' not implemented in advanced_models"
                )
        
        raise ValueError(f"Model implementation for '{model_name}' not found")
    
    def _validate_model_result(self, result: dict[str, Any], model_name: str) -> dict[str, Any]:
        """Validate and sanitize model results."""
        if result.get('error'):
            logger.error(f"Model {model_name} failed: {result['error']}")
            return result
        
        # Validate forecast
        forecast = result.get('forecast')
        if forecast is not None and not np.all(np.isfinite(forecast.values)):
            logger.warning(f"Model {model_name} produced non-finite forecast values")
            # Replace with interpolated values or fallback
            forecast = forecast.interpolate().fillna(method='bfill').fillna(method='ffill')
            result['forecast'] = forecast
        
        # Validate metrics
        for metric in ['mae', 'rmse', 'mape']:
            value = result.get(metric)
            if value is not None and (not np.isfinite(value) or value < 0):
                logger.warning(f"Invalid {metric} value for {model_name}: {value}")
                result[metric] = None
        
        return result
    
    def _fit_naive_model(self, series: pd.Series, test_size: int = 12) -> dict[str, Any]:
        """Fit naive (last value) model."""
        train = series.iloc[:-test_size] if test_size > 0 else series
        last_value = train.iloc[-1]
        
        # Create forecast index
        if test_size > 0:
            forecast_index = series.index[-test_size:]
        else:
            # Forecast beyond the series
            freq = pd.infer_freq(series.index) or 'M'
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=30),
                periods=12,
                freq=freq
            )
        
        forecast = pd.Series([last_value] * len(forecast_index), index=forecast_index)
        
        return {
            'model': 'naive',
            'forecast': forecast,
            'fitted': pd.Series([last_value] * len(train), index=train.index),
            'mae': None,
            'rmse': None
        }
    
    def _fit_seasonal_naive_model(
        self, series: pd.Series, test_size: int = 12, season_length: int = 12
    ) -> dict[str, Any]:
        """Fit seasonal naive model."""
        train = series.iloc[:-test_size] if test_size > 0 else series
        
        if len(train) < season_length:
            # Fall back to naive if insufficient data
            return self._fit_naive_model(series, test_size)
        
        # Create forecast index
        if test_size > 0:
            forecast_index = series.index[-test_size:]
        else:
            freq = pd.infer_freq(series.index) or 'M'
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=30),
                periods=12,
                freq=freq
            )
        
        # Get seasonal values
        seasonal_values = train.iloc[-season_length:].values
        forecast_values = []
        
        for i in range(len(forecast_index)):
            seasonal_idx = i % season_length
            forecast_values.append(seasonal_values[seasonal_idx])
        
        forecast = pd.Series(forecast_values, index=forecast_index)
        
        return {
            'model': 'seasonal_naive',
            'forecast': forecast,
            'fitted': None,
            'mae': None,
            'rmse': None
        }
    
    def _fit_drift_model(self, series: pd.Series, test_size: int = 12) -> dict[str, Any]:
        """Fit drift (linear trend) model."""
        train = series.iloc[:-test_size] if test_size > 0 else series
        
        # Fit linear trend
        x = np.arange(len(train))
        y = train.values
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        # Create forecast index
        if test_size > 0:
            forecast_index = series.index[-test_size:]
            start_x = len(train)
            forecast_x = np.arange(start_x, start_x + test_size)
        else:
            freq = pd.infer_freq(series.index) or 'M'
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=30),
                periods=12,
                freq=freq
            )
            start_x = len(train)
            forecast_x = np.arange(start_x, start_x + 12)
        
        forecast_values = intercept + slope * forecast_x
        forecast = pd.Series(forecast_values, index=forecast_index)
        
        # Fitted values
        fitted_values = intercept + slope * x
        fitted = pd.Series(fitted_values, index=train.index)
        
        return {
            'model': 'drift',
            'forecast': forecast,
            'fitted': fitted,
            'mae': None,
            'rmse': None
        }
    
    def generate_single_model_plot(self, model_name: str, test_size: int = 12,  # noqa: C901 - complexity acceptable for plotting
                                 figsize: tuple[int, int] = (12, 8),
                                 save_plot: bool = True, **kwargs: Any) -> matplotlib.figure.Figure:
        """
        Generate a plot for a single forecasting model.
        
        Args:
            model_name: Name of the model to plot
            test_size: Number of periods to hold out for testing
            figsize: Figure size tuple
            save_plot: Whether to save the plot to disk
            **kwargs: Additional model parameters
            
        Returns:
            matplotlib Figure object
        """
        if self.cpi_series is None:
            self.load_cpi_data()
        
        # Ensure we have data after loading
        assert self.cpi_series is not None, "Failed to load CPI data"
        
        # Fit the model
        result = self.fit_model(model_name, test_size=test_size, **kwargs)
        
        if result.get('error'):
            logger.error(f"Model fitting failed: {result['error']}")
            # Create error plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Model fitting failed:\n{result['error']}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title(f"CPI Forecasting - {self.available_models[model_name]}")
            return fig
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        train_data: pd.Series = self.cpi_series.iloc[:-test_size] if test_size > 0 else self.cpi_series
        test_data: pd.Series | None = self.cpi_series.iloc[-test_size:] if test_size > 0 else None
        
        ax.plot(train_data.index, train_data.values, 
               label='Training Data', color='blue', linewidth=2)
        
        if test_data is not None:
            ax.plot(test_data.index, test_data.values, 
                   label='Actual (Test)', color='black', linewidth=2, linestyle='--')
        
        # Plot fitted values
        if result.get('fitted') is not None:
            fitted = result['fitted']
            ax.plot(fitted.index, fitted.values, 
                   label='Fitted', color='green', linewidth=1.5, alpha=0.8)
        
        # Plot forecast
        if result.get('forecast') is not None:
            forecast = result['forecast']
            ax.plot(forecast.index, forecast.values, 
                   label='Forecast', color='red', linewidth=2.5)
        
        # Plot confidence intervals if available
        if result.get('conf_int') is not None:
            conf_int = result['conf_int']
            ax.fill_between(forecast.index, 
                           conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                           alpha=0.3, color='red', label='95% Confidence Interval')
        
        # Formatting
        ax.set_title(f"CPI Forecasting - {self.available_models[model_name]}", 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('CPI Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # type: ignore[no-untyped-call]
        ax.xaxis.set_major_locator(mdates.YearLocator())  # type: ignore[no-untyped-call]
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add metrics text if available
        metrics_text = []
        if result.get('mae') is not None:
            metrics_text.append(f"MAE: {result['mae']:.3f}")
        if result.get('rmse') is not None:
            metrics_text.append(f"RMSE: {result['rmse']:.3f}")
        if result.get('mape') is not None:
            metrics_text.append(f"MAPE: {result['mape']:.2f}%")
        
        if metrics_text:
            ax.text(0.02, 0.98, " | ".join(metrics_text), 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            filename = f"cpi_forecast_{model_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {filepath}")
        
        return fig
    
    def generate_comparison_plot(self, model_names: list[str], test_size: int = 12,
                               figsize: tuple[int, int] = (15, 10),
                               save_plot: bool = True) -> matplotlib.figure.Figure:
        """
        Generate a comparison plot for multiple forecasting models.
        
        Args:
            model_names: List of model names to compare
            test_size: Number of periods to hold out for testing
            figsize: Figure size tuple
            save_plot: Whether to save the plot to disk
            
        Returns:
            matplotlib Figure object
        """
        if self.cpi_series is None:
            self.load_cpi_data()
        
        # Ensure we have data after loading
        assert self.cpi_series is not None, "Failed to load CPI data"
        
        # Fit all models
        results = {}
        for model_name in model_names:
            results[model_name] = self.fit_model(model_name, test_size=test_size)
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        train_data: pd.Series = self.cpi_series.iloc[:-test_size] if test_size > 0 else self.cpi_series
        test_data: pd.Series | None = self.cpi_series.iloc[-test_size:] if test_size > 0 else None
        
        ax1.plot(train_data.index, train_data.values, 
                label='Training Data', color='black', linewidth=2.5)
        
        if test_data is not None:
            ax1.plot(test_data.index, test_data.values, 
                    label='Actual (Test)', color='black', linewidth=2.5, linestyle='--')
        
        # Plot forecasts for each model
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(model_names)))
        metrics_data = []
        
        for _, (model_name, color) in enumerate(zip(model_names, colors, strict=True)):
            result = results[model_name]
            
            if result.get('error'):
                logger.warning(f"Skipping {model_name} due to error: {result['error']}")
                continue
            
            if result.get('forecast') is not None:
                forecast = result['forecast']
                ax1.plot(forecast.index, forecast.values, 
                        label=f'{model_name.upper()}', color=color, linewidth=2)
            
            # Collect metrics
            if result.get('mae') is not None or result.get('rmse') is not None:
                metrics_data.append({
                    'Model': model_name.upper(),
                    'MAE': result.get('mae', np.nan),
                    'RMSE': result.get('rmse', np.nan),
                    'MAPE': result.get('mape', np.nan)
                })
        
        # Formatting main plot
        ax1.set_title('CPI Forecasting - Model Comparison', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('CPI Value', fontsize=12)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # type: ignore[no-untyped-call]
        ax1.xaxis.set_major_locator(mdates.YearLocator())  # type: ignore[no-untyped-call]
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Metrics table
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            # Create table
            ax2.axis('tight')
            ax2.axis('off')
            
            table = ax2.table(cellText=metrics_df.round(3).values,
                             colLabels=metrics_df.columns,
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            ax2.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No metrics available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.axis('off')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            model_names_str = "_".join(model_names)
            filename = f"cpi_forecast_comparison_{model_names_str}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved: {filepath}")
        
        return fig
    
    def generate_model_ranking_plot(self, model_names: list[str], test_size: int = 12,  # noqa: C901 - complexity acceptable for plotting
                                  cv_folds: int = 3, figsize: tuple[int, int] = (12, 8),
                                  save_plot: bool = True) -> matplotlib.figure.Figure:
        """
        Generate a model ranking plot based on cross-validation performance.
        
        Args:
            model_names: List of model names to evaluate
            test_size: Number of periods to hold out for testing
            cv_folds: Number of cross-validation folds
            figsize: Figure size tuple
            save_plot: Whether to save the plot to disk
            
        Returns:
            matplotlib Figure object
        """
        if self.cpi_series is None:
            self.load_cpi_data()
        
        logger.info(f"Performing cross-validation with {cv_folds} folds...")
        
        # Define simple model functions for CV
        def create_model_function(model_name: str) -> Any:
            def model_func(train_series: pd.Series, horizon: int) -> list[float]:
                try:
                    result = self.fit_model(model_name, test_size=0)
                    if result.get('forecast') is not None:
                        forecast_values = result['forecast'].values[:horizon]
                        if len(forecast_values) < horizon:
                            # Pad with last value if forecast is shorter
                            last_val = forecast_values[-1] if len(forecast_values) > 0 else train_series.iloc[-1]
                            forecast_values = list(forecast_values) + [last_val] * (horizon - len(forecast_values))
                        return [float(x) for x in forecast_values[:horizon]]
                    else:
                        # Fallback to naive forecast
                        return [float(train_series.iloc[-1])] * horizon
                except Exception as e:
                    logger.warning(f"Error in {model_name} CV: {e}")
                    return [float(train_series.iloc[-1])] * horizon
            return model_func
        
        # Create model dictionary for CV
        models_dict = {name: create_model_function(name) for name in model_names}
        
        try:
            # Compute CV metrics
            cv_results = compute_model_cv_metrics(
                self.cpi_series, 
                horizon=test_size, 
                n_folds=cv_folds, 
                models_to_eval=models_dict
            )
            
            # Prepare ranking data
            ranking_data = []
            for model_name, (mae, rmse) in cv_results.items():
                ranking_data.append({
                    'Model': model_name.upper(),
                    'MAE': mae,
                    'RMSE': rmse,
                    'Combined_Score': (mae + rmse) / 2  # Simple combined score
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df = ranking_df.sort_values('Combined_Score')
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            # Fallback to simple single-split evaluation
            ranking_data = []
            for model_name in model_names:
                result = self.fit_model(model_name, test_size=test_size)
                mae = result.get('mae', np.inf)
                rmse = result.get('rmse', np.inf)
                ranking_data.append({
                    'Model': model_name.upper(),
                    'MAE': mae if mae is not None else np.inf,
                    'RMSE': rmse if rmse is not None else np.inf,
                    'Combined_Score': ((mae or np.inf) + (rmse or np.inf)) / 2
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df = ranking_df.sort_values('Combined_Score')
        
        # Create ranking plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # MAE ranking
        sns.barplot(data=ranking_df, y='Model', x='MAE', ax=ax1, palette='viridis')
        ax1.set_title('Model Ranking by MAE', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Mean Absolute Error', fontsize=12)
        
        # RMSE ranking
        sns.barplot(data=ranking_df, y='Model', x='RMSE', ax=ax2, palette='plasma')
        ax2.set_title('Model Ranking by RMSE', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Root Mean Square Error', fontsize=12)
        
        # Add value labels on bars
        for ax in [ax1, ax2]:
            for i, (_, row) in enumerate(ranking_df.iterrows()):
                metric = 'MAE' if ax == ax1 else 'RMSE'
                value = row[metric]
                if np.isfinite(value):
                    ax.text(value + value*0.01, i, f'{value:.3f}', 
                           va='center', fontsize=10)
        
        plt.suptitle('CPI Forecasting Model Performance Ranking', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            filename = "cpi_model_ranking.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Ranking plot saved: {filepath}")
        
        return fig
    
    def generate_interactive_dashboard(self) -> str:
        """
        Generate an interactive HTML dashboard for CPI forecasting.
        
        Returns:
            Path to the generated HTML file
        """
        if self.cpi_series is None:
            self.load_cpi_data()
        
        # Ensure we have data after loading
        assert self.cpi_series is not None, "Failed to load CPI data"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CPI Forecasting Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .model-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .model-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f9f9f9; }}
                .model-card:hover {{ background-color: #e6f3ff; cursor: pointer; }}
                .stats {{ background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                button {{ background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
                button:hover {{ background-color: #45a049; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔮 CPI Forecasting Dashboard</h1>
                <p>Interactive forecasting dashboard for Consumer Price Index data</p>
                <p><strong>Data Period:</strong> {self.cpi_series.index[0].strftime('%Y-%m-%d')} to {self.cpi_series.index[-1].strftime('%Y-%m-%d')}</p>
                <p><strong>Total Observations:</strong> {len(self.cpi_series)}</p>
            </div>
            
            <div class="stats">
                <h3>📊 Data Statistics</h3>
                <p><strong>Mean:</strong> {self.cpi_series.mean():.2f}</p>
                <p><strong>Standard Deviation:</strong> {self.cpi_series.std():.2f}</p>
                <p><strong>Min:</strong> {self.cpi_series.min():.2f}</p>
                <p><strong>Max:</strong> {self.cpi_series.max():.2f}</p>
            </div>
            
            <h2>🤖 Available Forecasting Models</h2>
            <div class="model-grid">
        """
        
        for model_name, description in self.available_models.items():
            html_content += f"""
                <div class="model-card" onclick="generateModelPlot('{model_name}')">
                    <h4>{model_name.upper()}</h4>
                    <p>{description}</p>
                    <button onclick="event.stopPropagation(); generateModelPlot('{model_name}')">Generate Plot</button>
                </div>
            """
        
        html_content += """
            </div>
            
            <div style="margin: 30px 0;">
                <h3>🔄 Batch Operations</h3>
                <button onclick="generateComparisonPlot()">Compare All Models</button>
                <button onclick="generateRankingPlot()">Model Rankings</button>
                <button onclick="generateQuickComparison()">Top 5 Models</button>
            </div>
            
            <div id="output" style="margin-top: 30px;">
                <h3>📈 Generated Plots</h3>
                <p>Generated plots will be saved in the output directory and can be viewed there.</p>
            </div>
            
            <script>
                function generateModelPlot(modelName) {
                    alert(`Generating plot for ${modelName.toUpperCase()} model...\\nPlot will be saved in the output directory.`);
                    console.log(`Generate plot for model: ${modelName}`);
                }
                
                function generateComparisonPlot() {
                    alert('Generating comparison plot for all models...\\nThis may take a few minutes.');
                }
                
                function generateRankingPlot() {
                    alert('Generating model ranking plot...\\nThis will perform cross-validation.');
                }
                
                function generateQuickComparison() {
                    alert('Generating quick comparison for top 5 models...');
                }
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        html_filepath = os.path.join(self.output_dir, "cpi_forecasting_dashboard.html")
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive dashboard saved: {html_filepath}")
        return html_filepath
    
    def print_available_models(self) -> None:
        """Print all available forecasting models."""
        print("\n🤖 Available CPI Forecasting Models:")
        print("=" * 60)
        for i, (model_name, description) in enumerate(self.available_models.items(), 1):
            print(f"{i:2d}. {model_name.upper():<20} - {description}")
        print("=" * 60)
    
    def run_interactive_mode(self) -> None:  # noqa: C901, PLR0912, PLR0915 - interactive CLI is verbose by design
        """Run the plot generator in interactive mode."""
        print("\n🔮 Welcome to the CPI Forecasting Plot Generator!")
        print("=" * 60)
        
        # Load data
        if self.cpi_series is None:
            print("📊 Loading CPI data...")
            self.load_cpi_data()
        
        # Ensure we have data after loading
        assert self.cpi_series is not None, "Failed to load CPI data"
        
        print(f"✅ Loaded {len(self.cpi_series)} CPI observations")
        print(f"📅 Period: {self.cpi_series.index[0].strftime('%Y-%m')} to {self.cpi_series.index[-1].strftime('%Y-%m')}")
        
        while True:
            print("\n" + "=" * 60)
            print("Choose an option:")
            print("1. View available models")
            print("2. Generate single model plot")
            print("3. Generate model comparison plot")
            print("4. Generate model ranking plot")
            print("5. Generate interactive dashboard")
            print("6. Exit")
            print("=" * 60)
            
            try:
                choice = input("Enter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.print_available_models()
                
                elif choice == '2':
                    self.print_available_models()
                    model_name = input("\nEnter model name: ").strip().lower()
                    if model_name in self.available_models:
                        test_size = int(input("Enter test size (default 12): ") or "12")
                        print(f"🔄 Generating plot for {model_name.upper()}...")
                        self.generate_single_model_plot(model_name, test_size=test_size)
                        plt.show()
                        print("✅ Plot generated and saved!")
                    else:
                        print("❌ Invalid model name!")
                
                elif choice == '3':
                    self.print_available_models()
                    models_input = input("\nEnter model names (comma-separated): ").strip()
                    model_names = [name.strip().lower() for name in models_input.split(',')]
                    valid_models = [name for name in model_names if name in self.available_models]
                    
                    if valid_models:
                        test_size = int(input("Enter test size (default 12): ") or "12")
                        print(f"🔄 Generating comparison plot for {len(valid_models)} models...")
                        self.generate_comparison_plot(valid_models, test_size=test_size)
                        plt.show()
                        print("✅ Comparison plot generated and saved!")
                    else:
                        print("❌ No valid model names provided!")
                
                elif choice == '4':
                    print("🔄 Generating model ranking plot (this may take a while)...")
                    model_names = ['naive', 'drift', 'ses', 'arima', 'ets']  # Quick models for ranking
                    self.generate_model_ranking_plot(model_names)
                    plt.show()
                    print("✅ Ranking plot generated and saved!")
                
                elif choice == '5':
                    print("🔄 Generating interactive dashboard...")
                    html_path = self.generate_interactive_dashboard()
                    print(f"✅ Dashboard generated: {html_path}")
                    print("💡 Open the HTML file in your web browser to use the interactive dashboard.")
                
                elif choice == '6':
                    print("👋 Thank you for using the CPI Forecasting Plot Generator!")
                    break
                
                else:
                    print("❌ Invalid choice! Please enter 1-6.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# Convenience function for quick usage
def quick_cpi_forecast(model_name: str = 'arima', data_path: str | None = None, 
                      show_plot: bool = True) -> matplotlib.figure.Figure:
    """
    Quick function to generate a CPI forecast plot with a single model.
    
    Args:
        model_name: Name of the forecasting model to use
        data_path: Optional path to CPI data file
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    generator = CPIPlotGenerator(data_path=data_path)
    generator.load_cpi_data()
    
    fig = generator.generate_single_model_plot(model_name, save_plot=True)
    
    if show_plot:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Run in interactive mode when script is executed directly
    generator = CPIPlotGenerator()
    generator.run_interactive_mode()