"""
SAMIRA Model Visualization Utilities

Specialized plotting functions for visualizing SAMIRA model results,
including component decomposition, forecasts with uncertainty, and
adaptive coefficient evolution.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Optional, Tuple

from ..core.logutil import get_logger

logger = get_logger(__name__)


def plot_samira_forecast(forecast_result: Dict,
                        original_series: pd.Series,
                        save_path: Optional[str] = None,
                        title: str = "SAMIRA Forecast") -> plt.Figure:
    """
    Plot SAMIRA forecast with confidence intervals
    
    Args:
        forecast_result: Result dictionary from fit_samira_model
        original_series: Original time series
        save_path: Path to save plot (optional)
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original series
    ax.plot(original_series.index, original_series.values, 
            label='Observed', color='blue', linewidth=1.5)
    
    # Plot fitted values
    if forecast_result.get('fitted') is not None:
        fitted = forecast_result['fitted']
        ax.plot(fitted.index, fitted.values, 
                label='Fitted', color='green', linewidth=1, alpha=0.8)
    
    # Plot forecast
    if forecast_result.get('forecast') is not None:
        forecast = forecast_result['forecast']
        ax.plot(forecast.index, forecast.values, 
                label='Forecast', color='red', linewidth=2)
        
        # Add confidence intervals
        if 'confidence_intervals' in forecast_result:
            ci = forecast_result['confidence_intervals']
            if ci.get('lower') is not None and ci.get('upper') is not None:
                ax.fill_between(forecast.index, 
                              ci['lower'].values, ci['upper'].values,
                              alpha=0.3, color='red', label='95% CI')
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"SAMIRA forecast plot saved to {save_path}")
    
    return fig


def plot_samira_components(forecast_result: Dict,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot SAMIRA model component decomposition
    
    Args:
        forecast_result: Result dictionary from fit_samira_model
        save_path: Path to save plot (optional)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    components = forecast_result.get('components', {})
    
    if not components:
        raise ValueError("No components found in forecast result")
    
    # Filter components for plotting
    plot_components = {}
    for name, series in components.items():
        if not name.startswith('coeff_'):  # Skip coefficient series for main plot
            plot_components[name] = series
    
    n_components = len(plot_components)
    fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True)
    
    if n_components == 1:
        axes = [axes]
    
    for i, (name, series) in enumerate(plot_components.items()):
        axes[i].plot(series.index, series.values, linewidth=1.5)
        axes[i].set_title(f'{name.replace("_", " ").title()}', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Add zero line for effects
        if 'effect_' in name:
            axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.suptitle('SAMIRA Model Components', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"SAMIRA components plot saved to {save_path}")
    
    return fig


def plot_adaptive_coefficients(forecast_result: Dict,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot evolution of adaptive regression coefficients
    
    Args:
        forecast_result: Result dictionary from fit_samira_model
        save_path: Path to save plot (optional)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    components = forecast_result.get('components', {})
    
    # Extract coefficient series
    coeff_components = {name: series for name, series in components.items() 
                       if name.startswith('coeff_')}
    
    if not coeff_components:
        raise ValueError("No adaptive coefficients found in forecast result")
    
    n_coeffs = len(coeff_components)
    fig, axes = plt.subplots(n_coeffs, 1, figsize=figsize, sharex=True)
    
    if n_coeffs == 1:
        axes = [axes]
    
    for i, (name, series) in enumerate(coeff_components.items()):
        var_name = name.replace('coeff_', '')
        
        # Plot coefficient evolution
        axes[i].plot(series.index, series.values, linewidth=2, label=f'{var_name} coefficient')
        
        # Add moving average trend
        window = min(12, len(series) // 4)
        if window > 1:
            trend = series.rolling(window=window, center=True).mean()
            axes[i].plot(trend.index, trend.values, 
                        linestyle='--', alpha=0.7, label='Trend')
        
        axes[i].set_title(f'Adaptive Coefficient: {var_name}', fontsize=12)
        axes[i].set_ylabel('Coefficient Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add zero line
        axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.suptitle('Evolution of Adaptive Regression Coefficients', 
                 fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Adaptive coefficients plot saved to {save_path}")
    
    return fig


def plot_samira_diagnostics(forecast_result: Dict,
                           original_series: pd.Series,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create comprehensive SAMIRA model diagnostics plot
    
    Args:
        forecast_result: Result dictionary from fit_samira_model
        original_series: Original time series
        save_path: Path to save plot (optional)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
    
    # Main forecast plot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot original and fitted
    ax1.plot(original_series.index, original_series.values, 
             label='Observed', color='blue', linewidth=1.5)
    
    if forecast_result.get('fitted') is not None:
        fitted = forecast_result['fitted']
        ax1.plot(fitted.index, fitted.values, 
                label='Fitted', color='green', linewidth=1)
    
    # Plot forecast if available
    if forecast_result.get('forecast') is not None:
        forecast = forecast_result['forecast']
        ax1.plot(forecast.index, forecast.values, 
                label='Forecast', color='red', linewidth=2)
        
        if 'confidence_intervals' in forecast_result:
            ci = forecast_result['confidence_intervals']
            if ci.get('lower') is not None and ci.get('upper') is not None:
                ax1.fill_between(forecast.index, 
                               ci['lower'].values, ci['upper'].values,
                               alpha=0.3, color='red')
    
    ax1.set_title('SAMIRA Model: Fit and Forecast', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2 = fig.add_subplot(gs[1, 0])
    if forecast_result.get('fitted') is not None:
        fitted = forecast_result['fitted']
        residuals = original_series.loc[fitted.index] - fitted
        ax2.plot(residuals.index, residuals.values, color='purple', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Residuals')
        ax2.grid(True, alpha=0.3)
    
    # Residual distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if forecast_result.get('fitted') is not None:
        residuals = original_series.loc[fitted.index] - fitted
        ax3.hist(residuals.values, bins=20, alpha=0.7, color='purple', density=True)
        ax3.set_title('Residual Distribution')
        ax3.set_xlabel('Residual Value')
        ax3.set_ylabel('Density')
    
    # Model info
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Display model metrics and info
    info_text = []
    if 'mae' in forecast_result:
        info_text.append(f"MAE: {forecast_result['mae']:.4f}")
    if 'rmse' in forecast_result:
        info_text.append(f"RMSE: {forecast_result['rmse']:.4f}")
    if 'mape' in forecast_result:
        info_text.append(f"MAPE: {forecast_result['mape']:.2f}%")
    if 'adaptation_rate' in forecast_result:
        info_text.append(f"Adaptation Rate: {forecast_result['adaptation_rate']:.3f}")
    if 'seasonal_period' in forecast_result:
        info_text.append(f"Seasonal Period: {forecast_result['seasonal_period']}")
    
    info_str = " | ".join(info_text)
    ax4.text(0.5, 0.5, info_str, horizontalalignment='center', 
             verticalalignment='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"SAMIRA diagnostics plot saved to {save_path}")
    
    return fig


def create_samira_report(forecast_result: Dict,
                        original_series: pd.Series,
                        output_dir: str,
                        series_name: str = "series") -> Dict[str, str]:
    """
    Create a comprehensive SAMIRA model report with multiple visualizations
    
    Args:
        forecast_result: Result dictionary from fit_samira_model
        original_series: Original time series
        output_dir: Directory to save plots
        series_name: Name for the series (used in filenames)
        
    Returns:
        Dictionary mapping plot types to saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = {}
    
    try:
        # Main forecast plot
        forecast_path = os.path.join(output_dir, f"{series_name}_samira_forecast.png")
        plot_samira_forecast(forecast_result, original_series, forecast_path)
        saved_plots['forecast'] = forecast_path
        
        # Components plot
        if forecast_result.get('components'):
            components_path = os.path.join(output_dir, f"{series_name}_samira_components.png")
            plot_samira_components(forecast_result, components_path)
            saved_plots['components'] = components_path
            
            # Adaptive coefficients plot
            coeff_components = {name: series for name, series in forecast_result['components'].items() 
                              if name.startswith('coeff_')}
            if coeff_components:
                coeffs_path = os.path.join(output_dir, f"{series_name}_samira_coefficients.png")
                plot_adaptive_coefficients(forecast_result, coeffs_path)
                saved_plots['coefficients'] = coeffs_path
        
        # Diagnostics plot
        diagnostics_path = os.path.join(output_dir, f"{series_name}_samira_diagnostics.png")
        plot_samira_diagnostics(forecast_result, original_series, diagnostics_path)
        saved_plots['diagnostics'] = diagnostics_path
        
        logger.info(f"SAMIRA report created for {series_name}: {len(saved_plots)} plots saved")
        
    except Exception as e:
        logger.error(f"Error creating SAMIRA report: {e}")
    
    return saved_plots