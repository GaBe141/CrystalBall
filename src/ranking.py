import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_model_rankings(models: Dict,
                           metrics: List[str] = ['mae', 'rmse', 'mape'],
                           weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Compute comprehensive model rankings based on multiple metrics.
    
    Args:
        models: Dict of model results
        metrics: List of metrics to consider
        weights: Optional dict of metric weights (default: equal weights)
        
    Returns:
        DataFrame with model scores and rankings
    """
    if weights is None:
        weights = {m: 1 / max(1, len(metrics)) for m in metrics}

    def _to_float(x: Any) -> float:
        try:
            if x is None:
                return np.nan
            # already numeric
            if isinstance(x, (int, float, np.number)):
                return float(x)
            # try coercion
            return float(x)
        except Exception:
            return np.nan
        
    scores = []
    for model_name, result in models.items():
        score = {}
        score['model'] = model_name
        
        # Get available metrics
        for metric in metrics:
            val = result.get(metric, np.nan) if isinstance(result, dict) else np.nan
            score[metric] = _to_float(val)
                
        # Compute weighted score
        valid_metrics = [m for m in metrics if m in weights and not pd.isna(score[m]) and np.isfinite(score[m])]
        if valid_metrics:
            denom = sum(weights[m] for m in valid_metrics)
            denom = denom if denom and np.isfinite(denom) else 1.0
            norm_weights = {m: float(weights[m]) / denom for m in valid_metrics}
            score['weighted_score'] = float(sum(float(score[m]) * norm_weights[m] for m in valid_metrics))
        else:
            score['weighted_score'] = np.nan
            
        scores.append(score)
        
    df = pd.DataFrame(scores)
    if 'weighted_score' not in df.columns:
        df['weighted_score'] = np.nan
    # Lower scores better for error metrics; rank NaNs last
    df['rank'] = df['weighted_score'].rank(method='min', na_option='bottom', ascending=True)
    return df.sort_values(['rank', 'model']).reset_index(drop=True)

def visualize_rankings(rankings: pd.DataFrame,
                     title: str = "Model Performance Rankings",
                     figsize: Tuple[int,int] = (12,6)) -> plt.Figure:
    """
    Create a comprehensive ranking visualization.
    
    Args:
        rankings: DataFrame from compute_model_rankings
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Weighted scores
    # Use hue to avoid seaborn warning about palette without hue
    sns.barplot(
        data=rankings,
        x='weighted_score',
        y='model',
        hue='model',
        ax=ax1,
        palette='viridis',
        legend=False,
    )
    ax1.set_title('Overall Model Performance')
    ax1.set_xlabel('Weighted Score')
    
    # Right plot: Individual metrics
    metrics = [c for c in rankings.columns 
              if c not in ['model','rank','weighted_score']]
    
    data_long = rankings.melt(
        id_vars=['model'],
        value_vars=metrics,
        var_name='Metric',
        value_name='Value',
    )
    
    sns.barplot(data=data_long,
                x='Value',
                y='model',
                hue='Metric',
                ax=ax2)
    ax2.set_title('Individual Metrics')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.05)
    return fig

def evaluate_forecast_adherence(actual: pd.Series,
                              forecast: pd.Series | None,
                              window: int = 5) -> Dict:
    """
    Evaluate how well a forecast adheres to historical patterns.
    
    Args:
        actual: Actual time series values
        forecast: Forecast values
        window: Rolling window size for pattern analysis
        
    Returns:
        Dict of adherence metrics
    """
    metrics = {}
    # Guard against missing/empty forecast
    if forecast is None or len(forecast) == 0 or actual is None or len(actual) == 0:
        return {'direction_accuracy': np.nan, 'volatility_ratio': np.nan, 'seasonal_correlation': np.nan}

    # Align on common index or fallback to tail alignment by length
    try:
        common_idx = actual.index.intersection(forecast.index)
        if len(common_idx) >= max(3, window):
            a = actual.loc[common_idx].astype(float)
            f = forecast.loc[common_idx].astype(float)
        else:
            # fallback by position
            n = min(len(actual), len(forecast))
            a = actual.astype(float).tail(n)
            f = forecast.astype(float).tail(n)
    except Exception:
        n = min(len(actual), len(forecast))
        a = actual.astype(float).tail(n)
        f = forecast.astype(float).tail(n)

    if len(a) < 3 or len(f) < 3:
        return {'direction_accuracy': np.nan, 'volatility_ratio': np.nan, 'seasonal_correlation': np.nan}

    # Basic pattern adherence
    actual_diff = a.diff()
    forecast_diff = f.diff()

    # Direction accuracy
    try:
        direction_match = (np.sign(actual_diff) == np.sign(forecast_diff))
        metrics['direction_accuracy'] = float(direction_match.mean())
    except Exception:
        metrics['direction_accuracy'] = np.nan
    
    # Volatility adherence
    try:
        actual_vol = actual_diff.rolling(window).std()
        forecast_vol = forecast_diff.rolling(window).std()
        metrics['volatility_ratio'] = float((forecast_vol / actual_vol).mean())
    except Exception:
        metrics['volatility_ratio'] = np.nan
    
    # Cyclical pattern preservation (if applicable)
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        actual_seasonal = seasonal_decompose(a, period=window).seasonal
        forecast_seasonal = seasonal_decompose(f, period=window).seasonal
        metrics['seasonal_correlation'] = float(actual_seasonal.corr(forecast_seasonal))
    except:
        metrics['seasonal_correlation'] = np.nan
        
    return metrics

def create_adherence_report(models: Dict,
                          actual: pd.Series,
                          output_dir: str,
                          base_name: str | None = None) -> str:
    """
    Generate a comprehensive adherence report with visualizations.
    
    Args:
        models: Dict of model results
        actual: Actual time series values
        output_dir: Directory to save visualizations
        
    Returns:
        Path to generated report
    """
    results = []
    for model_name, result in models.items():
        if not isinstance(result, dict):
            continue
        fc = result.get('forecast')
        if fc is None or (hasattr(fc, '__len__') and len(fc) == 0):
            continue
        adherence = evaluate_forecast_adherence(actual, fc)
        adherence['model'] = model_name
        results.append(adherence)

    adherence_df = pd.DataFrame(results)

    # If nothing to show, create a simple placeholder image and return early
    if adherence_df.empty or 'model' not in adherence_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, 'No valid forecasts to evaluate adherence.', ha='center', va='center', fontsize=12)
        fname = f"{base_name}_adherence_analysis.png" if base_name else 'adherence_analysis.png'
        viz_path = os.path.join(output_dir, fname)
        fig.savefig(viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        return viz_path

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Forecast Pattern Adherence Analysis', size=14)

    # Direction accuracy
    sns.barplot(data=adherence_df,
                x='model',
                y='direction_accuracy',
                ax=axes[0, 0])
    axes[0, 0].set_title('Direction Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Volatility ratio
    sns.barplot(data=adherence_df,
                x='model',
                y='volatility_ratio',
                ax=axes[0, 1])
    axes[0, 1].set_title('Volatility Ratio (1.0 = Perfect)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Seasonal correlation
    if 'seasonal_correlation' in adherence_df.columns and not adherence_df['seasonal_correlation'].isna().all():
        sns.barplot(data=adherence_df,
                    x='model',
                    y='seasonal_correlation',
                    ax=axes[1, 0])
        axes[1, 0].set_title('Seasonal Pattern Preservation')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Seasonality not detected or insufficient data.', ha='center', va='center')

    # Combined visualization
    try:
        if actual is not None and len(actual) > 0:
            actual.plot(ax=axes[1, 1], label='Actual', color='black')
    except Exception:
        pass
    for model_name, result in models.items():
        if isinstance(result, dict):
            fc = result.get('forecast')
            if fc is None or (hasattr(fc, '__len__') and len(fc) == 0):
                continue
            try:
                fc.plot(ax=axes[1, 1], label=f"{model_name}", alpha=0.7)
            except Exception:
                continue
    axes[1, 1].set_title('Forecasts vs Actual')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1))

    plt.tight_layout()

    # Save visualization
    fname = f"{base_name}_adherence_analysis.png" if base_name else 'adherence_analysis.png'
    viz_path = os.path.join(output_dir, fname)
    fig.savefig(viz_path, bbox_inches='tight', dpi=300)
    plt.close()

    return viz_path