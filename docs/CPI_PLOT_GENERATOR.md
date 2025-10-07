# CPI Forecasting Plot Generator

A comprehensive plot generator for Consumer Price Index (CPI) forecasting with multiple model options and interactive features.

## Features

- **Multiple Forec### Output Directory Structure

```text
data/processed/visualizations/
├── cpi_forecast_arima.png          # Single model plots
├── cpi_forecast_ets.png
├── cpi_forecast_comparison_*.png   # Comparison plots
├── cpi_model_ranking.png           # Ranking plots
└── cpi_forecasting_dashboard.html  # Interactive dashboard
```els**: ARIMA, ETS, Prophet, SAMIRA, Neural Prophet, and simple baseline models
- **Interactive Visualizations**: Single model plots, model comparisons, and performance rankings
- **Flexible Data Input**: Auto-detects CPI data files or creates synthetic data for demonstration
- **Export Options**: High-quality PNG plots and interactive HTML dashboard
- **Command-line Interface**: Easy-to-use CLI for batch operations
- **Cross-validation**: Model performance evaluation with multiple metrics

## Quick Start

### Python API

```python
from src.visualization.cpi_plot_generator import CPIPlotGenerator

# Initialize the generator
generator = CPIPlotGenerator()

# Load CPI data (auto-detects from raw data directory)
cpi_data = generator.load_cpi_data()

# Generate a single model forecast plot
fig = generator.generate_single_model_plot('arima', test_size=12)

# Compare multiple models
fig = generator.generate_comparison_plot(['naive', 'arima', 'ets'], test_size=12)

# Generate model ranking plot
fig = generator.generate_model_ranking_plot(['naive', 'drift', 'arima'], cv_folds=3)

# Create interactive dashboard
html_path = generator.generate_interactive_dashboard()
```

### Quick Forecast Function

```python
from src.visualization.cpi_plot_generator import quick_cpi_forecast

# Generate quick ARIMA forecast
fig = quick_cpi_forecast(model_name='arima', show_plot=True)
```

### Command-line Interface

```bash
# Generate single model plot
python tools/cpi_forecast_cli.py --model arima

# Compare multiple models
python tools/cpi_forecast_cli.py --compare naive drift arima ets

# Rank models by performance
python tools/cpi_forecast_cli.py --rank naive drift ses arima ets

# Generate interactive dashboard
python tools/cpi_forecast_cli.py --dashboard

# Run in interactive mode
python tools/cpi_forecast_cli.py --interactive

# List available models
python tools/cpi_forecast_cli.py --list-models
```

## Available Models

| Model | Description | Type |
|-------|-------------|------|
| `arima` | AutoRegressive Integrated Moving Average | Statistical |
| `ets` | Exponential Smoothing | Statistical |
| `prophet` | Facebook's Time Series Forecasting | ML-based |
| `samira` | Self-Adaptive Model for Interval Regression | Advanced |
| `neural_prophet` | Neural Network based Prophet | Deep Learning |
| `croston` | Intermittent Demand Forecasting | Specialized |
| `ses` | Simple Exponential Smoothing | Simple |
| `holt` | Double Exponential Smoothing | Simple |
| `naive` | Last Value Persistence | Baseline |
| `seasonal_naive` | Last Seasonal Value | Baseline |
| `drift` | Linear Trend Extrapolation | Baseline |

## Data Input Options

1. **Auto-detection**: Automatically finds CPI data files in `data/raw/` directory
2. **Custom file path**: Specify path to your CPI data file
3. **Synthetic data**: Creates realistic synthetic CPI data for demonstration

### Supported File Formats

- Excel files (`.xlsx`, `.xls`)
- CSV files (`.csv`)

### Data Requirements

The system automatically detects:

- **CPI columns**: Columns containing 'cpi', 'consumer_price_index', 'price_index', or 'inflation'
- **Time columns**: Date/datetime columns for time series indexing

## Output Options

### Plot Types

1. **Single Model Plots**: Individual forecast visualizations with confidence intervals
2. **Comparison Plots**: Side-by-side model comparisons with performance metrics
3. **Ranking Plots**: Model performance rankings based on cross-validation
4. **Interactive Dashboard**: HTML dashboard for interactive exploration

### Export Formats

- **PNG**: High-resolution plots (300 DPI)
- **HTML**: Interactive dashboard with model selection
- **Metrics Tables**: Performance comparison tables

## Configuration Options

### Forecasting Parameters

- `test_size`: Number of periods to hold out for testing (default: 12)
- `cv_folds`: Number of cross-validation folds for ranking (default: 3)
- `figsize`: Plot dimensions as (width, height) tuple

### Model-specific Parameters

Different models accept various parameters:

```python
# ARIMA with custom parameters
generator.fit_model('arima', test_size=12, order=(1,1,1))

# ETS with trend and seasonality
generator.fit_model('ets', test_size=12, trend='add', seasonal='add')

# Prophet with custom seasonality
generator.fit_model('prophet', test_size=12, yearly_seasonality=True)
```

## Interactive Mode

Run the interactive mode for guided plot generation:

```python
generator = CPIPlotGenerator()
generator.run_interactive_mode()
```

This provides a menu-driven interface for:

- Viewing available models
- Generating single model plots
- Creating model comparisons
- Generating rankings
- Creating interactive dashboards

## Example Workflows

### 1. Basic CPI Forecasting

```python
from src.visualization.cpi_plot_generator import CPIPlotGenerator

# Initialize and load data
generator = CPIPlotGenerator()
generator.load_cpi_data()

# Generate ARIMA forecast
generator.generate_single_model_plot('arima', test_size=12, save_plot=True)
```

### 2. Model Comparison Study

```python
# Compare statistical vs ML models
models = ['arima', 'ets', 'prophet', 'naive']
generator.generate_comparison_plot(models, test_size=12, save_plot=True)

# Rank by performance
generator.generate_model_ranking_plot(models, cv_folds=5, save_plot=True)
```

### 3. Interactive Analysis

```python
# Create dashboard for interactive exploration
html_path = generator.generate_interactive_dashboard()
print(f"Open {html_path} in your browser for interactive analysis")
```

### 4. Batch Processing

```bash
# Generate all model comparisons
python tools/cpi_forecast_cli.py --compare naive drift ses arima ets prophet
```

## Output Directory Structure

```
data/processed/visualizations/
├── cpi_forecast_arima.png          # Single model plots
├── cpi_forecast_ets.png
├── cpi_forecast_comparison_*.png   # Comparison plots
├── cpi_model_ranking.png           # Ranking plots
└── cpi_forecasting_dashboard.html  # Interactive dashboard
```

## Error Handling

The system includes robust error handling:

- **Missing data**: Creates synthetic CPI data for demonstration
- **Model failures**: Gracefully handles model fitting errors
- **Invalid parameters**: Provides helpful error messages
- **File format issues**: Attempts multiple parsing strategies

## Performance Notes

- **Simple models** (naive, drift): Very fast, good for quick testing
- **Statistical models** (ARIMA, ETS): Moderate speed, good accuracy
- **ML models** (Prophet, SAMIRA): Slower but potentially more accurate
- **Cross-validation**: Can be time-consuming for many models/folds

## Tips for Best Results

1. **Start with simple models** to verify data loading and basic functionality
2. **Use appropriate test sizes** (typically 10-20% of data length)
3. **Consider seasonality** when choosing models for monthly CPI data
4. **Use cross-validation** for robust model selection
5. **Check the interactive dashboard** for comprehensive analysis

## Troubleshooting

### Common Issues

1. **"No CPI column found"**:
   - Check if your data contains CPI-related column names
   - The system falls back to synthetic data for demonstration

2. **Model fitting errors**:
   - Some models require minimum data lengths
   - Try simpler models first (naive, drift)

3. **Import errors**:
   - Ensure all dependencies are installed
   - Run from the project root directory

4. **Plot display issues**:
   - Set matplotlib backend: `matplotlib.use('Agg')` for headless environments
   - Use `save_plot=True` to save files instead of displaying

### Getting Help

- Check the logs for detailed error messages
- Use the `--list-models` flag to see available models
- Start with the interactive mode for guided usage
- Refer to individual model documentation for specific parameters