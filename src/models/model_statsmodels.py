# StatsModels forecasting model wrapper
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_statsmodels_arima(df, target_col, order=(1,1,1), forecast_horizon=12):
    series = pd.to_numeric(df[target_col], errors='coerce').dropna()
    model = ARIMA(series, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=forecast_horizon)
    return forecast

# Example usage:
# forecast = train_statsmodels_arima(df, 'value', order=(1,1,1), forecast_horizon=12)
