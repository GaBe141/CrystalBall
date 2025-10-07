# DARTS forecasting model wrapper
from darts import TimeSeries
from darts.models import ARIMA, NBEATS


def train_darts_model(df, time_col, target_col, model_type='NBEATS', forecast_horizon=12):
    series = TimeSeries.from_dataframe(df, time_col, target_col)
    if model_type == 'NBEATS':
        model = NBEATS()
    elif model_type == 'ARIMA':
        model = ARIMA()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.fit(series)
    forecast = model.predict(forecast_horizon)
    return forecast

# Example usage:
# forecast = train_darts_model(df, 'date', 'value', model_type='NBEATS', forecast_horizon=12)
