from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .. import utils
from ..model_registry import register_model
from .base import ensure_series_clean


@register_model("arima_auto", tags=["arima", "classical"])
def run_arima(series: pd.Series, *, test_size: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
    y = ensure_series_clean(series)
    return utils.fit_arima_series(y, test_size=test_size, auto_order=True, exog=exog)


@register_model("ets_add", tags=["ets", "classical"])
def run_ets(series: pd.Series, *, test_size: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
    y = ensure_series_clean(series)
    return utils.fit_exponential_smoothing(y, test_size=test_size, trend="add")


@register_model("theta", tags=["theta", "classical"])
def run_theta(series: pd.Series, *, test_size: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
    y = ensure_series_clean(series)
    return utils.fit_theta_method(y, test_size=test_size)


@register_model("croston", tags=["intermittent", "classical"])
def run_croston(series: pd.Series, *, test_size: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
    y = ensure_series_clean(series)
    zeros_fraction = float((y == 0).mean()) if len(y) else 0.0
    if zeros_fraction <= 0.3:
        return {"error": "not_intermittent_enough", "zeros_fraction": zeros_fraction}
    return utils.fit_croston(y, n_forecast=max(1, int(len(y) * 0.1)), test_size=test_size)


@register_model("statsforecast_auto", tags=["statsforecast", "auto"], requires=["statsforecast"])
def run_statsforecast(series: pd.Series, *, test_size: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
    y = ensure_series_clean(series)
    return utils.fit_statsforecast_model(y, test_size=test_size, model_name="auto", exog=exog)


@register_model("mlforecast_lags", tags=["mlforecast", "ml"], requires=["mlforecast"])
def run_mlforecast(series: pd.Series, *, test_size: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
    y = ensure_series_clean(series)
    return utils.fit_mlforecast_model(y, covariates=exog, lags=kwargs.get("lags", [1, 12]), test_size=test_size)
