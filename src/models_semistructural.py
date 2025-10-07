"""Semi-structural hybrid and ensemble models.

This module provides pragmatic, lightweight versions of:
- Semi-structural gap-trend hybrid (policy channels as exogenous inputs)
- ML-enhanced state-space baseline (optional)
- Suite-of-models combiner (stacking/weighted ensemble)

These are designed to run with public Python stacks and avoid heavy research
dependencies while providing value and extensibility hooks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model_registry import register_model


@register_model("semi_structural_gap_trend", tags=["structural", "hybrid"])  # FINEX-inspired
def fit_semi_structural_gap_trend(series: pd.Series,
                                  test_size: int,
                                  exog: Optional[pd.DataFrame] = None,
                                  policy_cols: Optional[List[str]] = None) -> Dict:
    """A pragmatic semi-structural hybrid: trend + cycle + policy channels.

    - Decompose series into trend (HP-like via STL trend) and cycle (resid)
    - Model cycle via AR with policy/external channels as exogenous inputs
    - Recompose forecast = trend_forecast + cycle_forecast
    """
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Work on a safe copy with a clean index to avoid duplicate-label reindex errors
    y = series.astype(float).copy()
    try:
        y = y.sort_index()
        if y.index.duplicated().any():
            y = y[~y.index.duplicated(keep='last')]
    except Exception:
        pass
    # Guard: empty series
    if y is None or len(y) == 0:
        return {'forecast': None, 'fitted': None, 'mae': np.nan, 'rmse': np.nan, 'components': {}}
    # Ensure test_size is sensible
    ts = int(max(1, test_size))
    ts = min(ts, len(y))
    # Decomposition
    # Infer a reasonable seasonal period; if datetime-like and freq can be inferred, map to common periods, else default 12
    period = 12
    if isinstance(y.index, pd.DatetimeIndex):
        try:
            freq = pd.infer_freq(y.index)
        except Exception:
            freq = None
        if freq:
            fl = str(freq).lower()
            if fl.startswith('m'):
                period = 12
            elif fl.startswith('q'):
                period = 4
            elif fl.startswith('w'):
                period = 52
            elif fl.startswith('d'):
                period = 7
            elif fl.startswith('a') or fl.startswith('y'):
                period = 1
    period = max(5, min(24, int(period)))
    try:
        stl = STL(y, robust=True, period=period)
        stl_res = stl.fit()
        trend = pd.Series(stl_res.trend, index=y.index)
        cycle = y - trend
    except Exception:
        # Fallback: simple rolling mean for trend
        trend = y.rolling(window=max(5, int(len(y)*0.1)), min_periods=3).mean().bfill()
        cycle = y - trend

    # Prepare exogenous policy channels for cycle model
    X = None
    if exog is not None and not exog.empty:
        if policy_cols is None:
            policy_cols = [c for c in exog.columns if any(k in c.lower() for k in ["fx", "rate", "fiscal", "bal", "flow", "debt", "ca"])]
            if not policy_cols:
                policy_cols = exog.columns.tolist()[:5]
        X = exog[policy_cols].reindex(y.index).astype(float)

    # Fit a low-order ARIMAX on cycle
    if ts >= len(cycle):
        train_cycle = cycle.iloc[: max(1, len(cycle) - 1)]
    else:
        train_cycle = cycle.iloc[:-ts]
    exog_train = X.iloc[:-ts] if X is not None and len(X) >= ts else None
    exog_test = X.iloc[-ts:] if X is not None and len(X) >= ts else None
    try:
        cyc_mod = SARIMAX(train_cycle, order=(1, 0, 1), trend='n', exog=exog_train, enforce_stationarity=False, enforce_invertibility=False)
        cyc_res = cyc_mod.fit(disp=False)
    except Exception:
        cyc_res = None

    # Forecast trend with flat or simple continuation
    idx_fc = y.index[-ts:] if ts <= len(y) else y.index
    last_val = trend.iloc[-1] if len(trend) else (y.iloc[-1] if len(y) else 0.0)
    trend_forecast = pd.Series([last_val] * len(idx_fc), index=idx_fc)
    # Optionally extrapolate trend with last slope if enough points
    if len(trend) >= 6:
        slope = trend.diff().iloc[-5:].mean()
        trend_forecast = pd.Series([last_val + slope * (i+1) for i in range(len(idx_fc))], index=idx_fc)

    # Forecast cycle
    if cyc_res is not None and len(idx_fc) > 0:
        try:
            cyc_fc = cyc_res.get_forecast(steps=len(idx_fc), exog=exog_test)
            cycle_forecast = pd.Series(cyc_fc.predicted_mean, index=idx_fc)
        except Exception:
            cycle_forecast = pd.Series([cycle.iloc[-1] if len(cycle) else 0.0] * len(idx_fc), index=idx_fc)
        fitted_cycle = pd.Series(cyc_res.fittedvalues, index=train_cycle.index) if hasattr(cyc_res, 'fittedvalues') else None
    else:
        cycle_forecast = pd.Series([cycle.iloc[-1] if len(cycle) else 0.0] * len(idx_fc), index=idx_fc)
        fitted_cycle = None

    # Align components safely
    # Align components safely (unique indices only)
    try:
        if cycle_forecast.index.duplicated().any():
            cycle_forecast = cycle_forecast[~cycle_forecast.index.duplicated(keep='last')]
    except Exception:
        pass
    cycle_forecast = cycle_forecast.reindex(trend_forecast.index)
    if cycle_forecast.isna().any():
        cycle_forecast = cycle_forecast.ffill().bfill()
    forecast = (trend_forecast + cycle_forecast).reindex(idx_fc)
    fitted_index = y.index[:-ts] if ts < len(y) else y.index[:0]
    # Ensure components have unique indices before reindex
    try:
        if trend.index.duplicated().any():
            trend = trend[~trend.index.duplicated(keep='last')]
    except Exception:
        pass
    fitted_trend = trend.reindex(fitted_index)
    if isinstance(fitted_cycle, pd.Series) and len(fitted_index):
        try:
            if fitted_cycle.index.duplicated().any():
                fitted_cycle = fitted_cycle[~fitted_cycle.index.duplicated(keep='last')]
        except Exception:
            pass
        fitted_cycle = fitted_cycle.reindex(fitted_index)
    else:
        fitted_cycle = (pd.Series(0, index=fitted_index) if len(fitted_index) else None)
    fitted = (fitted_trend + fitted_cycle) if (len(fitted_index) and isinstance(fitted_cycle, pd.Series)) else (fitted_trend if len(fitted_index) else None)

    actuals = y.reindex(idx_fc)
    if len(actuals) and len(forecast):
        mae = float(np.mean(np.abs(actuals - forecast)))
        rmse = float(np.sqrt(np.mean((actuals - forecast) ** 2)))
    else:
        mae = np.nan
        rmse = np.nan
    return {
        'forecast': forecast,
        'fitted': fitted,
        'mae': mae,
        'rmse': rmse,
        'components': {
            'trend': trend, 'cycle': cycle,
            'trend_forecast': trend_forecast, 'cycle_forecast': cycle_forecast
        }
    }


@register_model("ml_enhanced_state_space", tags=["ml", "hybrid"], requires=["torch"])
def fit_ml_enhanced_state_space(series: pd.Series, test_size: int, exog: Optional[pd.DataFrame] = None) -> Dict:
    """Baseline state-space with learned residuals via light NN (optional if torch present).

    Implementation is conservative: we fit a SARIMAX and train a tiny MLP on residuals
    over lagged features to add a correction term.
    """
    try:
        import torch
        from torch import nn, optim
    except Exception:
        # If torch is not available, fallback to plain SARIMAX
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = series.astype(float)
        mod = SARIMAX(y.iloc[:-test_size], order=(1, 1, 1), trend='c')
        res = mod.fit(disp=False)
        fc = res.get_forecast(steps=test_size).predicted_mean
        actuals = y.iloc[-test_size:]
        mae = float(np.mean(np.abs(actuals - fc)))
        rmse = float(np.sqrt(np.mean((actuals - fc) ** 2)))
        return {'forecast': pd.Series(fc, index=actuals.index), 'fitted': res.fittedvalues, 'mae': mae, 'rmse': rmse}

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = series.astype(float)
    if y is None or len(y) == 0:
        return {'forecast': None, 'fitted': None, 'mae': np.nan, 'rmse': np.nan}
    ts = int(max(1, test_size))
    ts = min(ts, len(y))
    if len(y) <= max(10, ts + 2):
        # Not enough data for the hybrid; fallback to naive mean forecast
        n = min(ts, max(1, len(y)))
        hist = y.iloc[:-n] if len(y) > n else y
        mean_val = float(hist.mean()) if len(hist) else float(y.mean()) if len(y) else 0.0
        fc_vals = np.repeat(mean_val, n)
        idx = y.index[-n:]
        actuals = y.iloc[-n:]
        mae = float(np.mean(np.abs(actuals - fc_vals)))
        rmse = float(np.sqrt(np.mean((actuals - fc_vals) ** 2)))
        return {'forecast': pd.Series(fc_vals, index=idx), 'fitted': None, 'mae': mae, 'rmse': rmse}

    base = SARIMAX(y.iloc[:-ts], order=(1, 1, 1), trend='c').fit(disp=False)
    base_fc = base.get_forecast(steps=ts).predicted_mean
    resid = (y.iloc[:-ts] - base.fittedvalues).astype(float)

    # Build simple lagged features for residual learning
    L = 6
    X_list, r_list = [], []
    for t in range(L, len(resid)):
        X_list.append(resid.iloc[t-L:t].values)
        r_list.append(resid.iloc[t])
    if not X_list:
        fc = base_fc
        actuals = y.iloc[-ts:]
        mae = float(np.mean(np.abs(actuals - fc)))
        rmse = float(np.sqrt(np.mean((actuals - fc) ** 2)))
        return {'forecast': pd.Series(fc, index=actuals.index), 'fitted': base.fittedvalues, 'mae': mae, 'rmse': rmse}

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    r = torch.tensor(np.array(r_list), dtype=torch.float32).unsqueeze(1)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(L, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP()
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(200):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, r)
        loss.backward()
        opt.step()

    # Iterative forecast correction on base forecast residual pattern
    last_resid = resid.iloc[-L:].values
    corrections = []
    for _ in range(ts):
        x = torch.tensor(last_resid.reshape(1, -1), dtype=torch.float32)
        corr = float(model(x).item())
        corrections.append(corr)
        last_resid = np.roll(last_resid, -1)
        last_resid[-1] = corr

    fc = base_fc.values + np.array(corrections)
    actuals = y.iloc[-ts:]
    mae = float(np.mean(np.abs(actuals - fc)))
    rmse = float(np.sqrt(np.mean((actuals - fc) ** 2)))
    return {'forecast': pd.Series(fc, index=actuals.index), 'fitted': base.fittedvalues, 'mae': mae, 'rmse': rmse}


def suite_of_models_weighted(models: Dict[str, Dict]) -> Optional[Dict]:
    """Combine multiple model forecasts with inverse-RMSE weights (ECB-style suite)."""
    fcs = [(k, v['forecast'], v.get('rmse')) for k, v in models.items() if isinstance(v, dict) and v.get('forecast') is not None and v.get('rmse')]
    fcs = [(k, fc, rmse) for k, fc, rmse in fcs if rmse and rmse > 0]
    if not fcs:
        return None
    # Align by common index
    # Build a common index across forecasts using unique labels only
    def _unique_index(s: pd.Series) -> pd.Index:
        try:
            if s.index.duplicated().any():
                s = s[~s.index.duplicated(keep='last')]
        except Exception:
            pass
        return s.index

    common_idx = _unique_index(fcs[0][1])
    for _, fc, _ in fcs[1:]:
        common_idx = common_idx.intersection(_unique_index(fc))
    if len(common_idx) == 0:
        return None
    weights = np.array([1.0 / r for _, _, r in fcs], dtype=float)
    weights /= weights.sum()
    rows = []
    for _, fc, _ in fcs:
        try:
            if fc.index.duplicated().any():
                fc = fc[~fc.index.duplicated(keep='last')]
        except Exception:
            pass
        rows.append(fc.reindex(common_idx).values)
    mat = np.vstack(rows)
    combo = (weights.reshape(-1, 1) * mat).sum(axis=0)
    return {'forecast': pd.Series(combo, index=common_idx), 'weights': {k: float(w) for (k, _, _), w in zip(fcs, weights)}}
