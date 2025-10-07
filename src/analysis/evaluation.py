from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd


def rolling_origin_cv(
    y: pd.Series,
    horizon: int,
    n_folds: int,
    fit_fn: Callable[[pd.Series], Any],
    predict_fn: Callable[[Any, int], pd.Series | np.ndarray | list],
) -> Dict[str, float]:
    """Rolling-origin cross-validation for univariate time series.

    Contract:
    - Inputs: y (pd.Series, indexed, may contain NaN), horizon (>0), n_folds (>0),
      fit_fn(train_series) -> model, predict_fn(model, horizon) -> sequence-like of length >= horizon.
    - Output: dict with 'cv_mae' and 'cv_rmse' across folds.
    - Errors: raises AssertionError for invalid sizes.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    y = y.dropna()
    assert horizon > 0 and n_folds > 0, "horizon and n_folds must be positive"
    # Ensure enough observations
    min_len = horizon * (n_folds + 1)
    assert len(y) >= min_len, f"Series too short: need >= {min_len}, got {len(y)}"

    maes: list[float] = []
    rmses: list[float] = []
    # We roll from earlier to later; each fold adds one horizon-sized block to training
    for i in range(n_folds):
        split = len(y) - (n_folds - i) * horizon
        train = y.iloc[:split]
        test = y.iloc[split : split + horizon]
        model = fit_fn(train)
        fcst = predict_fn(model, horizon)
        # Normalize forecast to Series aligned to test length
        if not isinstance(fcst, pd.Series):
            fcst = pd.Series(fcst)
        fcst = fcst.iloc[: len(test)]

        err = test.to_numpy() - fcst.to_numpy()
        maes.append(float(np.mean(np.abs(err))))
        rmses.append(float(np.sqrt(np.mean(err**2))))

    return {"cv_mae": float(np.mean(maes)), "cv_rmse": float(np.mean(rmses))}


def merge_cv_into_rankings(rankings: pd.DataFrame, cv_metrics: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Merge per-model CV metrics (cv_mae, cv_rmse) into rankings DataFrame.

    cv_metrics: dict model -> (cv_mae, cv_rmse)
    """
    r = rankings.copy()
    def _get(m: Any, i: int) -> float:
        try:
            return float(cv_metrics.get(str(m), (np.nan, np.nan))[i])
        except Exception:
            return float("nan")
    r["cv_mae"] = r["model"].map(lambda m: _get(m, 0))
    r["cv_rmse"] = r["model"].map(lambda m: _get(m, 1))
    return r


def compute_model_cv_metrics(y: pd.Series, horizon: int, n_folds: int, models_to_eval: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Compute per-model rolling-origin CV metrics using lightweight closures.

    models_to_eval: map of model name -> callable that given a train series returns a forecast of length horizon.
    This avoids deep coupling to the full pipeline while still providing useful CV signal.
    """
    y = pd.Series(y).dropna()
    out: Dict[str, Tuple[float, float]] = {}
    if horizon <= 0 or n_folds <= 0 or len(y) < horizon * (n_folds + 1):
        return out

    for name, forecaster in models_to_eval.items():
        try:
            def fit_fn(train: pd.Series):
                # Stateless: forecaster consumes train and returns a tiny object with forecast method
                return train

            def predict_fn(model_series: pd.Series, h: int):
                return forecaster(model_series, h)

            res = rolling_origin_cv(y, horizon=horizon, n_folds=n_folds, fit_fn=fit_fn, predict_fn=predict_fn)
            out[name] = (res.get("cv_mae", float("nan")), res.get("cv_rmse", float("nan")))
        except Exception:
            continue
    return out
