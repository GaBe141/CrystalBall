from __future__ import annotations

import logging
import numpy as np
import pandas as pd


def _cv_weights_from_rmse(cv_rmse: pd.Series, floor: float = 1e-9) -> pd.Series:
    x = pd.Series(cv_rmse).astype(float).replace([np.inf, -np.inf], np.nan)
    if not x.notna().any():
        # fallback to equal weights
        return pd.Series(1.0, index=x.index) / max(1, len(x))
    x = x.fillna(x.max()).clip(lower=floor)
    inv = 1.0 / x
    return inv / inv.sum()


def cv_weighted_ensemble(
    forecasts: dict[str, pd.Series],
    cv_rmse: pd.Series,
    name: str = "ensemble_cv_weighted",
) -> pd.Series:
    if not forecasts:
        raise ValueError("No forecasts provided")
    # Align on common index
    common_idx = None
    for s in forecasts.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    aligned = {k: v.reindex(common_idx) for k, v in forecasts.items()}
    # restrict to models with cv_rmse
    avail = {m: s for m, s in aligned.items() if m in cv_rmse.index and s.notna().any()}
    if not avail:
        raise ValueError("No overlapping models with cv_rmse for ensemble")
    weights = _cv_weights_from_rmse(cv_rmse.loc[list(avail.keys())])
    # weighted aggregation
    wsum = None
    wtot = None
    for m, series in avail.items():
        w = float(weights[m])
        x = series.astype(float)
        if wsum is None:
            wsum = w * x
            wtot = w * x.notna()
        else:
            wsum = wsum.add(w * x, fill_value=0.0)
            wtot = wtot.add(w * x.notna(), fill_value=0.0)
    out = wsum.divide(wtot.replace(0, np.nan))
    out.name = name
    return out


def _collect_forecasts_and_errors(
    models: dict[str, dict],
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    forecasts: dict[str, pd.Series] = {}
    errors: dict[str, float] = {}
    for k, r in models.items():
        if not isinstance(r, dict):
            continue
        fc = r.get("forecast")
        if fc is None:
            continue
        try:
            s = pd.Series(fc) if not isinstance(fc, pd.Series) else fc
            forecasts[k] = s.astype(float)
        except Exception as exc:
            logging.getLogger(__name__).debug("Failed to coerce forecast for %s: %s", k, exc)
            continue
        err = r.get("rmse") or r.get("mae")
        if err is not None and np.isfinite(err):
            errors[k] = float(err)
    return forecasts, errors


def _align_forecasts(forecasts: dict[str, pd.Series]) -> tuple[dict[str, pd.Series], pd.Index]:
    common_idx: pd.Index | None = None
    for s in forecasts.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    if common_idx is None or len(common_idx) == 0:
        # fallback to union, averaging with NaN-aware weights
        idxs = list({i for s in forecasts.values() for i in s.index})
        common_idx = pd.Index(sorted(idxs))
    aligned = {k: v.reindex(common_idx) for k, v in forecasts.items()}
    return aligned, common_idx


def _compute_weights(keys: list[str], errors: dict[str, float], power: float) -> dict[str, float]:
    if errors:
        eps = 1e-9
        acc = {k: (1.0 / (errors[k] + eps)) ** float(power) for k in errors}
        total = sum(acc.values()) or 1.0
        weights = {k: (acc.get(k, 0.0) / total) for k in keys}
        zero_keys = [k for k, w in weights.items() if w == 0.0]
        if zero_keys:
            rem = max(0.0, 1.0 - sum(weights.values()))
            if rem > 0 and len(zero_keys) > 0:
                add = rem / len(zero_keys)
                for k in zero_keys:
                    weights[k] = add
    else:
        n = max(1, len(keys))
        weights = {k: 1.0 / n for k in keys}
    return weights


def kwartz_ensemble(
    models: dict[str, dict],
    actual: pd.Series | None = None,
    power: float = 1.0,
    name: str = "kwartz",
) -> dict | None:
    """Kwartz: accuracy-weighted chimera ensemble over provided model results.

    - Uses inverse-error weights: w_i ∝ (1 / error_i)^power.
    - Prefers RMSE; falls back to MAE; if no metrics, uses equal weights.
    - Aligns forecasts on common index; averages with weights ignoring NaNs.

    Returns: dict with keys forecast, weights, mae, rmse – or None if no forecasts.
    """
    forecasts, errors = _collect_forecasts_and_errors(models)
    if not forecasts:
        return None

    aligned, _ = _align_forecasts(forecasts)
    weights = _compute_weights(list(aligned.keys()), errors, power)

    # Weighted aggregation ignoring NaNs
    wsum = None
    wcnt = None
    for k, s in aligned.items():
        w = float(weights.get(k, 0.0))
        x = s.astype(float)
        if wsum is None:
            wsum = w * x
            wcnt = w * x.notna()
        else:
            wsum = wsum.add(w * x, fill_value=0.0)
            wcnt = wcnt.add(w * x.notna(), fill_value=0.0)
    combined = wsum.divide(wcnt.replace(0, np.nan))
    combined.name = name

    # Metrics vs actual if provided
    mae: float | None = None
    rmse: float | None = None
    if actual is not None and len(actual) > 0:
        try:
            a = actual.reindex(combined.index).astype(float)
            yhat = combined.values
            mask = np.isfinite(a.values) & np.isfinite(yhat)
            if mask.any():
                diff = a.values[mask] - yhat[mask]
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff**2)))
        except Exception as exc:
            logging.getLogger(__name__).debug("Failed to compute Kwartz metrics: %s", exc)

    return {
        "forecast": combined,
        "weights": {k: float(v) for k, v in weights.items()},
        "mae": mae,
        "rmse": rmse,
    }
