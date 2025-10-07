from __future__ import annotations

from typing import Dict

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


def cv_weighted_ensemble(forecasts: Dict[str, pd.Series], cv_rmse: pd.Series, name: str = "ensemble_cv_weighted") -> pd.Series:
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
