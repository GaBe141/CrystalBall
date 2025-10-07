from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # type: ignore
    from statsmodels.tsa.seasonal import STL  # type: ignore
    from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore
except Exception:  # pragma: no cover - import errors handled at runtime
    adfuller = None
    kpss = None
    acorr_ljungbox = None
    het_arch = None
    STL = None

try:
    import ruptures as rpt  # optional
except Exception:  # pragma: no cover - optional
    rpt = None


def _safe_series(y: pd.Series) -> pd.Series:
    s = pd.Series(y).dropna()
    try:
        s = s.astype(float)
    except Exception:
        pass
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def stationarity_tests(y: pd.Series) -> Dict[str, float]:
    y = _safe_series(y)
    out: Dict[str, float] = {}
    # ADF
    try:
        if adfuller is not None and len(y) >= 8:
            res = adfuller(y, autolag="AIC")
            out["adf_stat"] = float(res[0])
            out["adf_pvalue"] = float(res[1])
        else:
            out["adf_stat"] = np.nan
            out["adf_pvalue"] = np.nan
    except Exception:
        out["adf_stat"] = np.nan
        out["adf_pvalue"] = np.nan
    # KPSS (trend-stationary null)
    try:
        if kpss is not None and len(y) >= 8:
            stat, pval, *_ = kpss(y, regression="ct", nlags="auto")
            out["kpss_stat"] = float(stat)
            out["kpss_pvalue"] = float(pval)
        else:
            out["kpss_stat"] = np.nan
            out["kpss_pvalue"] = np.nan
    except Exception:
        out["kpss_stat"] = np.nan
        out["kpss_pvalue"] = np.nan
    return out


def serial_corr_tests(y: pd.Series, max_lag: Optional[int] = None) -> Dict[str, float]:
    y = _safe_series(y)
    if max_lag is None:
        max_lag = max(10, int(len(y) ** 0.5))
    try:
        if acorr_ljungbox is None or len(y) < 5:
            return {"ljungbox_stat": np.nan, "ljungbox_pvalue": np.nan}
        lb = acorr_ljungbox(y, lags=[min(max_lag, max(1, len(y) - 2))], return_df=True)
        return {
            "ljungbox_stat": float(lb["lb_stat"].iloc[-1]),
            "ljungbox_pvalue": float(lb["lb_pvalue"].iloc[-1]),
        }
    except Exception:
        return {"ljungbox_stat": np.nan, "ljungbox_pvalue": np.nan}


def heteroskedasticity_tests(residuals: Optional[pd.Series]) -> Dict[str, float]:
    if residuals is None:
        return {"arch_lm_stat": np.nan, "arch_lm_pvalue": np.nan}
    r = _safe_series(residuals)
    try:
        if het_arch is None or len(r) < 10:
            return {"arch_lm_stat": np.nan, "arch_lm_pvalue": np.nan}
        stat, pval, _, _ = het_arch(r, nlags=min(10, max(2, len(r) // 10)))
        return {"arch_lm_stat": float(stat), "arch_lm_pvalue": float(pval)}
    except Exception:
        return {"arch_lm_stat": np.nan, "arch_lm_pvalue": np.nan}


def stl_outliers(y: pd.Series, period: Optional[int] = None) -> Dict[str, float]:
    y = _safe_series(y)
    if STL is None or len(y) < 7:
        return {"stl_outlier_frac": np.nan, "stl_outlier_count": np.nan, "stl_resid_std": np.nan}
    try:
        # Default monthly seasonality if unknown
        p = period or 12
        stl = STL(y, period=p)
        res = stl.fit()
        resid = pd.Series(res.resid, index=y.index)
        q1, q3 = resid.quantile(0.25), resid.quantile(0.75)
        iqr = max(float(q3 - q1), 1e-9)
        high = resid > (q3 + 1.5 * iqr)
        low = resid < (q1 - 1.5 * iqr)
        mask = high | low
        return {
            "stl_outlier_frac": float(mask.mean()),
            "stl_outlier_count": int(mask.sum()),
            "stl_resid_std": float(resid.std()),
        }
    except Exception:
        return {"stl_outlier_frac": np.nan, "stl_outlier_count": np.nan, "stl_resid_std": np.nan}


def structural_breaks(y: pd.Series, max_breaks: int = 3) -> Dict[str, float]:
    y = _safe_series(y)
    if rpt is None or len(y) < 16:
        # Heuristic fallback: spikes in rolling mean change
        try:
            roll = y.rolling(window=max(5, len(y)//10), min_periods=5).mean().diff().abs()
            thresh = roll.quantile(0.95)
            count = int((roll > thresh).sum())
            return {"break_count": count, "break_method": "heuristic"}
        except Exception:
            return {"break_count": np.nan, "break_method": "none"}
    try:
        algo = rpt.Pelt(model="rbf", min_size=8).fit(y.values)
        idxs = algo.predict(pen=5.0)
        cps = [i for i in idxs if i < len(y)]
        cps = cps[:max_breaks]
        return {"break_count": int(len(cps)), "break_method": "ruptures"}
    except Exception:
        return {"break_count": np.nan, "break_method": "error"}


def bootstrap_ci(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_boot: int = 400,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    yt = _safe_series(y_true)
    yp = pd.Series(y_pred).reindex(yt.index)
    mask = yt.notna() & yp.notna()
    yt, yp = yt[mask].values, yp[mask].values
    if len(yt) < 8:
        return (np.nan, np.nan), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    maes, rmses = [], []
    n = len(yt)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        e = yt[idx] - yp[idx]
        maes.append(float(np.mean(np.abs(e))))
        rmses.append(float(np.sqrt(np.mean(e ** 2))))
    return (
        (float(np.quantile(rmses, 0.05)), float(np.quantile(rmses, 0.95))),
        (float(np.quantile(maes, 0.05)), float(np.quantile(maes, 0.95))),
    )


def conformal_intervals_safe(train_residuals: pd.Series, point_forecast: pd.Series, alpha: float = 0.1) -> pd.DataFrame:
    r = _safe_series(train_residuals)
    if len(r) < 5:
        return pd.DataFrame({"lower": np.nan, "upper": np.nan}, index=point_forecast.index)
    q = float(np.quantile(np.abs(r.values), 1 - alpha))
    lower = point_forecast.values - q
    upper = point_forecast.values + q
    return pd.DataFrame({"lower": lower, "upper": upper}, index=point_forecast.index)


def run_robust_diagnostics(
    y: pd.Series,
    residuals: Optional[pd.Series] = None,
    period_hint: Optional[int] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        out.update(stationarity_tests(y))
    except Exception:
        pass
    try:
        out.update(serial_corr_tests(y))
    except Exception:
        pass
    try:
        out.update(heteroskedasticity_tests(residuals))
    except Exception:
        pass
    try:
        out.update(stl_outliers(y, period=period_hint))
    except Exception:
        pass
    try:
        out.update(structural_breaks(y))
    except Exception:
        pass
    return out


def attach_bootstrap_cis_to_rankings(rankings_csv: str, y_full: pd.Series, models: Dict[str, Dict], test_size: int = 0) -> bool:
    """Augment per-series rankings CSV with bootstrap CIs for RMSE and MAE.

    Returns True if the file was updated.
    """
    try:
        if test_size <= 0:
            return False
        df = pd.read_csv(rankings_csv)
        y_true = y_full.iloc[-test_size:]
        rmse_low, rmse_high, mae_low, mae_high = [], [], [], []
        for _, row in df.iterrows():
            name = str(row.get("model"))
            res = models.get(name) or {}
            y_pred = res.get("forecast")
            if y_pred is None:
                rmse_low.append(np.nan); rmse_high.append(np.nan)
                mae_low.append(np.nan); mae_high.append(np.nan)
                continue
            try:
                # ensure alignment with y_true's index
                yp = pd.Series(getattr(y_pred, 'values', y_pred), index=getattr(y_pred, 'index', None))
                if len(yp) != len(y_true):
                    rmse_low.append(np.nan); rmse_high.append(np.nan)
                    mae_low.append(np.nan); mae_high.append(np.nan)
                    continue
                (rmse_l, rmse_h), (mae_l, mae_h) = bootstrap_ci(y_true, yp)
            except Exception:
                rmse_l = rmse_h = mae_l = mae_h = np.nan
            rmse_low.append(rmse_l); rmse_high.append(rmse_h)
            mae_low.append(mae_l); mae_high.append(mae_h)
        # add columns if not present
        if "rmse_ci_low" not in df.columns:
            df["rmse_ci_low"] = rmse_low
            df["rmse_ci_high"] = rmse_high
            df["mae_ci_low"] = mae_low
            df["mae_ci_high"] = mae_high
            df.to_csv(rankings_csv, index=False)
            return True
        return False
    except Exception:
        return False
