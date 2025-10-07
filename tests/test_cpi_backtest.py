import numpy as np
import pandas as pd

from src.analysis.evaluation import (
    rolling_origin_cv,
    compute_model_cv_metrics,
    accuracy_to_weights,
    weights_to_matrix,
)


def _make_cpi_like_series(n: int = 120, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="ME")
    # level + small monthly seasonality + gentle upward trend + noise
    trend = 100 + 0.2 * np.arange(n)
    season = 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 0.3, size=n)
    y = trend + season + noise
    return pd.Series(y, index=idx)


def test_cpi_backtest_rank_and_weights():
    y = _make_cpi_like_series()
    h = 12
    folds = 4

    # Define simple forecasting closures: last value, mean, linear trend, ETS-like (fallback to SES)
    def fc_last(train: pd.Series, horizon: int):
        return [float(train.iloc[-1])] * horizon

    def fc_mean(train: pd.Series, horizon: int):
        m = float(train.mean())
        return [m] * horizon

    def fc_linear(train: pd.Series, horizon: int):
        x = np.arange(len(train))
        coef = np.polyfit(x, train.values.astype(float), 1)
        slope, intercept = float(coef[0]), float(coef[1])
        fut = intercept + slope * np.arange(len(train), len(train) + horizon)
        return fut

    def fc_ses(train: pd.Series, horizon: int):
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            m = SimpleExpSmoothing(train.astype(float)).fit(optimized=True)
            return m.forecast(horizon)
        except Exception:
            return fc_last(train, horizon)

    models = {
        "naive_last": fc_last,
        "mean": fc_mean,
        "linear": fc_linear,
        "ses": fc_ses,
    }

    cv = compute_model_cv_metrics(y, horizon=h, n_folds=folds, models_to_eval=models)
    # Should have metrics for all models
    assert set(cv.keys()) == set(models.keys())
    for k, (mae, rmse) in cv.items():
        assert np.isfinite(mae) and np.isfinite(rmse)
        assert rmse >= 0 and mae >= 0

    # Convert to weights - inverse RMSE
    wdf = accuracy_to_weights(cv, method="inv_rmse")
    assert not wdf.empty
    assert abs(wdf["weight"].sum() - 1.0) < 1e-6
    assert (wdf["weight"] >= 0).all()

    # Build weighting matrix
    W = weights_to_matrix(wdf)
    assert list(W.index) == list(W.columns)
    # Diagonal should match weights and sum to 1
    assert np.allclose(np.diag(W.values).sum(), 1.0, atol=1e-6)
    # Off-diagonals are zeros
    off_diag_sum = (W.values.sum() - np.diag(W.values).sum())
    assert abs(off_diag_sum) < 1e-9
