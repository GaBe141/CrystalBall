import numpy as np
import pandas as pd

from src.evaluation import rolling_origin_cv
from src.model_registry import ModelRegistry


def test_rolling_origin_cv_basics():
    # Create a simple increasing series with noise
    rng = np.random.default_rng(0)
    y = pd.Series(np.arange(100, dtype=float) + rng.normal(0, 0.1, size=100))

    def fit_fn(train):
        # Fit a naive model that predicts last value
        return train.iloc[-1]

    def predict_fn(model, horizon):
        return [model] * horizon

    res = rolling_origin_cv(y, horizon=5, n_folds=10, fit_fn=fit_fn, predict_fn=predict_fn)
    assert "cv_mae" in res and "cv_rmse" in res
    assert res["cv_mae"] >= 0
    assert res["cv_rmse"] >= res["cv_mae"]


def test_model_registry_register_and_get():
    reg = ModelRegistry()

    def dummy_runner(x=1):
        return {"ok": True, "x": x}

    reg.register("dummy", family="test", runner=dummy_runner, optional_dep=None)
    entry = reg.get("dummy")
    assert entry["family"] == "test"
    assert callable(entry["runner"])
    out = entry["runner"](x=2)
    assert out["ok"] and out["x"] == 2
