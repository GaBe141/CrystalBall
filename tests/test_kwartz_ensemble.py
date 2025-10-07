import numpy as np
import pandas as pd

from src.analysis.ensemble import kwartz_ensemble


def test_kwartz_weights_and_forecast_basic():
    idx = pd.RangeIndex(0, 3)
    models = {
        "m1": {"forecast": pd.Series([10.0, 10.0, 10.0], index=idx), "rmse": 1.0},
        "m2": {"forecast": pd.Series([0.0, 0.0, 0.0], index=idx), "rmse": 2.0},
        "m3": {"forecast": pd.Series([5.0, 5.0, 5.0], index=idx), "rmse": 1.0},
    }
    actual = pd.Series([8.0, 8.0, 8.0], index=idx)

    res = kwartz_ensemble(models, actual=actual, power=1.0, name="kwartz")
    assert res is not None
    weights = res["weights"]
    # inverse-error weights: [1, 0.5, 1] -> normalized [0.4, 0.2, 0.4]
    assert np.isclose(weights["m1"], 0.4, atol=1e-6)
    assert np.isclose(weights["m2"], 0.2, atol=1e-6)
    assert np.isclose(weights["m3"], 0.4, atol=1e-6)

    fc = res["forecast"]
    assert isinstance(fc, pd.Series)
    assert fc.name == "kwartz"

    # combined forecast should be 10*0.4 + 0*0.2 + 5*0.4 = 6 for all points
    assert np.allclose(fc.values, [6.0, 6.0, 6.0], atol=1e-6)

    # metrics vs actual 8 -> diff 2 -> mae=rmse=2
    assert np.isclose(res["mae"], 2.0, atol=1e-6)
    assert np.isclose(res["rmse"], 2.0, atol=1e-6)


ess = pd.Series

def test_kwartz_equal_weights_when_no_errors():
    idx = pd.RangeIndex(0, 2)
    models = {
        "a": {"forecast": pd.Series([1.0, 3.0], index=idx)},
        "b": {"forecast": pd.Series([3.0, 1.0], index=idx)},
    }

    res = kwartz_ensemble(models, actual=None, power=1.0, name="kwartz")
    assert res is not None
    weights = res["weights"]
    assert np.isclose(weights["a"], 0.5, atol=1e-6)
    assert np.isclose(weights["b"], 0.5, atol=1e-6)

    fc = res["forecast"]
    assert np.allclose(fc.values, [2.0, 2.0], atol=1e-6)
