"""Generate dummy rankings CSVs and images to test export robustness.

Creates a set of synthetic per-series ranking CSV files and matching PNG
images in a "visualizations" folder, mimicking the structure expected by
batch_export_reports().
"""
from __future__ import annotations

import os
import random
from collections.abc import Iterable
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_rankings_df(n_models: int = 5, with_weighted: bool = True) -> pd.DataFrame:
    models = [f"model_{i+1}" for i in range(n_models)]
    rng = np.random.default_rng(42)
    rmse = rng.uniform(0.1, 10.0, n_models)
    mape = rng.uniform(0.5, 50.0, n_models)
    mean_rank = np.argsort(np.argsort(rmse)).astype(float) + 1.0
    df = pd.DataFrame({
        'model': models,
        'rmse': rmse,
        'mape': mape,
        'mean_rank': mean_rank,
    })
    if with_weighted:
        # Lower rmse and mape should be better -> invert and normalize
        score = (1 / (1 + (rmse / rmse.max()))) * 0.6 + (1 / (1 + (mape / mape.max()))) * 0.4
        df['weighted_score'] = score
    return df


def _make_dummy_image(save_path: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    x = np.arange(20)
    y = np.cumsum(np.random.randn(20))
    plt.plot(x, y, label='series')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.close()


def generate_dummy_results(results_dir: str,
                           visuals_dir: str,
                           series_names: Iterable[str] | None = None,
                           n_models: int = 5) -> List[str]:
    """Generate dummy rankings and images for the given series.

    Returns the list of base_names created.
    """
    _ensure_dir(results_dir)
    _ensure_dir(visuals_dir)
    if series_names is None:
        series_names = ["lorem", "ipsum", "dolor", "sit", "amet"]
    created: List[str] = []
    for name in series_names:
        # alternate including weighted_score to exercise Excel chart fallback
        with_weighted = random.random() > 0.5
        df = _make_rankings_df(n_models=n_models, with_weighted=with_weighted)
        base = name
        csv_path = os.path.join(results_dir, f"{base}_rankings.csv")
        df.to_csv(csv_path, index=False)

        # Images the exporter can discover: rankings + a couple forecasts
        _make_dummy_image(os.path.join(visuals_dir, f"{base}_rankings.png"), f"Rankings - {name}")
        _make_dummy_image(os.path.join(visuals_dir, f"{base}_forecast_arima.png"), f"Forecast ARIMA - {name}")
        _make_dummy_image(os.path.join(visuals_dir, f"{base}_forecast_prophet.png"), f"Forecast Prophet - {name}")
        # Also drop an adherence-like image
        _make_dummy_image(os.path.join(visuals_dir, f"{base}_adherence_analysis.png"), f"Adherence - {name}")
        created.append(base)
    return created


def main() -> None:
    # Minimal CLI: writes into configured processed/visuals dirs by default
    from src.config import load_config
    cfg = load_config()
    generate_dummy_results(cfg.paths.processed_dir, cfg.paths.visuals_dir)
    print(f"Dummy results created under: {cfg.paths.processed_dir} and images in {cfg.paths.visuals_dir}")


if __name__ == "__main__":
    main()
