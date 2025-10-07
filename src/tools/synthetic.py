from __future__ import annotations
"""Compatibility shim to import synthetic helpers from top-level tools.

This assumes tests run with project root on sys.path, so absolute import works.
"""
from tools.synthetic import *  # type: ignore  # noqa: F401,F403


def series_trend_seasonal(n: int = 240, slope: float = 0.05, amp: float = 2.0, noise: float = 0.5, seed: int = 123) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = slope * t
    seasonal = amp * np.sin(2 * np.pi * t / 12)
    eps = rng.normal(scale=noise, size=n)
    y = trend + seasonal + eps
    return pd.Series(y, index=_dt_index(n))


def series_with_outliers(n: int = 240, n_outliers: int = 6, magnitude: float = 6.0, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = series_trend_seasonal(n=n, slope=0.02, amp=1.5, noise=0.7, seed=seed)
    idxs = rng.choice(np.arange(n // 6, n - n // 6), size=n_outliers, replace=False)
    s = base.copy()
    s.iloc[idxs] += rng.choice([-1, 1], size=n_outliers) * magnitude
    return s


def series_structural_breaks(n: int = 300, n_breaks: int = 2, jump: float = 5.0, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    blocks = n_breaks + 1
    seg_len = n // blocks
    y = []
    level = 0.0
    for b in range(blocks):
        level += (jump if b > 0 else 0.0) * rng.choice([-1, 1])
        seg = level + rng.normal(scale=1.0, size=seg_len)
        y.append(seg)
    y = np.concatenate(y)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)), mode='edge')
    return pd.Series(y, index=_dt_index(n))


def series_heteroskedastic(n: int = 300, omega: float = 0.2, alpha: float = 0.7, seed: int = 1234) -> pd.Series:
    """Simple ARCH(1)-like variance process to induce heteroskedasticity."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha) if alpha < 1 else omega
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t-1] ** 2)
        eps[t] = np.sqrt(max(sigma2[t], 1e-6)) * z[t]
    return pd.Series(eps, index=_dt_index(n))


def series_intermittent(n: int = 240, p_event: float = 0.05, mag_low: float = 1.0, mag_high: float = 5.0, seed: int = 321) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    events = rng.random(size=n) < p_event
    y[events] = rng.uniform(low=mag_low, high=mag_high, size=events.sum())
    return pd.Series(y, index=_dt_index(n))
