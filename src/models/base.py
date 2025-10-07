from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import pandas as pd


class ModelRunner(Protocol):
    def __call__(self, series: pd.Series, *, test_size: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
        ...


@dataclass
class ModelSpec:
    name: str
    tags: list[str]
    runner: ModelRunner
    requires: Optional[list[str]] = None


def ensure_series_clean(y: pd.Series) -> pd.Series:
    s = y.copy()
    s = s.dropna()
    # enforce float and monotonic index when possible
    try:
        s = s.astype(float)
    except Exception:
        pass
    try:
        s = s.sort_index()
        if getattr(s.index, "duplicated", None) is not None and s.index.duplicated().any():
            s = s[~s.index.duplicated(keep="last")]
    except Exception:
        pass
    return s
