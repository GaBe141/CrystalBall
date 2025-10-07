"""Models package: registry-backed wrappers for forecasting models.

This package exposes lightweight wrappers that register into the central
model registry so they can be orchestrated uniformly (validation harness,
experiments, etc.). The main pipeline can keep using utils-based calls,
but new workflows can rely on these registered runners.
"""

from . import base  # re-export base helpers

__all__ = [
    "base",
]
