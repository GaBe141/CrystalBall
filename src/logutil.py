from __future__ import annotations
"""Compatibility shim: expose core.logutil as src.logutil for legacy imports."""
from .core.logutil import *  # noqa: F401,F403
