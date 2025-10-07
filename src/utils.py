"""Compatibility shim: expose core.utils as src.utils for legacy imports.

This module intentionally re-exports everything from src.core.utils to avoid
code duplication and drift between two copies. If you need to modify utility
functions, edit src/core/utils.py only.
"""

from __future__ import annotations

from .core.utils import *  # noqa: F401,F403
