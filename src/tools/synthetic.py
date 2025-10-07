from __future__ import annotations
"""Compatibility shim to import synthetic helpers from top-level tools.

This assumes tests run with project root on sys.path, so absolute import works.
Re-export all public helpers directly from tools.synthetic without redefining
them here to avoid referencing private names like `_dt_index`.
"""

# Re-export public functions
from tools.synthetic import *  # type: ignore  # noqa: F401,F403

# Explicitly define __all__ to mirror the top-level tools.synthetic public API
# (anything not starting with an underscore)
try:
    import tools.synthetic as _syn  # type: ignore
    __all__ = [n for n in dir(_syn) if not n.startswith('_')]
except Exception:  # best-effort; tests import specific symbols
    __all__ = []
