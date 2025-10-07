from __future__ import annotations
"""Compatibility shim to import generate_dummy_results from top-level tools.

This assumes tests run with project root on sys.path, so absolute import works.
"""
from tools.generate_dummy_results import *  # type: ignore  # noqa: F401,F403
