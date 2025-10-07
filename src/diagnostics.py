"""Runtime diagnostics and preflight checks for CrystalBall.

This module provides utilities to:
- Log Python/platform and key library versions.
- Capture warnings into logging.
- Install a global exception hook to ensure uncaught errors are logged.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
from typing import Any, Dict


def _safe_import_version(module_name: str) -> str:
    """Try to import a module and return its version string, otherwise a tag.

    Returns:
        A human-friendly version string, or one of:
        - "missing" if the module cannot be imported
        - "unknown" if imported but no __version__ is found
    """
    try:
        mod = __import__(module_name)
    except Exception:
        return "missing"
    # Common places for version
    for attr in ("__version__", "VERSION", "version"):
        try:
            v = getattr(mod, attr)
            return str(v)
        except Exception:
            continue
    return "unknown"


def collect_environment_versions() -> Dict[str, Any]:
    """Collect environment details and key library versions."""
    libs = [
        # core
        "pandas", "numpy", "scipy", "statsmodels", "matplotlib", "seaborn",
        # visualization / docs
        "plotly", "xlsxwriter", "pptx", "docx", "fpdf", "reportlab",
        # modeling
        "sklearn", "lightgbm", "prophet", "tbats", "darts",
        # schema / utils
        "pydantic",
    ]
    versions = {name: _safe_import_version(name) for name in libs}
    env = {
        "python": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "versions": versions,
    }
    return env


def log_environment(logger: logging.Logger, write_json_path: str | None = None) -> Dict[str, Any]:
    """Log environment and library versions; optionally write to JSON file."""
    env = collect_environment_versions()
    logger.info("Environment: %s", env["platform"])
    logger.info("Python: %s (%s)", env["python"], env["implementation"])
    logger.info("Executable: %s", env["executable"])
    logger.info("Working directory: %s", env["cwd"])
    for name, ver in env["versions"].items():
        level = logging.INFO if ver not in ("missing", "unknown") else logging.WARNING
        logger.log(level, "Library %s: %s", name, ver)
    if write_json_path:
        try:
            os.makedirs(os.path.dirname(write_json_path), exist_ok=True)
            with open(write_json_path, "w", encoding="utf-8") as f:
                json.dump(env, f, indent=2)
        except Exception:
            logger.exception("Failed writing environment report: %s", write_json_path)
    return env


def enable_warning_capture():
    """Route Python warnings to the logging system."""
    import logging as _logging
    _logging.captureWarnings(True)


def install_global_exception_logger(logger: logging.Logger):
    """Install a sys.excepthook that logs uncaught exceptions."""
    def _hook(exc_type, exc, tb):
        logger.error("Uncaught exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = _hook


def guarded_call(logger: logging.Logger, fn, *args, **kwargs):
    """Call a function and log exceptions without raising them.

    Returns the function's return value or None if an exception occurred.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("Error in %s", getattr(fn, "__name__", str(fn)))
        return None


def preflight_verify(logger: logging.Logger) -> Dict[str, Any]:
    """Verify presence of key libraries and log any critical gaps.

    Returns a dict with problems found, e.g. {"missing": [...], "unknown": [...]}.
    """
    env = collect_environment_versions()
    missing = [k for k, v in env["versions"].items() if v == "missing"]
    unknown = [k for k, v in env["versions"].items() if v == "unknown"]
    if missing:
        logger.error("Missing libraries: %s", ", ".join(missing))
    if unknown:
        logger.warning("Unknown versions: %s", ", ".join(unknown))
    return {"missing": missing, "unknown": unknown, "versions": env["versions"]}
