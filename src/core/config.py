"""Configuration management for CrystalBall.

Loads paths and settings from config/config.yaml if present, otherwise
falls back to sensible defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional dependency


@dataclass
class Paths:
    raw_data_dir: str = os.path.join('data', 'raw')
    processed_dir: str = os.path.join('data', 'processed')
    visuals_dir: str = os.path.join('data', 'processed', 'visualizations')
    exports_dir: str = os.path.join('data', 'processed', 'exports')


@dataclass
class Settings:
    test_size_fraction: float = 0.2
    max_affinity_features: int = 5
    
    # API defaults
    api_timeout_seconds: float = 15.0
    api_max_retries: int = 3
    api_backoff_seconds: float = 0.5
    api_cache_ttl_seconds: int = 3600
    api_user_agent: str = "CrystalBall/1.0"
    
    # Model evaluation settings
    cv_folds: int = 5
    forecast_horizon_auto: bool = True
    min_series_length: int = 10
    
    # Performance and resource limits
    max_parallel_processes: int = 4
    memory_limit_mb: int = 2048
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = False
    log_file_path: str = "logs/crystalball.log"


@dataclass
class AppConfig:
    paths: Paths
    settings: Settings


def _load_yaml(path: str) -> dict[str, Any] | None:
    if yaml is None:
        return None
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def load_config() -> AppConfig:
    cfg = _load_yaml(os.path.join('config', 'config.yaml')) or {}

    paths_cfg = cfg.get('paths', {}) if isinstance(cfg, dict) else {}
    settings_cfg = cfg.get('settings', {}) if isinstance(cfg, dict) else {}

    paths = Paths(
        raw_data_dir=paths_cfg.get('raw_data_dir', Paths.raw_data_dir),
        processed_dir=paths_cfg.get('processed_dir', Paths.processed_dir),
        visuals_dir=paths_cfg.get('visuals_dir', Paths.visuals_dir),
        exports_dir=paths_cfg.get('exports_dir', Paths.exports_dir),
    )
    settings = Settings(
        test_size_fraction=float(
            settings_cfg.get('test_size_fraction', Settings.test_size_fraction)
        ),
        max_affinity_features=int(
            settings_cfg.get('max_affinity_features', Settings.max_affinity_features)
        ),
        api_timeout_seconds=float(
            settings_cfg.get('api_timeout_seconds', Settings.api_timeout_seconds)
        ),
        api_max_retries=int(settings_cfg.get('api_max_retries', Settings.api_max_retries)),
        api_backoff_seconds=float(
            settings_cfg.get('api_backoff_seconds', Settings.api_backoff_seconds)
        ),
        api_cache_ttl_seconds=int(
            settings_cfg.get('api_cache_ttl_seconds', Settings.api_cache_ttl_seconds)
        ),
        api_user_agent=str(settings_cfg.get('api_user_agent', Settings.api_user_agent)),
        cv_folds=int(settings_cfg.get('cv_folds', Settings.cv_folds)),
        forecast_horizon_auto=bool(
            settings_cfg.get('forecast_horizon_auto', Settings.forecast_horizon_auto)
        ),
        min_series_length=int(
            settings_cfg.get('min_series_length', Settings.min_series_length)
        ),
        max_parallel_processes=int(
            settings_cfg.get('max_parallel_processes', Settings.max_parallel_processes)
        ),
        memory_limit_mb=int(settings_cfg.get('memory_limit_mb', Settings.memory_limit_mb)),
        log_level=str(settings_cfg.get('logging', {}).get('level', Settings.log_level)),
        log_format=str(settings_cfg.get('logging', {}).get('format', Settings.log_format)),
        file_logging=bool(
            settings_cfg.get('logging', {}).get('file_logging', Settings.file_logging)
        ),
        log_file_path=str(
            settings_cfg.get('logging', {}).get('log_file_path', Settings.log_file_path)
        ),
    )
    # ensure directories exist
    os.makedirs(paths.processed_dir, exist_ok=True)
    os.makedirs(paths.visuals_dir, exist_ok=True)
    os.makedirs(paths.exports_dir, exist_ok=True)

    return AppConfig(paths=paths, settings=settings)
