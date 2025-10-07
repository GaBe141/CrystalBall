"""API processing module for CrystalBall.

Provides a robust foundation to integrate external API data sources:
- ApiClient with retries/backoff, timeouts, and simple rate limiting
- File-based cache with TTL to avoid redundant calls
- Provider abstraction + registry for multiple APIs
- DataAdapter to normalize responses to a standard DataFrame shape
- A DummyProvider for local/deterministic testing (no network calls)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover - requests is in requirements
    requests = None  # type: ignore

from src.config import load_config
from src.logutil import get_logger

# ------------------------------ Config Types ------------------------------ #


@dataclass
class ApiConfig:
    base_url: Optional[str] = None
    timeout_seconds: float = 15.0
    max_retries: int = 3
    backoff_seconds: float = 0.5
    user_agent: str = "CrystalBall/1.0"
    calls_per_second: Optional[float] = None  # simple client-side rate limit
    cache_ttl_seconds: int = 3600
    cache_dir: str = os.path.join('data', 'processed', 'api_cache')


# ------------------------------ File Cache ------------------------------- #


class FileCache:
    def __init__(self, cache_dir: str, ttl_seconds: int, logger: logging.Logger) -> None:
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        self.logger = logger
        os.makedirs(cache_dir, exist_ok=True)

    def _key_path(self, key: str) -> str:
        h = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._key_path(key)
        if not os.path.exists(path):
            return None
        # expired?
        if self.ttl > 0 and (time.time() - os.path.getmtime(path) > self.ttl):
            try:
                os.remove(path)
            except Exception:
                pass
            return None
        try:
            with open(path, encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return None

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._key_path(key)
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)
        except Exception as e:
            self.logger.debug(f"Cache write failed: {e}")


# ------------------------------- ApiClient -------------------------------- #


class ApiClient:
    def __init__(self, cfg: ApiConfig, logger: Optional[logging.Logger] = None) -> None:
        self.cfg = cfg
        self.logger = logger or get_logger("crystalball.api")
        self.cache = FileCache(cfg.cache_dir, cfg.cache_ttl_seconds, self.logger)
        self._last_call_ts: float = 0.0
        self._session = requests.Session() if requests else None
        if self._session:
            self._session.headers.update({'User-Agent': cfg.user_agent})

    def _respect_rate_limit(self) -> None:
        if not self.cfg.calls_per_second:
            return
        min_interval = 1.0 / float(self.cfg.calls_per_second)
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_ts = time.time()

    def request(self, method: str, path: str | None = None, *, url: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None,
                json_body: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None,
                use_cache: bool = True) -> Dict[str, Any]:
        """Perform an HTTP request with retry/backoff and optional caching.

        If url is provided, it is used as-is; otherwise base_url + path is used.
        """
        if url is None:
            if not self.cfg.base_url or path is None:
                raise ValueError("Either url or (base_url and path) must be provided")
            url = self.cfg.base_url.rstrip('/') + '/' + path.lstrip('/')

        # build cache key
        key_obj = {
            'm': method.upper(),
            'u': url,
            'p': params or {},
            'b': json_body or {},
            'h': headers or {},
        }
        cache_key = json.dumps(key_obj, sort_keys=True)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # no actual network in tests if requests missing
        if not self._session:
            raise RuntimeError("Requests not available in environment")

        # retry loop
        attempt = 0
        while True:
            attempt += 1
            try:
                self._respect_rate_limit()
                resp = self._session.request(
                    method=method.upper(), url=url, params=params, json=json_body,
                    headers=headers, timeout=self.cfg.timeout_seconds,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:256]}")
                data = resp.json()
                if use_cache:
                    self.cache.set(cache_key, data)
                return data
            except Exception as e:
                if attempt >= self.cfg.max_retries:
                    self.logger.error(f"API request failed after {attempt} attempts: {e}")
                    raise
                sleep_s = self.cfg.backoff_seconds * attempt
                self.logger.warning(f"API request error (attempt {attempt}): {e} -> sleeping {sleep_s:.2f}s")
                time.sleep(sleep_s)


# ------------------------------- Adapters --------------------------------- #


class DataAdapter:
    """Adapter to normalize API payload into a tidy DataFrame.

    The canonical shape is: columns = ['date', 'value', 'series', 'source']
    """

    def normalize(self, payload: Dict[str, Any], *, series: str, source: str) -> pd.DataFrame:  # pragma: no cover - to be overridden
        raise NotImplementedError


class TimeseriesAdapter(DataAdapter):
    """Default adapter for [ {date: ..., value: ...}, ... ] payloads."""

    def normalize(self, payload: Dict[str, Any], *, series: str, source: str) -> pd.DataFrame:
        records = payload if isinstance(payload, list) else payload.get('data', [])
        df = pd.DataFrame(records)
        # Try common date and value keys
        date_col = next((c for c in df.columns if str(c).lower() in {'date', 'dt', 'time', 'timestamp'}), None)
        value_col = next((c for c in df.columns if str(c).lower() in {'value', 'val', 'y'}), None)
        if date_col is None or value_col is None:
            raise ValueError("Payload missing recognizable date/value columns")
        out = pd.DataFrame({
            'date': pd.to_datetime(df[date_col], errors='coerce'),
            'value': pd.to_numeric(df[value_col], errors='coerce'),
            'series': series,
            'source': source,
        }).dropna(subset=['date', 'value'])
        out.sort_values('date', inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out


# ------------------------------- Providers -------------------------------- #


class Provider:
    def __init__(self, name: str, client: ApiClient, adapter: DataAdapter, source_label: Optional[str] = None) -> None:
        self.name = name
        self.client = client
        self.adapter = adapter
        self.source_label = source_label or name

    def fetch_series(self, series: str, *, endpoint: Optional[str] = None,
                     params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Fetch a series from the API and normalize it to a DataFrame."""
        payload = self.client.request('GET', path=endpoint or f"series/{series}", params=params)
        return self.adapter.normalize(payload, series=series, source=self.source_label)


class DummyProvider(Provider):
    """A deterministic, offline provider for testing purposes.

    Generates synthetic time series without any network calls.
    """

    def __init__(self, name: str = 'dummy') -> None:
        # Build a fake ApiClient that will never be used to call network
        cfg = load_config()
        api_cfg = ApiConfig(
            base_url=None,
            timeout_seconds=cfg.settings.api_timeout_seconds,
            max_retries=cfg.settings.api_max_retries,
            backoff_seconds=cfg.settings.api_backoff_seconds,
            user_agent=cfg.settings.api_user_agent,
            calls_per_second=None,
            cache_ttl_seconds=cfg.settings.api_cache_ttl_seconds,
            cache_dir=os.path.join(cfg.paths.processed_dir, 'api_cache'),
        )
        # Create a client but never call request(); we'll override fetch
        super().__init__(name=name, client=ApiClient(api_cfg), adapter=TimeseriesAdapter(), source_label='dummy')

    def fetch_series(self, series: str, *, endpoint: Optional[str] = None,
                     params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        # produce 100 daily points deterministically per series
        rng = pd.Series(range(100))
        base = abs(int(hashlib.sha256(series.encode('utf-8')).hexdigest(), 16)) % 1000
        vals = (rng * 0.5 + (base % 7)).astype(float)  # simple trend with offset
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(rng), freq='D')
        out = pd.DataFrame({'date': dates, 'value': vals, 'series': series, 'source': self.source_label})
        return out


# ------------------------------- Registry --------------------------------- #


class APIRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, Provider] = {}

    def register(self, name: str, provider: Provider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> Provider:
        if name not in self._providers:
            raise KeyError(f"Provider not registered: {name}")
        return self._providers[name]

    def names(self) -> list[str]:
        return sorted(self._providers.keys())


registry = APIRegistry()


def init_default_providers() -> None:
    """Register built-in providers (currently only dummy)."""
    if 'dummy' not in registry.names():
        registry.register('dummy', DummyProvider())


def fetch_and_write_csv(provider_name: str, series: str, output_dir: str) -> str:
    """Helper to fetch a series and write to CSV for later processing."""
    init_default_providers()
    provider = registry.get(provider_name)
    df = provider.fetch_series(series)
    os.makedirs(output_dir, exist_ok=True)
    safe = series.replace(' ', '_')
    path = os.path.join(output_dir, f"api_{provider_name}_{safe}.csv")
    df.to_csv(path, index=False)
    return path
