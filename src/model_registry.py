from __future__ import annotations

from typing import Any, Callable, Dict, Optional

RegistryEntry = Dict[str, Any]


class ModelRegistry:
    """Simple plugin-style registry for models.

    Each registered model entry is a dict with keys:
    - family: str (e.g., 'arima', 'ets', 'ml')
    - runner: Callable[..., dict]  # executes training/eval and returns results
    - optional_dep: Optional[str]  # pip package name if optional
    - meta: Optional[dict]
    """

    def __init__(self) -> None:
        self._reg: Dict[str, RegistryEntry] = {}

    def register(
        self,
        name: str,
        *,
        family: str,
        runner: Callable[..., dict],
        optional_dep: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> None:
        self._reg[name] = {
            "family": family,
            "runner": runner,
            "optional_dep": optional_dep,
            "meta": meta or {},
        }

    def get(self, name: str) -> RegistryEntry:
        return self._reg[name]

    def available(self, names: Optional[list[str]] = None) -> Dict[str, RegistryEntry]:
        if names:
            return {k: v for k, v in self._reg.items() if k in names}
        return dict(self._reg)


# Singleton-like default registry
REGISTRY = ModelRegistry()
"""Default registry instance for convenience."""

# Backward-compatible decorator-based API used by existing model modules
from dataclasses import dataclass
from typing import List


@dataclass
class ModelInfo:
    name: str
    fn: Callable
    tags: List[str]
    requires: Optional[List[str]] = None


_DECORATOR_REGISTRY: Dict[str, ModelInfo] = {}


def register_model(name: str, *, tags: List[str], requires: Optional[List[str]] = None):
    """Decorator to register a model function.

    Registers into a simple local registry for compatibility, and also mirrors the
    registration into the class-based REGISTRY with meta information.
    """

    def _decorator(fn: Callable):
        info = ModelInfo(name=name, fn=fn, tags=tags, requires=requires)
        _DECORATOR_REGISTRY[name] = info
        # Mirror into class-based registry
        REGISTRY.register(
            name,
            family=tags[0] if tags else "unknown",
            runner=lambda *args, **kwargs: fn(*args, **kwargs),
            optional_dep=(requires[0] if requires else None),
            meta={"tags": tags, "requires": requires or []},
        )
        return fn

    return _decorator


def get_model(name: str) -> Optional[ModelInfo]:
    return _DECORATOR_REGISTRY.get(name)


def list_models(tag: Optional[str] = None) -> List[str]:
    if tag is None:
        return list(_DECORATOR_REGISTRY.keys())
    return [k for k, v in _DECORATOR_REGISTRY.items() if tag in v.tags]
