"""Executable task families."""

from importlib import import_module
from types import ModuleType

games: ModuleType
physics: ModuleType

__all__ = ["games", "physics"]


def __getattr__(name: str) -> ModuleType:
    """Lazily import task subpackages."""

    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
