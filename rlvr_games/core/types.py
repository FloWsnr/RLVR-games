"""Shared data structures for environment interaction."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Observation:
    """Model-facing observation."""

    text: str | None = None
    image_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeConfig:
    """Runtime configuration for an episode."""

    max_turns: int | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    """Result returned by `step()`."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
