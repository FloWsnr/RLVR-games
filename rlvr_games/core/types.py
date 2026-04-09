"""Shared data structures for environment interaction."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Observation:
    """Model-facing observation emitted by an environment.

    Attributes
    ----------
    text : str | None
        Optional text channel shown to the model for the current turn.
    image_paths : tuple[str, ...]
        Zero or more filesystem paths to rendered images associated with the
        observation.
    metadata : dict[str, Any]
        Structured auxiliary data derived from canonical state. This is useful
        for evaluation, logging, or debugging, but it is not the authoritative
        game state.
    """

    text: str | None = None
    image_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeConfig:
    """Runtime configuration controlling episode execution.

    Attributes
    ----------
    max_turns : int | None
        Optional upper bound on the number of agent turns before the episode is
        truncated.
    seed : int | None
        Optional default seed forwarded to the scenario when `reset()` is
        called without an explicit seed.
    metadata : dict[str, Any]
        Free-form configuration data for experiment bookkeeping or
        environment-specific options.
    """

    max_turns: int | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    """Result returned by a single environment transition.

    Attributes
    ----------
    observation : Observation
        Next model-facing observation after the action has been applied.
    reward : float
        Reward assigned to the transition by the configured reward function.
    terminated : bool
        Whether the environment reached a natural terminal state according to
        the game rules.
    truncated : bool
        Whether the episode ended for an external reason such as a turn limit.
    info : dict[str, Any]
        Transition metadata provided by the backend, including verifier-derived
        details such as move annotations or terminal outcome information.
    """

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
