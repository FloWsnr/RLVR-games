"""Shared data structures for environment interaction."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TypeVar

ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class ParseResult(Generic[ActionT]):
    """Structured outcome of parsing a raw model action.

    Attributes
    ----------
    action : ActionT | None
        Canonical parsed action when parsing succeeds.
    error : str | None
        Rejection message when parsing fails for the current state.
    """

    action: ActionT | None
    error: str | None

    def __post_init__(self) -> None:
        """Validate that the parse result is either accepted or rejected.

        Raises
        ------
        ValueError
            If both `action` and `error` are populated or both are missing.
        """
        if (self.action is None) == (self.error is None):
            raise ValueError("ParseResult requires exactly one of action or error.")

    def require_action(self) -> ActionT:
        """Return the parsed action or fail for rejected results.

        Returns
        -------
        ActionT
            Parsed backend action.

        Raises
        ------
        ValueError
            If called on a rejected parse result.
        """
        if self.action is None:
            raise ValueError("Rejected parse results do not contain an action.")
        return self.action


@dataclass(slots=True)
class Observation:
    """Model-facing observation emitted by an environment.

    Attributes
    ----------
    text : str | None
        Optional text channel shown to the model for the current turn.
    image_paths : tuple[Path, ...]
        Zero or more filesystem paths to rendered images associated with the
        observation.
    metadata : dict[str, Any]
        Structured auxiliary data derived from canonical state. This is useful
        for evaluation, logging, or debugging, but it is not the authoritative
        game state.
    """

    text: str | None = None
    image_paths: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class InvalidActionMode(StrEnum):
    """Policy modes for handling invalid environment actions."""

    RAISE = "raise"
    PENALIZE_CONTINUE = "penalize-continue"
    PENALIZE_TRUNCATE = "penalize-truncate"


@dataclass(slots=True)
class InvalidActionPolicy:
    """Configuration describing how invalid actions should be handled.

    Attributes
    ----------
    mode : InvalidActionMode
        High-level invalid-action handling mode.
    penalty : float | None
        Reward assigned to rejected actions when the mode penalizes them. This
        must be `None` for `raise` mode and a concrete float otherwise.
    """

    mode: InvalidActionMode
    penalty: float | None

    def __post_init__(self) -> None:
        """Validate that the configured mode and penalty are coherent.

        Raises
        ------
        ValueError
            If the policy mode and penalty combination is inconsistent.
        """
        if self.mode == InvalidActionMode.RAISE and self.penalty is not None:
            raise ValueError("Raise mode does not accept an invalid-action penalty.")
        if self.mode != InvalidActionMode.RAISE and self.penalty is None:
            raise ValueError("Penalized invalid-action modes require a penalty.")


def _default_invalid_action_policy() -> InvalidActionPolicy:
    """Return the default fail-fast invalid-action policy."""
    return InvalidActionPolicy(mode=InvalidActionMode.RAISE, penalty=None)


@dataclass(slots=True)
class EpisodeConfig:
    """Runtime configuration controlling episode execution.

    Attributes
    ----------
    max_attempts : int | None
        Optional upper bound on the number of raw action attempts, including
        penalized invalid attempts, before the episode is truncated.
    max_transitions : int | None
        Optional upper bound on the number of accepted state transitions before
        the episode is truncated.
    invalid_action_policy : InvalidActionPolicy
        Policy describing whether invalid actions should raise immediately or
        produce explicit rejected attempts in the trajectory.
    metadata : dict[str, Any]
        Free-form configuration data for experiment bookkeeping or
        environment-specific options.
    """

    max_attempts: int | None = None
    max_transitions: int | None = None
    invalid_action_policy: InvalidActionPolicy = field(
        default_factory=_default_invalid_action_policy
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate episode limit configuration.

        Raises
        ------
        ValueError
            If any configured episode limit is not strictly positive.
        """
        if self.max_attempts is not None and self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive when provided.")
        if self.max_transitions is not None and self.max_transitions <= 0:
            raise ValueError("max_transitions must be positive when provided.")


@dataclass(slots=True)
class StepResult:
    """Result returned by a single environment transition.

    Attributes
    ----------
    observation : Observation
        Next model-facing observation after the action has been applied.
    reward : float
        Reward assigned to the transition by the configured reward function.
    accepted : bool
        Whether the raw action was accepted by the verifier and applied to the
        canonical state.
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
    accepted: bool
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
