"""Shared data structures for environment interaction."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

from PIL.Image import Image as PILImage

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
class RenderedImage:
    """One in-memory image payload attached to an observation.

    Attributes
    ----------
    key : str
        Stable renderer-defined identifier for the image content and render
        configuration. Callers can use this key for caching or persistence.
    image : PILImage
        In-memory raster image payload ready for multimodal training or local
        persistence.
    """

    key: str
    image: PILImage

    def copy(self) -> "RenderedImage":
        """Return a deep copy of the rendered image payload.

        Returns
        -------
        RenderedImage
            Copy whose underlying raster data does not alias the source image.
        """
        return RenderedImage(key=self.key, image=self.image.copy())


@dataclass(slots=True)
class Observation:
    """Model-facing observation emitted by an environment.

    Attributes
    ----------
    text : str | None
        Optional text channel shown to the model for the current turn.
    images : tuple[RenderedImage, ...]
        Zero or more in-memory images associated with the observation.
    metadata : dict[str, Any]
        Structured auxiliary data derived from canonical state that remains
        safe to expose alongside the observation. This is useful for
        evaluation or logging, but it is not the authoritative game state.
    """

    text: str | None = None
    images: tuple[RenderedImage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AutoAction(Generic[ActionT]):
    """One internal action selected by an auto-advance policy.

    Attributes
    ----------
    source : str
        Structured label describing who produced the action, for example
        ``"opponent"`` or ``"chance"``.
    raw_action : str
        Serialized form of the action for logging and trajectory recording.
    action : ActionT
        Parsed backend action ready to be applied.
    """

    source: str
    raw_action: str
    action: ActionT

    def __post_init__(self) -> None:
        """Validate that the selected action can be recorded coherently.

        Raises
        ------
        ValueError
            If `source` or `raw_action` is empty.
        """
        if not self.source:
            raise ValueError("AutoAction source must be non-empty.")
        if not self.raw_action:
            raise ValueError("AutoAction raw_action must be non-empty.")


@dataclass(slots=True, frozen=True)
class EpisodeBoundary:
    """Explicit episode boundary supplied by an auto-advance policy.

    Attributes
    ----------
    terminated : bool
        Whether the episode should end as a natural task completion.
    truncated : bool
        Whether the episode should end due to an external cutoff.
    info : dict[str, Any]
        Additional metadata describing why the boundary was reached.
    """

    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that exactly one episode boundary flag is set.

        Raises
        ------
        ValueError
            If both boundary flags are set or both are unset.
        """
        if self.terminated == self.truncated:
            raise ValueError(
                "EpisodeBoundary requires exactly one of terminated or truncated."
            )


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
    """Result returned by one environment step.

    Attributes
    ----------
    observation : Observation
        Next model-facing observation after the agent action and any internal
        auto-advanced actions have been applied.
    reward : float
        Reward assigned to the step by the configured reward function.
    accepted : bool
        Whether the raw action was accepted by the verifier and applied to the
        canonical state. Accepted steps may still include additional internal
        auto-advanced transitions after the agent action.
    terminated : bool
        Whether the environment reached a natural terminal state according to
        the game rules.
    truncated : bool
        Whether the episode ended for an external reason such as a turn limit.
    info : dict[str, Any]
        Public-safe step metadata including verifier-derived details for
        accepted transitions and any internal auto-advanced actions.
    """

    observation: Observation
    reward: float
    accepted: bool
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
