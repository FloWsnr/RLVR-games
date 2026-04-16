"""Trajectory recording types."""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from rlvr_games.core.types import Observation

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class RecordedTransition(Generic[ActionT]):
    """One accepted backend transition recorded within an env step.

    Attributes
    ----------
    source : str
        Structured label describing who produced the transition, for example
        ``"agent"`` or ``"opponent"``.
    raw_action : str
        Serialized action string applied for this transition.
    action : ActionT
        Parsed backend action that was applied.
    info : dict[str, Any]
        Public-safe verifier metadata emitted for this accepted transition.
    debug_info : dict[str, Any]
        Privileged transition trace intended for debugging and offline
        analysis. This may include canonical-state details that are not safe
        to expose to the agent.
    """

    source: str
    raw_action: str
    action: ActionT
    info: dict[str, Any] = field(default_factory=dict)
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppliedResetEvent(Generic[StateT]):
    """One authoritative reset-time event applied before the first agent turn.

    Attributes
    ----------
    source : str
        Structured label describing who produced the event, for example
        ``"chance"``, ``"dealer"``, or ``"opponent"``.
    label : str
        Structured serialized label describing the event, such as an opening
        deal or forced move.
    next_state : StateT
        Canonical state after the reset-time event has been applied.
    info : dict[str, Any]
        Public-safe metadata describing the applied event.
    """

    source: str
    label: str
    next_state: StateT
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the applied reset event can be recorded coherently."""
        if not self.source:
            raise ValueError("AppliedResetEvent source must be non-empty.")
        if not self.label:
            raise ValueError("AppliedResetEvent label must be non-empty.")


@dataclass(slots=True)
class RecordedResetEvent:
    """One reset-time event recorded before the first agent action.

    Attributes
    ----------
    source : str
        Structured label describing who produced the event, for example
        ``"chance"``, ``"dealer"``, or ``"opponent"``.
    label : str
        Structured serialized label describing the event.
    info : dict[str, Any]
        Public-safe metadata describing the event.
    debug_info : dict[str, Any]
        Privileged event trace intended for debugging and offline analysis.
        This may include canonical-state details that are not safe to expose
        to the agent.
    """

    source: str
    label: str
    info: dict[str, Any] = field(default_factory=dict)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the recorded reset event can be inspected coherently."""
        if not self.source:
            raise ValueError("RecordedResetEvent source must be non-empty.")
        if not self.label:
            raise ValueError("RecordedResetEvent label must be non-empty.")


@dataclass(slots=True)
class TrajectoryStep(Generic[ActionT]):
    """One recorded environment transition.

    Attributes
    ----------
    raw_action : str
        Raw action string produced by the model.
    action : ActionT | None
        Parsed backend action derived from `raw_action`, or `None` when the
        verifier rejected the attempt before a canonical action could be
        constructed.
    accepted : bool
        Whether the environment accepted the raw action and applied it.
    observation : Observation
        Observation returned after the transition.
    reward : float
        Reward assigned to the transition.
    terminated : bool
        Whether the transition ended the episode naturally.
    truncated : bool
        Whether the episode ended due to an external cutoff.
    info : dict[str, Any]
        Public-safe step metadata emitted by the environment.
    debug_info : dict[str, Any]
        Privileged step trace intended for debugging and offline analysis.
        This may include canonical-state details that are not safe to expose
        to the agent.
    transitions : tuple[RecordedTransition[ActionT], ...]
        Accepted backend transitions applied during this env step. The first
        transition is the agent action, followed by any internal auto-advanced
        transitions.
    """

    raw_action: str
    action: ActionT | None
    accepted: bool
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    debug_info: dict[str, Any] = field(default_factory=dict)
    transitions: tuple[RecordedTransition[ActionT], ...] = ()


@dataclass(slots=True)
class ScenarioReset(Generic[StateT]):
    """Structured result returned by scenario reset logic.

    Attributes
    ----------
    initial_state : StateT
        Canonical state from which the environment should continue reset
        resolution.
    reset_info : dict[str, Any]
        Public-safe scenario metadata for the reset.
    """

    initial_state: StateT
    reset_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrajectory(Generic[ActionT]):
    """Recorded data for a full environment episode.

    Attributes
    ----------
    initial_observation : Observation
        Observation returned immediately after `reset()`.
    reset_events : tuple[RecordedResetEvent, ...]
        Reset-time history recorded before the first agent action. This may
        include chance, dealer, or other authoritative setup events together
        with any reset-time auto-advanced transitions applied by the
        environment.
    reset_info : dict[str, Any]
        Public-safe metadata emitted by the scenario during reset.
    debug_reset_info : dict[str, Any]
        Privileged reset trace intended for debugging and offline analysis.
        This may include canonical-state details that are not safe to expose
        to the agent.
    steps : list[TrajectoryStep[ActionT]]
        Ordered transition records accumulated during the episode.
    """

    initial_observation: Observation
    reset_events: tuple[RecordedResetEvent, ...] = ()
    reset_info: dict[str, Any] = field(default_factory=dict)
    debug_reset_info: dict[str, Any] = field(default_factory=dict)
    steps: list[TrajectoryStep[ActionT]] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        """Return the cumulative reward for the recorded episode.

        Returns
        -------
        float
            Sum of the rewards stored in `steps`.
        """
        return sum(step.reward for step in self.steps)

    @property
    def accepted_step_count(self) -> int:
        """Return the number of accepted env steps in the episode.

        Returns
        -------
        int
            Count of trajectory steps whose agent actions were accepted.
        """
        return sum(1 for step in self.steps if step.accepted)
