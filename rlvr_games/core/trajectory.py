"""Trajectory recording types."""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from rlvr_games.core.types import Observation

ActionT = TypeVar("ActionT")


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
        Transition metadata emitted by the backend.
    """

    raw_action: str
    action: ActionT | None
    accepted: bool
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrajectory(Generic[ActionT]):
    """Recorded data for a full environment episode.

    Attributes
    ----------
    initial_observation : Observation
        Observation returned immediately after `reset()`.
    reset_info : dict[str, Any]
        Metadata emitted by the scenario during reset.
    steps : list[TrajectoryStep[ActionT]]
        Ordered transition records accumulated during the episode.
    """

    initial_observation: Observation
    reset_info: dict[str, Any] = field(default_factory=dict)
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
