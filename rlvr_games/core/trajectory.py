"""Trajectory recording types."""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from rlvr_games.core.types import Observation

ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class TrajectoryStep(Generic[ActionT]):
    """One environment transition."""

    raw_action: str
    action: ActionT
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrajectory(Generic[ActionT]):
    """Recorded episode data."""

    initial_observation: Observation
    reset_info: dict[str, Any] = field(default_factory=dict)
    steps: list[TrajectoryStep[ActionT]] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        """Return the cumulative reward."""
        return sum(step.reward for step in self.steps)
