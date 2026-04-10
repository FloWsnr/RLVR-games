"""RLVR environments built around executable game verifiers."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.rollout import run_episode
from rlvr_games.core.types import EpisodeConfig, Observation, StepResult

__all__ = [
    "EpisodeConfig",
    "Observation",
    "StepResult",
    "TurnBasedEnv",
    "run_episode",
]
