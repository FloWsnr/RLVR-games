"""Core environment abstractions for RLVR games."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import (
    EpisodeFinishedError,
    EnvironmentNotResetError,
    InvalidActionError,
    RLVRGamesError,
)
from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.trajectory import EpisodeTrajectory, TrajectoryStep
from rlvr_games.core.types import EpisodeConfig, Observation, StepResult

__all__ = [
    "EpisodeConfig",
    "EpisodeFinishedError",
    "EpisodeTrajectory",
    "EnvironmentNotResetError",
    "ImageRenderer",
    "InvalidActionError",
    "Observation",
    "RLVRGamesError",
    "StepResult",
    "TextRenderer",
    "TrajectoryStep",
    "TurnBasedEnv",
    "ZeroReward",
]
