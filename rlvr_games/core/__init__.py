"""Core environment abstractions for RLVR games."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import (
    EpisodeFinishedError,
    EnvironmentNotResetError,
    InvalidActionError,
    RLVRGamesError,
)
from rlvr_games.core.protocol import Environment, ImageRenderer, TextRenderer
from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.rollout import (
    ActionContext,
    RolloutAgent,
    RolloutResult,
    build_action_context,
    run_episode,
)
from rlvr_games.core.trajectory import EpisodeTrajectory, TrajectoryStep
from rlvr_games.core.types import EpisodeConfig, Observation, StepResult
from rlvr_games.core.wrappers import (
    InvalidActionMode,
    InvalidActionPolicy,
    InvalidActionPolicyEnv,
)

__all__ = [
    "ActionContext",
    "Environment",
    "EpisodeConfig",
    "EpisodeFinishedError",
    "EpisodeTrajectory",
    "EnvironmentNotResetError",
    "ImageRenderer",
    "InvalidActionError",
    "InvalidActionMode",
    "InvalidActionPolicy",
    "InvalidActionPolicyEnv",
    "Observation",
    "RLVRGamesError",
    "RolloutAgent",
    "RolloutResult",
    "StepResult",
    "TextRenderer",
    "TrajectoryStep",
    "TurnBasedEnv",
    "ZeroReward",
    "build_action_context",
    "run_episode",
]
