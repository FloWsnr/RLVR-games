"""RLVR environments built around executable game verifiers."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.rollout import run_episode
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
    Observation,
    ParseResult,
    RenderedImage,
    StepResult,
)

__all__ = [
    "EpisodeConfig",
    "InvalidActionMode",
    "InvalidActionPolicy",
    "Observation",
    "ParseResult",
    "RenderedImage",
    "StepResult",
    "TurnBasedEnv",
    "run_episode",
]
