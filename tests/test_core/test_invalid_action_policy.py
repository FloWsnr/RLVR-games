"""Invalid-action policy tests."""

import pytest

from rlvr_games.core import (
    EpisodeFinishedError,
    EpisodeConfig,
    InvalidActionError,
    InvalidActionMode,
    InvalidActionPolicy,
)
from rlvr_games.core.rollout import ActionContext, run_episode
from rlvr_games.core.types import Observation
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessEnv,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN


class ScriptedAgent:
    """Deterministic agent used to test env-owned policy behavior."""

    def __init__(self, actions: list[str]) -> None:
        """Initialize the agent with scripted action strings."""
        self._actions = list(actions)

    def act(self, observation: Observation, context: ActionContext) -> str:
        """Return the next scripted action."""
        del observation
        del context
        if not self._actions:
            raise AssertionError("ScriptedAgent ran out of actions.")
        return self._actions.pop(0)


def make_env_with_invalid_action_policy(
    *, mode: InvalidActionMode, penalty: float | None
) -> ChessEnv:
    """Construct a chess environment for invalid-action tests.

    Parameters
    ----------
    mode : InvalidActionMode
        Invalid-action handling mode to apply.
    penalty : float | None
        Reward assigned to rejected actions when the mode penalizes them.

    Returns
    -------
    ChessEnv
        Chess environment configured with the requested policy.
    """
    return make_chess_env(
        initial_fen=STANDARD_START_FEN,
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(mode=mode, penalty=penalty),
        ),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        image_output_dir=None,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )


def test_raise_mode_preserves_existing_invalid_action_behavior() -> None:
    env = make_env_with_invalid_action_policy(
        mode=InvalidActionMode.RAISE,
        penalty=None,
    )
    env.reset(seed=17)

    with pytest.raises(InvalidActionError):
        env.step("e2e5")

    assert env.state.fen == STANDARD_START_FEN
    assert len(env.trajectory.steps) == 0


def test_penalize_continue_records_rejected_attempt_and_keeps_episode_open() -> None:
    env = make_env_with_invalid_action_policy(
        mode=InvalidActionMode.PENALIZE_CONTINUE,
        penalty=-1.5,
    )
    env.reset(seed=3)

    rejected = env.step("e2e5")
    accepted = env.step("e2e4")

    assert rejected.accepted is False
    assert rejected.reward == -1.5
    assert rejected.terminated is False
    assert rejected.truncated is False
    assert rejected.info["invalid_action"] is True
    assert env.trajectory.steps[0].action is None
    assert env.trajectory.steps[0].accepted is False
    assert env.trajectory.steps[1].accepted is True
    assert env.trajectory.steps[1].action is not None
    assert env.state.fen != STANDARD_START_FEN
    assert accepted.accepted is True


def test_penalize_truncate_records_rejected_attempt_and_finishes_episode() -> None:
    env = make_env_with_invalid_action_policy(
        mode=InvalidActionMode.PENALIZE_TRUNCATE,
        penalty=-2.0,
    )
    env.reset(seed=5)

    result = env.step("e2e5")

    assert result.accepted is False
    assert result.reward == -2.0
    assert result.terminated is False
    assert result.truncated is True
    assert result.info["truncated_reason"] == "invalid_action"
    assert env.trajectory.steps[0].accepted is False

    with pytest.raises(EpisodeFinishedError):
        env.step("e2e4")


def test_run_episode_works_with_penalize_truncate_policy() -> None:
    env = make_env_with_invalid_action_policy(
        mode=InvalidActionMode.PENALIZE_TRUNCATE,
        penalty=-3.0,
    )
    agent = ScriptedAgent(["e2e5"])

    result = run_episode(env=env, agent=agent, seed=9)

    assert result.terminated is False
    assert result.truncated is True
    assert result.turn_count == 0
    assert result.trajectory.steps[0].accepted is False
    assert result.trajectory.steps[0].action is None
    assert result.trajectory.total_reward == -3.0


def test_run_episode_turn_count_only_counts_accepted_transitions() -> None:
    env = make_chess_env(
        initial_fen=STANDARD_START_FEN,
        config=EpisodeConfig(
            max_attempts=2,
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        image_output_dir=None,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
    agent = ScriptedAgent(["e2e5", "e2e4"])

    result = run_episode(env=env, agent=agent, seed=4)

    assert result.terminated is False
    assert result.truncated is True
    assert len(result.trajectory.steps) == 2
    assert result.turn_count == 1
    assert result.trajectory.accepted_step_count == 1
