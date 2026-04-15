"""2048 environment integration tests."""

import pytest

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.game2048 import (
    FixedBoardScenario,
    Game2048AsciiBoardFormatter,
    Game2048Backend,
    Game2048ChanceModel,
    Game2048ObservationRenderer,
    RandomStartScenario,
    TargetTileReward,
)
from rlvr_games.games.game2048.state import inspect_game2048_state

WIN_BOARD = (
    (1024, 1024, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)
NO_MOVES_BOARD = (
    (2, 4, 2, 4),
    (4, 2, 4, 2),
    (2, 4, 2, 4),
    (4, 2, 4, 8),
)


def make_renderer() -> Game2048ObservationRenderer:
    """Return the standard text-only 2048 renderer."""
    return Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=None,
    )


def test_reaching_target_tile_terminates_with_reward_and_metadata() -> None:
    chance_model = Game2048ChanceModel()
    env = TurnBasedEnv(
        backend=Game2048Backend(chance_model=chance_model),
        scenario=FixedBoardScenario(
            initial_board=WIN_BOARD,
            initial_score=0,
            initial_move_count=0,
            target_value=2048,
            chance_model=chance_model,
        ),
        renderer=make_renderer(),
        inspect_canonical_state_fn=inspect_game2048_state,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
    )
    env.reset(seed=1)

    result = env.step("left")

    assert result.accepted is True
    assert result.reward == 2048.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["spawned_tile"] is None
    assert result.info["termination"] == "target_tile"
    assert result.info["won"] is True
    assert env.state.board == (
        (2048, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert env.state.legal_actions == ()
    assert result.observation.metadata["is_terminal"] is True
    assert result.observation.metadata["won"] is True


def test_terminal_reset_marks_episode_finished_and_rejects_steps() -> None:
    chance_model = Game2048ChanceModel()
    env = TurnBasedEnv(
        backend=Game2048Backend(chance_model=chance_model),
        scenario=FixedBoardScenario(
            initial_board=NO_MOVES_BOARD,
            initial_score=0,
            initial_move_count=0,
            target_value=2048,
            chance_model=chance_model,
        ),
        renderer=make_renderer(),
        inspect_canonical_state_fn=inspect_game2048_state,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
    )

    observation, _ = env.reset(seed=23)

    assert env.episode_finished is True
    assert observation.metadata["is_terminal"] is True
    assert observation.metadata["termination"] == "no_moves"

    with pytest.raises(EpisodeFinishedError):
        env.step("left")


def test_env_records_trajectory_with_real_backend() -> None:
    chance_model = Game2048ChanceModel()
    env = TurnBasedEnv(
        backend=Game2048Backend(chance_model=chance_model),
        scenario=RandomStartScenario(
            size=4,
            target_value=2048,
            start_tile_count=2,
            chance_model=chance_model,
        ),
        renderer=make_renderer(),
        inspect_canonical_state_fn=inspect_game2048_state,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
    )
    observation, info = env.reset(seed=0)
    initial_debug_reset_info = dict(env.trajectory.debug_reset_info)
    reset_debug_state = env.inspect_canonical_state()

    result = env.step("left")

    assert "seed" not in info
    assert "2048 board:" in (observation.text or "")
    assert initial_debug_reset_info["rng_state"] == reset_debug_state["rng_state"]
    assert result.reward == 0.0
    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].accepted is True
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.label == "left"
    assert env.trajectory.steps[0].info["direction"] == "left"
    assert "rng_state" not in result.info
    assert (
        env.trajectory.steps[0].debug_info["rng_state"]
        == env.inspect_canonical_state()["rng_state"]
    )
