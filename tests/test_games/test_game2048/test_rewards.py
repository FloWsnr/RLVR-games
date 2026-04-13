"""2048 reward tests."""

from random import Random

from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.game2048 import (
    Game2048Backend,
    Game2048ChanceModel,
    Game2048State,
    ScoreDeltaReward,
    TargetTileReward,
    make_game2048_env,
)

MERGE_BOARD = (
    (2, 2, 2, 2),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)


def test_score_delta_reward_returns_merge_score_gain_for_non_terminal_move() -> None:
    backend = Game2048Backend(chance_model=Game2048ChanceModel())
    state = Game2048State(
        board=MERGE_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(5).getstate(),
    )
    action = backend.parse_action(state, "left").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = ScoreDeltaReward().evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 8.0


def test_target_tile_reward_ignores_non_terminal_score_gain() -> None:
    env = make_game2048_env(
        size=4,
        target_value=2048,
        initial_board=MERGE_BOARD,
        initial_score=0,
        initial_move_count=0,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )
    env.reset(seed=5)

    result = env.step("left")

    assert result.accepted is True
    assert result.reward == 0.0
    assert result.info["score_gain"] == 8
    assert result.terminated is False
