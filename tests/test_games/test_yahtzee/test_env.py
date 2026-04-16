"""Yahtzee environment integration tests."""

import pytest

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.yahtzee import (
    FinalScoreReward,
    ScoreDeltaReward,
    YahtzeeAction,
    YahtzeeState,
    make_yahtzee_env,
)
from rlvr_games.games.yahtzee.chance import YahtzeeChanceModel


def make_standard_env() -> TurnBasedEnv[YahtzeeState, YahtzeeAction]:
    """Return the standard Yahtzee environment from a fresh game."""
    return make_yahtzee_env(
        initial_state=None,
        reward_fn=FinalScoreReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=320,
    )


def make_active_state() -> YahtzeeState:
    """Return a representative fixed active state for env tests."""
    return YahtzeeState(
        dice=(6, 6, 6, 2, 2),
        rolls_used_in_turn=2,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=YahtzeeChanceModel().initial_rng_state(seed=0),
    )


def test_reset_applies_the_opening_roll_and_records_a_reset_event() -> None:
    env = make_standard_env()

    observation, info = env.reset(seed=0)

    assert info["scenario"] == "standard_game"
    assert observation.metadata["dice"] == (4, 4, 1, 3, 5)
    assert observation.metadata["rolls_used_in_turn"] == 1
    assert len(env.trajectory.reset_events) == 1
    assert env.trajectory.reset_events[0].source == "chance"
    assert env.trajectory.reset_events[0].label == "opening-roll 4 4 1 3 5"
    assert "rng_state" not in info
    assert env.trajectory.reset_events[0].debug_info["rng_state"] != ()


def test_env_scores_a_category_and_starts_the_next_turn() -> None:
    env = make_yahtzee_env(
        initial_state=make_active_state(),
        reward_fn=ScoreDeltaReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=320,
    )
    env.reset(seed=7)

    result = env.step("score full-house")

    assert result.accepted is True
    assert result.reward == 25.0
    assert result.terminated is False
    assert result.info["transition_count"] == 2
    assert result.info["transition_count_delta"] == 2
    assert result.info["auto_advanced"] is True
    assert result.info["score_value"] == 25
    assert len(result.info["internal_transitions"]) == 1
    assert result.info["internal_transitions"][0]["source"] == "chance"
    assert result.info["internal_transitions"][0]["info"]["dice"] == (4, 4, 1, 3, 5)
    assert env.state.turn_number == 2
    assert env.state.total_score == 25
    assert env.state.dice == (4, 4, 1, 3, 5)
    assert len(env.trajectory.steps[0].transitions) == 2
    assert env.trajectory.steps[0].transitions[1].raw_action == "opening-roll"


def test_score_step_respects_max_transition_truncation_after_opening_roll() -> None:
    env = make_yahtzee_env(
        initial_state=make_active_state(),
        reward_fn=ScoreDeltaReward(),
        config=EpisodeConfig(max_transitions=1),
        include_images=False,
        image_size=320,
    )
    env.reset(seed=7)

    result = env.step("score full-house")

    assert result.accepted is True
    assert result.truncated is True
    assert result.info["truncated_reason"] == "max_transitions"
    assert result.info["transition_count"] == 2
    assert result.info["transition_count_delta"] == 2


def test_terminal_reset_marks_episode_finished_and_rejects_steps() -> None:
    terminal_state = YahtzeeState(
        dice=(1, 1, 1, 1, 1),
        rolls_used_in_turn=3,
        turns_completed=13,
        awaiting_roll=False,
        category_scores=tuple(0 for _ in range(13)),
        rng_state=YahtzeeChanceModel().initial_rng_state(seed=0),
    )
    env = make_yahtzee_env(
        initial_state=terminal_state,
        reward_fn=FinalScoreReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=320,
    )

    observation, _ = env.reset(seed=23)

    assert env.episode_finished is True
    assert observation.metadata["is_terminal"] is True
    assert observation.metadata["final_score"] == 0

    with pytest.raises(EpisodeFinishedError):
        env.step("score chance")


def test_fixed_awaiting_roll_states_reset_cleanly_and_apply_the_opening_roll() -> None:
    awaiting_roll_state = YahtzeeState(
        dice=(0, 0, 0, 0, 0),
        rolls_used_in_turn=0,
        turns_completed=0,
        awaiting_roll=True,
        rng_state=YahtzeeChanceModel().initial_rng_state(seed=0),
    )
    env = make_yahtzee_env(
        initial_state=awaiting_roll_state,
        reward_fn=FinalScoreReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=320,
    )

    observation, info = env.reset(seed=23)

    assert info["scenario"] == "fixed_state"
    assert observation.metadata["dice"] == (4, 4, 1, 3, 5)
    assert env.state.awaiting_roll is False
    assert len(env.trajectory.reset_events) == 1
    assert env.trajectory.reset_events[0].source == "chance"
