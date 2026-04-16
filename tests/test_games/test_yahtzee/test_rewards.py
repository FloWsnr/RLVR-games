"""Yahtzee reward tests."""

from random import Random

from rlvr_games.games.yahtzee import (
    CATEGORY_ORDER,
    FinalScoreReward,
    ScoreDeltaReward,
    YahtzeeBackend,
    YahtzeeCategory,
    YahtzeeChanceModel,
    YahtzeeState,
)


def test_score_delta_reward_matches_added_points() -> None:
    backend = YahtzeeBackend(chance_model=YahtzeeChanceModel())
    state = YahtzeeState(
        dice=(6, 6, 6, 2, 2),
        rolls_used_in_turn=2,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(0).getstate(),
    )
    action = backend.parse_action(state, "score full-house").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = ScoreDeltaReward().evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 25.0


def test_final_score_reward_pays_only_on_terminal_transition() -> None:
    backend = YahtzeeBackend(chance_model=YahtzeeChanceModel())
    preterminal_scores = tuple(
        None if category == YahtzeeCategory.YAHTZEE else 0
        for category in CATEGORY_ORDER
    )
    state = YahtzeeState(
        dice=(1, 1, 1, 1, 1),
        rolls_used_in_turn=3,
        turns_completed=12,
        awaiting_roll=False,
        category_scores=preterminal_scores,
        rng_state=Random(3).getstate(),
    )
    action = backend.parse_action(state, "score yahtzee").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = FinalScoreReward().evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 50.0
