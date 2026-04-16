"""Yahtzee scenario tests."""

from rlvr_games.games.yahtzee import (
    FixedStateScenario,
    StandardGameScenario,
    YahtzeeChanceModel,
    YahtzeeState,
)


def test_standard_game_scenario_returns_awaiting_roll_state_and_rules_metadata() -> (
    None
):
    scenario = StandardGameScenario(chance_model=YahtzeeChanceModel())

    reset = scenario.reset(seed=0)
    state = reset.initial_state
    info = reset.reset_info

    assert state.awaiting_roll is True
    assert state.rolls_used_in_turn == 0
    assert state.dice == (0, 0, 0, 0, 0)
    assert info["scenario"] == "standard_game"
    assert info["turn_count"] == 13
    assert info["upper_bonus_enabled"] is False
    assert info["extra_yahtzee_bonus_enabled"] is False
    assert info["joker_rule_enabled"] is False


def test_fixed_state_scenario_preserves_the_supplied_state() -> None:
    initial_state = YahtzeeState(
        dice=(4, 4, 1, 3, 5),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=YahtzeeChanceModel().initial_rng_state(seed=0),
    )
    scenario = FixedStateScenario(initial_state=initial_state)

    reset = scenario.reset(seed=99)

    assert reset.initial_state == initial_state
    assert reset.initial_state is not initial_state
    assert reset.reset_info["scenario"] == "fixed_state"
    assert reset.reset_info["total_score"] == 0
