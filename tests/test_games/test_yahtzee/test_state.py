"""Yahtzee state tests."""

from copy import deepcopy
import pickle
from random import Random

import pytest

from rlvr_games.games.yahtzee import CATEGORY_ORDER, YahtzeeState


def test_state_derives_score_options_and_legal_actions_for_first_roll() -> None:
    state = YahtzeeState(
        dice=(1, 2, 3, 4, 5),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(0).getstate(),
    )

    assert state.rerolls_remaining == 2
    assert state.turn_number == 1
    assert state.legal_action_count == 44
    assert state.available_score_options["ones"] == 1
    assert state.available_score_options["small-straight"] == 30
    assert state.available_score_options["large-straight"] == 40
    assert state.available_score_options["chance"] == 15
    assert state.legal_actions[:3] == ("score ones", "score twos", "score threes")
    assert "reroll 1" in state.legal_actions
    assert "reroll 1 2 3 4 5" in state.legal_actions


def test_state_rejects_awaiting_roll_with_nonzero_dice() -> None:
    with pytest.raises(ValueError, match="zero placeholder dice"):
        YahtzeeState(
            dice=(1, 1, 1, 1, 1),
            rolls_used_in_turn=0,
            turns_completed=0,
            awaiting_roll=True,
            rng_state=Random(0).getstate(),
        )


def test_terminal_state_has_final_score_and_no_legal_actions() -> None:
    state = YahtzeeState(
        dice=(1, 1, 1, 1, 1),
        rolls_used_in_turn=3,
        turns_completed=len(CATEGORY_ORDER),
        awaiting_roll=False,
        category_scores=tuple(0 for _ in CATEGORY_ORDER),
        rng_state=Random(0).getstate(),
    )

    assert state.is_terminal is True
    assert state.outcome.final_score == 0
    assert state.turn_number == 13
    assert state.legal_actions == ()


def test_available_score_options_do_not_alias_internal_state() -> None:
    state = YahtzeeState(
        dice=(1, 2, 3, 4, 5),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(0).getstate(),
    )
    available_score_options = state.available_score_options

    available_score_options["chance"] = 999

    assert state.available_score_options["chance"] == 15


def test_state_supports_deepcopy_and_pickle_roundtrip() -> None:
    state = YahtzeeState(
        dice=(1, 2, 3, 4, 5),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(0).getstate(),
    )

    copied_state = deepcopy(state)
    restored_state = pickle.loads(pickle.dumps(state))

    assert copied_state == state
    assert copied_state is not state
    assert restored_state == state
