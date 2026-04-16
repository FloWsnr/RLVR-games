"""Yahtzee backend tests."""

from random import Random

from rlvr_games.games.yahtzee import (
    CATEGORY_ORDER,
    YahtzeeAction,
    YahtzeeActionKind,
    YahtzeeBackend,
    YahtzeeCategory,
    YahtzeeChanceModel,
    YahtzeeState,
)


def make_backend() -> YahtzeeBackend:
    """Return the standard Yahtzee backend used by backend tests."""
    return YahtzeeBackend(chance_model=YahtzeeChanceModel())


def test_parse_action_accepts_bare_category_and_reroll_all() -> None:
    backend = make_backend()
    state = YahtzeeState(
        dice=(6, 6, 6, 2, 2),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(0).getstate(),
    )

    score_result = backend.parse_action(state, "full house")
    reroll_result = backend.parse_action(state, "reroll all")

    assert score_result.require_action().label == "score full-house"
    assert reroll_result.require_action().label == "reroll 1 2 3 4 5"


def test_parse_action_rejects_agent_actions_while_waiting_for_opening_roll() -> None:
    backend = make_backend()
    state = YahtzeeState(
        dice=(0, 0, 0, 0, 0),
        rolls_used_in_turn=0,
        turns_completed=0,
        awaiting_roll=True,
        rng_state=Random(0).getstate(),
    )

    parse_result = backend.parse_action(state, "score chance")

    assert parse_result.action is None
    assert parse_result.error == "Yahtzee is waiting for an internal opening roll."


def test_apply_reroll_updates_only_requested_positions() -> None:
    backend = make_backend()
    state = YahtzeeState(
        dice=(1, 2, 3, 4, 5),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(7).getstate(),
    )
    action = backend.parse_action(state, "reroll 2 4").require_action()

    next_state, info = backend.apply_action(state, action)

    assert next_state.dice == (1, 3, 3, 2, 5)
    assert next_state.rolls_used_in_turn == 2
    assert next_state.rerolls_remaining == 1
    assert info["rerolled_positions"] == (2, 4)
    assert info["dice_before"] == (1, 2, 3, 4, 5)


def test_apply_score_records_value_and_starts_the_next_turn() -> None:
    backend = make_backend()
    state = YahtzeeState(
        dice=(6, 6, 6, 2, 2),
        rolls_used_in_turn=2,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(0).getstate(),
    )
    action = backend.parse_action(state, "score full-house").require_action()

    next_state, info = backend.apply_action(state, action)

    assert next_state.turns_completed == 1
    assert next_state.turn_number == 2
    assert next_state.awaiting_roll is True
    assert next_state.dice == (0, 0, 0, 0, 0)
    assert next_state.rolls_used_in_turn == 0
    assert (
        next_state.category_scores[CATEGORY_ORDER.index(YahtzeeCategory.FULL_HOUSE)]
        == 25
    )
    assert next_state.total_score == 25
    assert info["score_value"] == 25
    assert info["next_turn_pending"] is True
    assert info["total_score"] == 25


def test_apply_score_on_final_category_terminates_without_next_roll() -> None:
    backend = make_backend()
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

    next_state, info = backend.apply_action(state, action)

    assert next_state.is_terminal is True
    assert next_state.outcome.final_score == 50
    assert next_state.legal_actions == ()
    assert info["next_turn_pending"] is False


def test_apply_internal_opening_roll_action_is_allowed_when_waiting() -> None:
    backend = make_backend()
    state = YahtzeeState(
        dice=(0, 0, 0, 0, 0),
        rolls_used_in_turn=0,
        turns_completed=1,
        awaiting_roll=True,
        category_scores=tuple(
            25 if category == YahtzeeCategory.FULL_HOUSE else None
            for category in CATEGORY_ORDER
        ),
        rng_state=Random(0).getstate(),
    )

    next_state, info = backend.apply_action(
        state,
        YahtzeeAction(kind=YahtzeeActionKind.OPENING_ROLL),
    )

    assert next_state.awaiting_roll is False
    assert next_state.dice == (4, 4, 1, 3, 5)
    assert info["event_kind"] == "opening_roll"


def test_direct_reroll_actions_are_canonicalized_before_legality_checks() -> None:
    backend = make_backend()
    state = YahtzeeState(
        dice=(1, 2, 3, 4, 5),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=Random(7).getstate(),
    )
    action = YahtzeeAction(
        kind=YahtzeeActionKind.REROLL,
        reroll_positions=(4, 3),
    )

    next_state, info = backend.apply_action(state, action)

    assert action.reroll_positions == (3, 4)
    assert action.label == "reroll 4 5"
    assert next_state.dice == (1, 2, 3, 3, 2)
    assert info["rerolled_positions"] == (4, 5)
