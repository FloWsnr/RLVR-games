"""Mastermind backend tests."""

import pytest

from rlvr_games.games.mastermind import MastermindBackend, MastermindState


@pytest.mark.parametrize(
    "raw_action",
    ["", "guess 1 2 3", "guess 1 2 3 7", "code 1 2 3 4", "1 a 2 2"],
)
def test_parse_action_rejects_invalid_inputs(raw_action: str) -> None:
    backend = MastermindBackend()
    state = MastermindState(secret_code=(1, 1, 2, 2))

    parse_result = backend.parse_action(state, raw_action)

    assert parse_result.action is None
    assert parse_result.error is not None


def test_apply_action_records_feedback_and_candidate_count() -> None:
    backend = MastermindBackend()
    state = MastermindState(secret_code=(1, 1, 2, 2))

    action = backend.parse_action(state, "guess 1 2 1 2").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.move_count == 1
    assert next_state.is_terminal is False
    assert next_state.guess_history[0].black_pegs == 2
    assert next_state.guess_history[0].white_pegs == 2
    assert info["guess_text"] == "1 2 1 2"
    assert info["black_pegs"] == 2
    assert info["white_pegs"] == 2
    assert info["candidate_count"] == next_state.candidate_count


def test_repeating_wrong_guesses_until_turn_ten_loses_the_game() -> None:
    backend = MastermindBackend()
    state = MastermindState(secret_code=(1, 1, 2, 2))
    info: dict[str, object] = {}

    for _ in range(10):
        action = backend.parse_action(state, "3456").require_action()
        state, info = backend.apply_action(state, action)

    assert state.is_terminal is True
    assert state.outcome.termination == "out_of_guesses"
    assert info["termination"] == "out_of_guesses"
