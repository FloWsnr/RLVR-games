"""Mastermind state tests."""

from rlvr_games.games.mastermind import MastermindGuessRecord, MastermindState


def test_initial_state_exposes_all_standard_guesses() -> None:
    state = MastermindState(secret_code=(1, 1, 2, 2))

    assert state.is_terminal is False
    assert state.candidate_count == 1296
    assert state.legal_action_count == 1296
    assert state.legal_actions[:2] == ("guess 1 1 1 1", "guess 1 1 1 2")


def test_cracked_state_has_no_legal_actions() -> None:
    state = MastermindState(
        secret_code=(1, 1, 2, 2),
        guess_history=(
            MastermindGuessRecord(
                guess=(1, 1, 2, 2),
                black_pegs=4,
                white_pegs=0,
            ),
        ),
    )

    assert state.is_terminal is True
    assert state.outcome.termination == "cracked"
    assert state.candidate_count == 1
    assert state.legal_actions == ()
