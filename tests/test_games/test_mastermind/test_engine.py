"""Mastermind engine tests."""

from rlvr_games.games.mastermind import (
    Feedback,
    consistent_code_count,
    score_guess,
)


def test_score_guess_handles_duplicate_digits_without_overcounting() -> None:
    feedback = score_guess(secret_code=(1, 1, 2, 2), guess=(1, 2, 1, 2))

    assert feedback == Feedback(black_pegs=2, white_pegs=2)


def test_consistent_code_count_collapses_to_one_for_a_solved_feedback() -> None:
    solved_feedback = Feedback(black_pegs=4, white_pegs=0)

    assert consistent_code_count(history=(((1, 1, 2, 2), solved_feedback),)) == 1
