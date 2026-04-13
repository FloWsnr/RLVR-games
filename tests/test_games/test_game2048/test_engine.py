"""2048 pure engine tests."""

from rlvr_games.games.game2048 import spawn_outcomes
from rlvr_games.games.game2048.actions import MoveDirection
from rlvr_games.games.game2048.engine import apply_move
import pytest


def test_apply_move_matches_open_spiel_three_tiles_downward_case() -> None:
    board = (
        (0, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 0, 0, 0),
    )

    move_summary = apply_move(board=board, direction=MoveDirection.DOWN)

    assert move_summary.board == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (2, 0, 0, 0),
        (4, 0, 0, 0),
    )


def test_apply_move_matches_open_spiel_one_merge_per_turn_case() -> None:
    board = (
        (2, 4, 0, 4),
        (0, 2, 0, 2),
        (0, 0, 0, 0),
        (0, 2, 0, 0),
    )

    move_summary = apply_move(board=board, direction=MoveDirection.DOWN)

    assert move_summary.board == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 4, 0, 4),
        (2, 4, 0, 2),
    )


def test_spawn_outcomes_cover_all_empty_cells_with_probabilities() -> None:
    board = (
        (2, 4, 8, 16),
        (32, 64, 128, 256),
        (512, 1024, 0, 4096),
        (8192, 16384, 0, 32768),
    )

    outcomes = spawn_outcomes(board=board)

    assert len(outcomes) == 4
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    assert {
        (outcome.spawned_tile.row, outcome.spawned_tile.col) for outcome in outcomes
    } == {
        (2, 2),
        (3, 2),
    }
    probability_by_tile = {
        (
            outcome.spawned_tile.row,
            outcome.spawned_tile.col,
            outcome.spawned_tile.value,
        ): outcome.probability
        for outcome in outcomes
    }
    assert probability_by_tile[(2, 2, 2)] == pytest.approx(0.45)
    assert probability_by_tile[(2, 2, 4)] == pytest.approx(0.05)
    assert probability_by_tile[(3, 2, 2)] == pytest.approx(0.45)
    assert probability_by_tile[(3, 2, 4)] == pytest.approx(0.05)
