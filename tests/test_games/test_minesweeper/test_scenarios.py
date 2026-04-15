"""Minesweeper scenario tests."""

from rlvr_games.games.minesweeper import (
    FixedBoardScenario,
    RandomBoardScenario,
    normalize_initial_board,
)

FIXED_BOARD = ("*..", "...", "..*")


def test_random_board_scenario_is_seeded_and_starts_with_pending_layout() -> None:
    scenario = RandomBoardScenario(rows=3, columns=3, mine_count=2)

    state, info = scenario.reset(seed=11)

    assert state.hidden_board is None
    assert state.has_pending_mines is True
    assert state.placement_seed == 11
    assert info == {
        "scenario": "random_board",
        "seed": 11,
        "rows": 3,
        "columns": 3,
        "mine_count": 2,
        "pending_mine_layout": True,
    }


def test_fixed_board_scenario_uses_the_supplied_hidden_layout() -> None:
    scenario = FixedBoardScenario(
        hidden_board=normalize_initial_board(board=FIXED_BOARD)
    )

    state, info = scenario.reset(seed=23)

    assert state.hidden_board == (
        (True, False, False),
        (False, False, False),
        (False, False, True),
    )
    assert state.mine_count == 2
    assert state.has_pending_mines is False
    assert info["scenario"] == "fixed_board"
    assert info["rows"] == 3
    assert info["columns"] == 3
    assert info["mine_count"] == 2
