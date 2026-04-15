"""Minesweeper state tests."""

from rlvr_games.games.minesweeper import MinesweeperState, normalize_initial_board

FIXED_BOARD = ("*..", "...", "..*")


def test_pending_state_exposes_reveal_and_flag_actions() -> None:
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=None,
        move_count=0,
        placement_seed=7,
    )

    assert state.has_pending_mines is True
    assert state.hidden_board is None
    assert state.is_terminal is False
    assert state.legal_action_count == 18
    assert state.legal_actions[:3] == (
        "reveal 1 1",
        "reveal 1 2",
        "reveal 1 3",
    )


def test_cleared_terminal_state_has_no_legal_actions() -> None:
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        revealed=(
            (False, True, True),
            (True, True, True),
            (True, True, False),
        ),
        move_count=4,
        placement_seed=None,
    )

    assert state.is_terminal is True
    assert state.outcome.termination == "cleared"
    assert state.remaining_safe_cells == 0
    assert state.legal_actions == ()
