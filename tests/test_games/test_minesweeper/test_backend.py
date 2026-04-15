"""Minesweeper backend tests."""

import pytest

from rlvr_games.games.minesweeper import (
    MinesweeperBackend,
    MinesweeperState,
    normalize_initial_board,
)

FIXED_BOARD = ("*..", "...", "..*")


def test_legal_actions_exclude_revealed_cells_and_replace_flagged_cells_with_unflag() -> (
    None
):
    backend = MinesweeperBackend()
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        revealed=(
            (False, False, True),
            (False, False, False),
            (False, False, False),
        ),
        flagged=(
            (True, False, False),
            (False, False, False),
            (False, False, False),
        ),
        move_count=2,
        placement_seed=None,
    )

    legal_actions = backend.legal_actions(state)

    assert "reveal 1 1" not in legal_actions
    assert "flag 1 1" not in legal_actions
    assert "unflag 1 1" in legal_actions
    assert "reveal 1 3" not in legal_actions


@pytest.mark.parametrize(
    "raw_action",
    ["", "dig 1 1", "reveal 4 1", "unflag 1 2"],
)
def test_parse_action_rejects_invalid_or_illegal_actions(raw_action: str) -> None:
    backend = MinesweeperBackend()
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        move_count=0,
        placement_seed=None,
    )

    parse_result = backend.parse_action(state, raw_action)

    assert parse_result.action is None
    assert parse_result.error is not None


def test_first_reveal_lazily_places_mines_and_keeps_the_clicked_cell_safe() -> None:
    backend = MinesweeperBackend()
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=None,
        move_count=0,
        placement_seed=7,
    )

    action = backend.parse_action(state, "reveal 2 2").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.hidden_board is not None
    assert next_state.hidden_board[1][1] is False
    assert next_state.has_pending_mines is False
    assert sum(1 for row in next_state.hidden_board for cell in row if cell) == 2
    assert info["generated_mines"] is True
    newly_revealed_count = info["newly_revealed_count"]
    assert isinstance(newly_revealed_count, int)
    assert newly_revealed_count >= 1


def test_apply_action_reveals_zero_region_and_boundary_numbers() -> None:
    backend = MinesweeperBackend()
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        move_count=0,
        placement_seed=None,
    )

    action = backend.parse_action(state, "reveal 1 3").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.revealed == (
        (False, True, True),
        (False, True, True),
        (False, False, False),
    )
    assert next_state.revealed_safe_count == 4
    assert info["newly_revealed_count"] == 4
    assert info["exploded"] is False
    assert info["adjacent_mines"] == 0


def test_apply_action_revealing_a_mine_terminates_the_episode() -> None:
    backend = MinesweeperBackend()
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        move_count=0,
        placement_seed=None,
    )

    action = backend.parse_action(state, "reveal 1 1").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.is_terminal is True
    assert next_state.outcome.termination == "mine"
    assert info["exploded"] is True
    assert info["termination"] == "mine"
    assert info["exploded_cell"] == {
        "row": 1,
        "col": 1,
        "row_index": 0,
        "col_index": 0,
    }


def test_flag_and_unflag_do_not_resolve_a_pending_layout() -> None:
    backend = MinesweeperBackend()
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=None,
        move_count=0,
        placement_seed=17,
    )

    flagged_state, flagged_info = backend.apply_action(
        state,
        backend.parse_action(state, "flag 1 1").require_action(),
    )
    unflagged_state, unflagged_info = backend.apply_action(
        flagged_state,
        backend.parse_action(flagged_state, "unflag 1 1").require_action(),
    )

    assert flagged_state.hidden_board is None
    assert flagged_state.flagged[0][0] is True
    assert flagged_info["generated_mines"] is False
    assert unflagged_state.hidden_board is None
    assert unflagged_state.flagged[0][0] is False
    assert unflagged_info["generated_mines"] is False
