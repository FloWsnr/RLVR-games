"""Connect 4 scenario tests."""

import pytest

from rlvr_games.games.connect4 import (
    FixedBoardScenario,
    RandomPositionScenario,
)
from tests.test_games.test_connect4.support import PRE_WIN_BOARD, board_from_rows


def test_random_position_scenario_is_deterministic_and_non_terminal() -> None:
    scenario = RandomPositionScenario(
        rows=6,
        columns=7,
        connect_length=4,
        min_start_moves=6,
        max_start_moves=6,
    )

    first_reset = scenario.reset(seed=19)
    second_reset = scenario.reset(seed=19)
    first_state = first_reset.initial_state
    second_state = second_reset.initial_state
    first_info = first_reset.reset_info
    second_info = second_reset.reset_info

    assert first_state.board == second_state.board
    assert first_info == second_info
    assert first_state.is_terminal is False
    assert first_info["applied_start_moves"] == 6


def test_fixed_board_scenario_preserves_board() -> None:
    scenario = FixedBoardScenario(
        initial_board=PRE_WIN_BOARD,
        connect_length=4,
    )

    reset = scenario.reset(seed=7)
    state = reset.initial_state
    info = reset.reset_info

    assert state.board == PRE_WIN_BOARD
    assert info["scenario"] == "fixed_board"
    assert info["initial_board"] == PRE_WIN_BOARD


def test_random_position_scenario_rejects_non_standard_variant() -> None:
    with pytest.raises(ValueError, match="standard 6x7 board"):
        RandomPositionScenario(
            rows=5,
            columns=7,
            connect_length=4,
        )


def test_fixed_board_scenario_rejects_non_standard_board_size() -> None:
    with pytest.raises(ValueError, match="standard 6x7 board"):
        FixedBoardScenario(
            initial_board=board_from_rows(
                ".......",
                ".......",
                ".......",
                ".......",
                ".......",
            )
        )
