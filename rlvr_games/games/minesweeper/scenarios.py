"""Minesweeper scenario initializers."""

from dataclasses import dataclass
from typing import Sequence

from rlvr_games.games.minesweeper.engine import (
    MineBoard,
    count_mines,
    normalize_mine_board,
)
from rlvr_games.games.minesweeper.state import MinesweeperState

STANDARD_MINESWEEPER_ROWS = 9
STANDARD_MINESWEEPER_COLUMNS = 9
STANDARD_MINESWEEPER_MINE_COUNT = 10


@dataclass(slots=True)
class RandomBoardScenario:
    """Scenario that initializes Minesweeper with deferred random placement.

    Attributes
    ----------
    rows : int
        Board row count.
    columns : int
        Board column count.
    mine_count : int
        Number of mines to place on the first reveal action.
    """

    rows: int
    columns: int
    mine_count: int

    def __post_init__(self) -> None:
        """Validate scenario configuration."""
        if self.rows <= 0:
            raise ValueError("Minesweeper rows must be positive.")
        if self.columns <= 0:
            raise ValueError("Minesweeper columns must be positive.")
        if self.mine_count < 0:
            raise ValueError("Minesweeper mine_count must be non-negative.")
        if self.mine_count >= self.rows * self.columns:
            raise ValueError("mine_count must leave at least one safe cell.")

    def reset(self, *, seed: int) -> tuple[MinesweeperState, dict[str, object]]:
        """Create a fresh random Minesweeper episode.

        Parameters
        ----------
        seed : int
            Explicit seed used for deterministic deferred mine placement.

        Returns
        -------
        tuple[MinesweeperState, dict[str, object]]
            Canonical initial state and reset metadata.
        """
        state = MinesweeperState(
            rows=self.rows,
            columns=self.columns,
            mine_count=self.mine_count,
            hidden_board=None,
            move_count=0,
            placement_seed=seed,
        )
        return state, {
            "scenario": "random_board",
            "seed": seed,
            "rows": self.rows,
            "columns": self.columns,
            "mine_count": self.mine_count,
            "pending_mine_layout": True,
        }


@dataclass(slots=True)
class FixedBoardScenario:
    """Scenario that initializes Minesweeper from an explicit mine layout.

    Attributes
    ----------
    hidden_board : MineBoard
        Explicit hidden mine layout in row-major order.
    """

    hidden_board: MineBoard

    def __post_init__(self) -> None:
        """Normalize the configured hidden board."""
        self.hidden_board = normalize_mine_board(board=self.hidden_board)

    def reset(self, *, seed: int) -> tuple[MinesweeperState, dict[str, object]]:
        """Create a fresh fixed-layout Minesweeper episode.

        Parameters
        ----------
        seed : int
            Explicit seed supplied for API consistency. It is reported in the
            reset metadata and otherwise unused.

        Returns
        -------
        tuple[MinesweeperState, dict[str, object]]
            Canonical initial state and reset metadata.
        """
        del seed
        state = MinesweeperState(
            rows=len(self.hidden_board),
            columns=len(self.hidden_board[0]),
            mine_count=count_mines(board=self.hidden_board),
            hidden_board=self.hidden_board,
            move_count=0,
            placement_seed=None,
        )
        return state, {
            "scenario": "fixed_board",
            "rows": state.rows,
            "columns": state.columns,
            "mine_count": state.mine_count,
            "hidden_board": self.hidden_board,
            "pending_mine_layout": False,
        }


def normalize_initial_board(
    *,
    board: Sequence[Sequence[bool | str | int] | str],
) -> MineBoard:
    """Normalize a programmatic or CLI-provided mine layout."""
    return normalize_mine_board(board=board)
