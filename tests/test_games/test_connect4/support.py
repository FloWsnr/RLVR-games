"""Shared Connect 4 test helpers and fixtures."""

from rlvr_games.games.connect4.state import Board


def board_from_rows(*rows: str) -> Board:
    """Return a board fixture from top-down compact row strings."""
    return tuple(tuple(cell for cell in row) for row in rows)


PRE_WIN_BOARD = board_from_rows(
    ".......",
    ".......",
    ".......",
    ".......",
    "ooo....",
    "xxx....",
)

VERTICAL_THREAT_BOARD = board_from_rows(
    ".......",
    ".......",
    ".......",
    "xo.....",
    "xo.....",
    "xo.....",
)

FULL_FIRST_COLUMN_BOARD = board_from_rows(
    "o......",
    "x......",
    "o......",
    "x......",
    "o......",
    "xx.....",
)

X_WIN_BOARD = board_from_rows(
    ".......",
    ".......",
    "x......",
    "xo.....",
    "xo.....",
    "xo.....",
)

DRAW_BOARD = board_from_rows(
    "ooxxxoo",
    "xxoooxx",
    "ooxxxoo",
    "xxoooxx",
    "ooxxxoo",
    "xxoooxx",
)
