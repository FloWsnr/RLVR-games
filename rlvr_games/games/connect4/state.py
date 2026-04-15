"""Canonical Connect 4 state types."""

from dataclasses import dataclass, field
from typing import Sequence

EMPTY_CELL = "."
PLAYER_X = "x"
PLAYER_O = "o"
TERMINAL_PLAYER = "terminal"
PLAYER_TOKENS = (PLAYER_X, PLAYER_O)

type Board = tuple[tuple[str, ...], ...]
type Coordinate = tuple[int, int]

_WIN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


@dataclass(slots=True, frozen=True)
class Connect4Outcome:
    """Terminal outcome summary for a canonical Connect 4 position.

    Attributes
    ----------
    is_terminal : bool
        Whether the state ends the episode.
    winner : str | None
        Winning token label, or `None` for drawn games.
    termination : str | None
        Structured termination reason. Supported values are
        ``"connect_length"`` and ``"draw"``.
    winning_cells : tuple[Coordinate, ...]
        Coordinates of one detected winning line in top-down row-major order.
        This is empty for non-terminal and drawn positions.
    """

    is_terminal: bool
    winner: str | None = None
    termination: str | None = None
    winning_cells: tuple[Coordinate, ...] = ()

    def __post_init__(self) -> None:
        """Validate that terminal metadata is internally coherent.

        Raises
        ------
        ValueError
            If terminal flags and terminal metadata disagree.
        """
        has_terminal_metadata = (
            self.winner is not None
            or self.termination is not None
            or bool(self.winning_cells)
        )
        if self.is_terminal and self.termination is None:
            raise ValueError("Terminal Connect 4 outcomes require a termination.")
        if not self.is_terminal and has_terminal_metadata:
            raise ValueError(
                "Non-terminal Connect 4 outcomes must not carry terminal metadata."
            )
        if self.winner is not None and self.winner not in PLAYER_TOKENS:
            raise ValueError("Connect 4 winners must be 'x', 'o', or None.")
        if self.termination == "connect_length" and self.winner is None:
            raise ValueError("connect_length termination requires a winner.")
        if self.termination == "draw" and self.winner is not None:
            raise ValueError("Draw termination cannot also include a winner.")
        if self.termination == "connect_length" and not self.winning_cells:
            raise ValueError("Winning Connect 4 outcomes require winning cells.")
        if self.termination != "connect_length" and self.winning_cells:
            raise ValueError(
                "Only winning Connect 4 outcomes may include winning cells."
            )

    def metadata(self) -> dict[str, object]:
        """Return dictionary metadata for the outcome.

        Returns
        -------
        dict[str, object]
            Empty dictionary for non-terminal states, otherwise terminal
            outcome metadata.
        """
        if not self.is_terminal:
            return {}
        return {
            "winner": self.winner,
            "termination": self.termination,
            "winning_cells": self.winning_cells,
        }


@dataclass(init=False, slots=True, frozen=True)
class Connect4State:
    """Canonical Connect 4 state.

    Attributes
    ----------
    board : Board
        Immutable top-down grid of ``"."``, ``"x"``, and ``"o"`` tokens.
    connect_length : int
        Number of contiguous tokens required for a win.
    rows : int
        Number of board rows.
    columns : int
        Number of board columns.
    move_count : int
        Number of pieces currently on the board.
    current_player : str
        Token whose turn it is, or ``"terminal"`` for completed positions.
    legal_actions : tuple[str, ...]
        One-based column labels accepted for the current position.
    column_heights : tuple[int, ...]
        Occupied cell count for each column.
    piece_count_x : int
        Number of ``"x"`` tokens on the board.
    piece_count_o : int
        Number of ``"o"`` tokens on the board.
    outcome : Connect4Outcome
        Terminal outcome summary for the current position.
    """

    board: Board
    connect_length: int
    rows: int = field(init=False)
    columns: int = field(init=False)
    move_count: int = field(init=False)
    current_player: str = field(init=False)
    legal_actions: tuple[str, ...] = field(init=False)
    column_heights: tuple[int, ...] = field(init=False)
    piece_count_x: int = field(init=False)
    piece_count_o: int = field(init=False)
    outcome: Connect4Outcome = field(init=False)

    def __init__(
        self,
        *,
        board: Board,
        connect_length: int,
    ) -> None:
        """Create a canonical Connect 4 state.

        Parameters
        ----------
        board : Board
            Immutable or board-like top-down grid.
        connect_length : int
            Number of contiguous tokens required for a win.

        Raises
        ------
        ValueError
            If the board or connect-length configuration is invalid.
        """
        normalized_board = normalize_board(rows=board)
        row_count = len(normalized_board)
        column_count = len(normalized_board[0])
        if connect_length < 2:
            raise ValueError("Connect 4 connect_length must be at least 2.")
        if connect_length > max(row_count, column_count):
            raise ValueError(
                "Connect 4 connect_length must fit within the board dimensions."
            )

        _validate_gravity(board=normalized_board)
        piece_count_x = sum(row.count(PLAYER_X) for row in normalized_board)
        piece_count_o = sum(row.count(PLAYER_O) for row in normalized_board)
        if piece_count_x < piece_count_o or piece_count_x > piece_count_o + 1:
            raise ValueError(
                "Connect 4 boards must have equal pieces or exactly one extra 'x'."
            )

        x_winning_cells = find_winning_cells(
            board=normalized_board,
            player=PLAYER_X,
            connect_length=connect_length,
        )
        o_winning_cells = find_winning_cells(
            board=normalized_board,
            player=PLAYER_O,
            connect_length=connect_length,
        )
        if x_winning_cells and o_winning_cells:
            raise ValueError(
                "Connect 4 boards cannot contain winning lines for both players."
            )
        if x_winning_cells and piece_count_x != piece_count_o + 1:
            raise ValueError(
                "Winning Connect 4 positions for 'x' require exactly one extra 'x'."
            )
        if o_winning_cells and piece_count_x != piece_count_o:
            raise ValueError(
                "Winning Connect 4 positions for 'o' require equal piece counts."
            )

        move_count = piece_count_x + piece_count_o
        column_heights = _column_heights(board=normalized_board)
        if x_winning_cells:
            outcome = Connect4Outcome(
                is_terminal=True,
                winner=PLAYER_X,
                termination="connect_length",
                winning_cells=x_winning_cells,
            )
            current_player = TERMINAL_PLAYER
            legal_actions: tuple[str, ...] = ()
        elif o_winning_cells:
            outcome = Connect4Outcome(
                is_terminal=True,
                winner=PLAYER_O,
                termination="connect_length",
                winning_cells=o_winning_cells,
            )
            current_player = TERMINAL_PLAYER
            legal_actions = ()
        elif move_count == row_count * column_count:
            outcome = Connect4Outcome(
                is_terminal=True,
                winner=None,
                termination="draw",
                winning_cells=(),
            )
            current_player = TERMINAL_PLAYER
            legal_actions = ()
        else:
            outcome = Connect4Outcome(is_terminal=False)
            current_player = PLAYER_X if piece_count_x == piece_count_o else PLAYER_O
            legal_actions = tuple(
                str(column_index + 1)
                for column_index, height in enumerate(column_heights)
                if height < row_count
            )

        object.__setattr__(self, "board", normalized_board)
        object.__setattr__(self, "connect_length", connect_length)
        object.__setattr__(self, "rows", row_count)
        object.__setattr__(self, "columns", column_count)
        object.__setattr__(self, "move_count", move_count)
        object.__setattr__(self, "current_player", current_player)
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "column_heights", column_heights)
        object.__setattr__(self, "piece_count_x", piece_count_x)
        object.__setattr__(self, "piece_count_o", piece_count_o)
        object.__setattr__(self, "outcome", outcome)

    @property
    def legal_action_count(self) -> int:
        """Return the number of legal columns in the current position.

        Returns
        -------
        int
            Number of legal actions.
        """
        return len(self.legal_actions)

    @property
    def is_terminal(self) -> bool:
        """Return whether the current state ends the episode.

        Returns
        -------
        bool
            `True` when the outcome marks the position as terminal.
        """
        return self.outcome.is_terminal


def make_empty_board(*, rows: int, columns: int) -> Board:
    """Return an empty Connect 4 board with the requested dimensions.

    Parameters
    ----------
    rows : int
        Number of rows in the board.
    columns : int
        Number of columns in the board.

    Returns
    -------
    Board
        Empty top-down board.
    """
    if rows < 1:
        raise ValueError("Connect 4 boards must have at least one row.")
    if columns < 1:
        raise ValueError("Connect 4 boards must have at least one column.")
    return tuple(tuple(EMPTY_CELL for _ in range(columns)) for _ in range(rows))


def normalize_board(*, rows: Sequence[Sequence[str]]) -> Board:
    """Normalize a programmatic or CLI-provided Connect 4 board.

    Parameters
    ----------
    rows : Sequence[Sequence[str]]
        Top-down nested board representation.

    Returns
    -------
    Board
        Immutable normalized board.

    Raises
    ------
    ValueError
        If the board dimensions or cell values are invalid.
    """
    if not rows:
        raise ValueError("Connect 4 boards must contain at least one row.")

    normalized_rows: list[tuple[str, ...]] = []
    expected_columns: int | None = None
    for row_index, row in enumerate(rows):
        normalized_row = tuple(str(cell).strip().lower() for cell in row)
        if not normalized_row:
            raise ValueError("Connect 4 boards must contain at least one column.")
        if expected_columns is None:
            expected_columns = len(normalized_row)
        elif len(normalized_row) != expected_columns:
            raise ValueError("Connect 4 boards must be rectangular.")

        for cell in normalized_row:
            if cell not in {EMPTY_CELL, PLAYER_X, PLAYER_O}:
                raise ValueError(
                    "Connect 4 cells must be '.', 'x', or 'o': "
                    f"row {row_index + 1} contains {cell!r}."
                )
        normalized_rows.append(normalized_row)

    return tuple(normalized_rows)


def drop_piece(
    *,
    board: Board,
    column: int,
    player: str,
) -> tuple[Board, Coordinate]:
    """Drop one token into a column and return the updated board.

    Parameters
    ----------
    board : Board
        Canonical top-down board before the move.
    column : int
        Zero-based target column index.
    player : str
        Token label to drop. Must be ``"x"`` or ``"o"``.

    Returns
    -------
    tuple[Board, Coordinate]
        Updated board and the placed token coordinate in top-down indexing.

    Raises
    ------
    ValueError
        If the player token is invalid, the column is out of range, or the
        column is already full.
    """
    if player not in PLAYER_TOKENS:
        raise ValueError("Connect 4 drop_piece player must be 'x' or 'o'.")

    row_count = len(board)
    column_count = len(board[0])
    if column < 0 or column >= column_count:
        raise ValueError(
            f"Connect 4 columns must be in [0, {column_count - 1}]: {column}."
        )

    row_index = row_count - 1
    while row_index >= 0 and board[row_index][column] != EMPTY_CELL:
        row_index -= 1
    if row_index < 0:
        raise ValueError(f"Connect 4 column {column + 1} is full.")

    mutable_rows = [list(row) for row in board]
    mutable_rows[row_index][column] = player
    return tuple(tuple(row) for row in mutable_rows), (row_index, column)


def find_winning_cells(
    *,
    board: Board,
    player: str,
    connect_length: int,
) -> tuple[Coordinate, ...]:
    """Return one detected winning line for a player.

    Parameters
    ----------
    board : Board
        Canonical top-down board to inspect.
    player : str
        Token whose winning line should be searched for.
    connect_length : int
        Required number of contiguous cells for a win.

    Returns
    -------
    tuple[Coordinate, ...]
        Coordinates for one winning line, or an empty tuple when none exists.
    """
    row_count = len(board)
    column_count = len(board[0])
    for row_index in range(row_count):
        for column_index in range(column_count):
            if board[row_index][column_index] != player:
                continue
            for delta_row, delta_column in _WIN_DIRECTIONS:
                previous_row = row_index - delta_row
                previous_column = column_index - delta_column
                if (
                    _in_bounds(
                        row=previous_row,
                        column=previous_column,
                        rows=row_count,
                        columns=column_count,
                    )
                    and board[previous_row][previous_column] == player
                ):
                    continue

                winning_cells: list[Coordinate] = []
                for step_index in range(connect_length):
                    next_row = row_index + (step_index * delta_row)
                    next_column = column_index + (step_index * delta_column)
                    if not _in_bounds(
                        row=next_row,
                        column=next_column,
                        rows=row_count,
                        columns=column_count,
                    ):
                        break
                    if board[next_row][next_column] != player:
                        break
                    winning_cells.append((next_row, next_column))

                if len(winning_cells) == connect_length:
                    return tuple(winning_cells)

    return ()


def public_connect4_metadata(state: Connect4State) -> dict[str, object]:
    """Return model-safe observation metadata for a Connect 4 state.

    Parameters
    ----------
    state : Connect4State
        Canonical Connect 4 state to summarize.

    Returns
    -------
    dict[str, object]
        Observation metadata derived from public board state only.
    """
    metadata: dict[str, object] = {
        "board": state.board,
        "rows": state.rows,
        "columns": state.columns,
        "connect_length": state.connect_length,
        "move_count": state.move_count,
        "current_player": state.current_player,
        "column_heights": state.column_heights,
        "piece_count_x": state.piece_count_x,
        "piece_count_o": state.piece_count_o,
        "is_terminal": state.is_terminal,
    }
    metadata.update(state.outcome.metadata())
    return metadata


def inspect_connect4_state(state: Connect4State) -> dict[str, object]:
    """Return a structured canonical summary of a Connect 4 state.

    Parameters
    ----------
    state : Connect4State
        Canonical Connect 4 state to inspect.

    Returns
    -------
    dict[str, object]
        Debug-oriented state summary derived from cached state fields.
    """
    metadata = public_connect4_metadata(state=state)
    metadata.update(
        {
            "legal_actions": state.legal_actions,
            "legal_action_count": state.legal_action_count,
        }
    )
    return metadata


def _validate_gravity(*, board: Board) -> None:
    """Validate that no pieces float above empty cells in any column."""
    row_count = len(board)
    column_count = len(board[0])
    for column_index in range(column_count):
        found_piece = False
        for row_index in range(row_count):
            cell = board[row_index][column_index]
            if cell == EMPTY_CELL:
                if found_piece:
                    raise ValueError(
                        "Connect 4 columns must not contain gaps beneath pieces: "
                        f"column {column_index + 1} is invalid."
                    )
            else:
                found_piece = True


def _column_heights(*, board: Board) -> tuple[int, ...]:
    """Return the occupied cell count for each column."""
    return tuple(
        sum(1 for row in board if row[column_index] != EMPTY_CELL)
        for column_index in range(len(board[0]))
    )


def _in_bounds(*, row: int, column: int, rows: int, columns: int) -> bool:
    """Return whether one row/column coordinate lies inside the board."""
    return 0 <= row < rows and 0 <= column < columns
