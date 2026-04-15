"""Pure board helpers for Minesweeper."""

from collections import deque
from random import Random
from typing import Sequence

Coordinate = tuple[int, int]
BoolBoard = tuple[tuple[bool, ...], ...]
MineBoard = tuple[tuple[bool, ...], ...]
AdjacentMineBoard = tuple[tuple[int, ...], ...]


def make_boolean_board(*, rows: int, columns: int, fill: bool) -> BoolBoard:
    """Return a rectangular immutable boolean grid.

    Parameters
    ----------
    rows : int
        Number of rows in the board.
    columns : int
        Number of columns in the board.
    fill : bool
        Boolean value to place in every cell.

    Returns
    -------
    BoolBoard
        Rectangular immutable boolean grid.
    """
    if rows <= 0:
        raise ValueError("Minesweeper boards must have at least one row.")
    if columns <= 0:
        raise ValueError("Minesweeper boards must have at least one column.")
    return tuple(tuple(fill for _ in range(columns)) for _ in range(rows))


def normalize_mine_board(
    *,
    board: Sequence[Sequence[bool | str | int] | str],
) -> MineBoard:
    """Normalize a board-like mine layout into an immutable boolean grid.

    Parameters
    ----------
    board : Sequence[Sequence[bool | str | int] | str]
        Mine-layout rows using booleans, ``0``/``1``, or single-character
        string markers such as ``"."`` and ``"*"``

    Returns
    -------
    MineBoard
        Normalized immutable boolean mine grid where `True` means mine.
    """
    normalized_rows: list[tuple[bool, ...]] = []
    expected_columns: int | None = None
    for row in board:
        if isinstance(row, str):
            raw_cells = tuple(row.strip())
        else:
            raw_cells = tuple(row)
        if not raw_cells:
            raise ValueError("Minesweeper boards cannot contain empty rows.")

        normalized_row = tuple(_normalize_mine_cell(cell=cell) for cell in raw_cells)
        if expected_columns is None:
            expected_columns = len(normalized_row)
        elif len(normalized_row) != expected_columns:
            raise ValueError("Minesweeper boards must be rectangular.")
        normalized_rows.append(normalized_row)

    if not normalized_rows:
        raise ValueError("Minesweeper boards must contain at least one row.")
    return tuple(normalized_rows)


def count_mines(*, board: MineBoard) -> int:
    """Return the number of mines in a normalized board."""
    return sum(1 for row in board for cell in row if cell)


def mine_board_text_rows(*, board: MineBoard) -> tuple[str, ...]:
    """Return human-readable mine-layout rows.

    Parameters
    ----------
    board : MineBoard
        Normalized mine grid.

    Returns
    -------
    tuple[str, ...]
        Rows using ``"*"`` for mines and ``"."`` for safe cells.
    """
    return tuple("".join("*" if cell else "." for cell in row) for row in board)


def neighbors(*, rows: int, columns: int, row: int, col: int) -> tuple[Coordinate, ...]:
    """Return the neighboring cell coordinates around one location.

    Parameters
    ----------
    rows : int
        Total row count.
    columns : int
        Total column count.
    row : int
        Zero-based row index.
    col : int
        Zero-based column index.

    Returns
    -------
    tuple[Coordinate, ...]
        Neighbor coordinates in row-major order.
    """
    coordinates: list[Coordinate] = []
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            next_row = row + row_offset
            next_col = col + col_offset
            if 0 <= next_row < rows and 0 <= next_col < columns:
                coordinates.append((next_row, next_col))
    return tuple(coordinates)


def adjacent_mine_counts(*, board: MineBoard) -> AdjacentMineBoard:
    """Return the adjacent-mine count board for a mine layout.

    Parameters
    ----------
    board : MineBoard
        Normalized mine layout.

    Returns
    -------
    AdjacentMineBoard
        Adjacent-mine counts for every cell in the board.
    """
    rows = len(board)
    columns = len(board[0])
    counts: list[tuple[int, ...]] = []
    for row_index in range(rows):
        row_counts: list[int] = []
        for col_index in range(columns):
            row_counts.append(
                sum(
                    1
                    for neighbor_row, neighbor_col in neighbors(
                        rows=rows,
                        columns=columns,
                        row=row_index,
                        col=col_index,
                    )
                    if board[neighbor_row][neighbor_col]
                )
            )
        counts.append(tuple(row_counts))
    return tuple(counts)


def place_random_mines(
    *,
    rows: int,
    columns: int,
    mine_count: int,
    safe_cell: Coordinate,
    seed: int,
) -> MineBoard:
    """Place mines deterministically while excluding the first revealed cell.

    Parameters
    ----------
    rows : int
        Total row count.
    columns : int
        Total column count.
    mine_count : int
        Number of mines to place.
    safe_cell : Coordinate
        Cell that must remain safe.
    seed : int
        Deterministic placement seed.

    Returns
    -------
    MineBoard
        Deterministic mine layout excluding `safe_cell`.
    """
    safe_row, safe_col = safe_cell
    if not (0 <= safe_row < rows and 0 <= safe_col < columns):
        raise ValueError("safe_cell must be inside the Minesweeper board.")
    if mine_count < 0:
        raise ValueError("Minesweeper mine_count must be non-negative.")
    if mine_count >= rows * columns:
        raise ValueError("mine_count must leave at least one safe cell.")

    eligible_cells = [
        (row_index, col_index)
        for row_index in range(rows)
        for col_index in range(columns)
        if (row_index, col_index) != safe_cell
    ]
    chosen_cells = set(Random(seed).sample(eligible_cells, mine_count))
    return tuple(
        tuple((row_index, col_index) in chosen_cells for col_index in range(columns))
        for row_index in range(rows)
    )


def reveal_cells(
    *,
    board: MineBoard,
    counts: AdjacentMineBoard,
    revealed: BoolBoard,
    flagged: BoolBoard,
    start: Coordinate,
) -> tuple[BoolBoard, tuple[Coordinate, ...]]:
    """Reveal one cell and any safe flood-fill region implied by it.

    Parameters
    ----------
    board : MineBoard
        Hidden mine layout.
    counts : AdjacentMineBoard
        Adjacent-mine count board derived from `board`.
    revealed : BoolBoard
        Current revealed-cell mask.
    flagged : BoolBoard
        Current flagged-cell mask.
    start : Coordinate
        Zero-based target cell to reveal.

    Returns
    -------
    tuple[BoolBoard, tuple[Coordinate, ...]]
        Updated revealed mask and the coordinates newly revealed by the move.
    """
    row, col = start
    if flagged[row][col] or revealed[row][col]:
        return revealed, ()

    updated_rows = [list(current_row) for current_row in revealed]
    if board[row][col]:
        updated_rows[row][col] = True
        return tuple(tuple(current_row) for current_row in updated_rows), ((row, col),)

    queue: deque[Coordinate] = deque([start])
    seen = {start}
    newly_revealed: list[Coordinate] = []
    rows = len(board)
    columns = len(board[0])
    while queue:
        current_row, current_col = queue.popleft()
        if flagged[current_row][current_col] or updated_rows[current_row][current_col]:
            continue

        updated_rows[current_row][current_col] = True
        newly_revealed.append((current_row, current_col))
        if counts[current_row][current_col] != 0:
            continue

        for neighbor_row, neighbor_col in neighbors(
            rows=rows,
            columns=columns,
            row=current_row,
            col=current_col,
        ):
            if flagged[neighbor_row][neighbor_col]:
                continue
            if updated_rows[neighbor_row][neighbor_col]:
                continue
            if board[neighbor_row][neighbor_col]:
                continue
            neighbor = (neighbor_row, neighbor_col)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)

    return tuple(tuple(current_row) for current_row in updated_rows), tuple(
        newly_revealed
    )


def _normalize_mine_cell(*, cell: bool | str | int) -> bool:
    """Normalize one mine-layout cell into a boolean."""
    if isinstance(cell, bool):
        return cell
    if isinstance(cell, int):
        if cell not in {0, 1}:
            raise ValueError("Integer Minesweeper cells must be 0 or 1.")
        return bool(cell)
    normalized = cell.strip().lower()
    if normalized in {"*", "m", "x", "1"}:
        return True
    if normalized in {".", "_", "-", "0"}:
        return False
    raise ValueError(
        "String Minesweeper cells must be one of '.', '*', 'm', 'x', '0', or '1'."
    )
