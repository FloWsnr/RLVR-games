"""Pure board mechanics for 2048."""

from dataclasses import dataclass
from random import Random
from typing import Sequence

from rlvr_games.games.game2048.actions import ACTION_ORDER, MoveDirection

Board = tuple[tuple[int, ...], ...]
Position = tuple[int, int]


def is_power_of_two(value: int) -> bool:
    """Return whether the supplied integer is a positive power of two.

    Parameters
    ----------
    value : int
        Integer to validate.

    Returns
    -------
    bool
        `True` when `value` is positive and contains exactly one set bit.
    """
    return value > 0 and (value & (value - 1)) == 0


def make_empty_board(*, size: int) -> Board:
    """Return an empty square 2048 board.

    Parameters
    ----------
    size : int
        Number of rows and columns in the square board.

    Returns
    -------
    Board
        Immutable square board filled with zeros.

    Raises
    ------
    ValueError
        If `size` is smaller than `2`.
    """
    if size < 2:
        raise ValueError("2048 boards must be at least 2x2.")
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))


def normalize_board(*, rows: Sequence[Sequence[int]]) -> Board:
    """Validate and normalize a board-like nested sequence.

    Parameters
    ----------
    rows : Sequence[Sequence[int]]
        Nested integer rows representing a 2048 board.

    Returns
    -------
    Board
        Immutable square board tuple.

    Raises
    ------
    ValueError
        If the supplied rows are empty, not square, or contain invalid tile
        values.
    """
    if not rows:
        raise ValueError("2048 boards must contain at least one row.")

    size = len(rows)
    if size < 2:
        raise ValueError("2048 boards must be at least 2x2.")

    normalized_rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(rows):
        if len(row) != size:
            raise ValueError("2048 boards must be square.")
        normalized_row: list[int] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "2048 board values must be integers; "
                    f"received {value!r} at ({row_index}, {column_index})."
                )
            if value < 0:
                raise ValueError("2048 board values must be non-negative.")
            if value != 0 and not is_power_of_two(value):
                raise ValueError(
                    "2048 board values must be zero or powers of two; "
                    f"received {value!r} at ({row_index}, {column_index})."
                )
            normalized_row.append(value)
        normalized_rows.append(tuple(normalized_row))
    return tuple(normalized_rows)


def empty_positions(*, board: Board) -> tuple[Position, ...]:
    """Return the empty cell coordinates of a board in row-major order.

    Parameters
    ----------
    board : Board
        Board to inspect.

    Returns
    -------
    tuple[Position, ...]
        Empty-cell coordinates as ``(row, column)`` tuples.
    """
    return tuple(
        (row_index, column_index)
        for row_index, row in enumerate(board)
        for column_index, value in enumerate(row)
        if value == 0
    )


@dataclass(slots=True, frozen=True)
class MergeSummary:
    """One merge event produced by a verified 2048 move.

    Attributes
    ----------
    row : int
        Zero-based row index of the merged destination tile.
    col : int
        Zero-based column index of the merged destination tile.
    value : int
        Value of the merged destination tile.
    sources : tuple[Position, Position]
        Source coordinates of the two tiles that merged.
    """

    row: int
    col: int
    value: int
    sources: tuple[Position, Position]


@dataclass(slots=True, frozen=True)
class SpawnSummary:
    """One random tile spawn event produced after a legal 2048 move.

    Attributes
    ----------
    row : int
        Zero-based row index of the spawned tile.
    col : int
        Zero-based column index of the spawned tile.
    value : int
        Value of the spawned tile, typically ``2`` or ``4``.
    """

    row: int
    col: int
    value: int


@dataclass(slots=True, frozen=True)
class MoveSummary:
    """Pure move result before the post-move random tile spawn.

    Attributes
    ----------
    board : Board
        Board after all slide and merge mechanics have been applied.
    score_gain : int
        Score increment produced by the merges in the move.
    moved : bool
        Whether the direction changed the board.
    merges : tuple[MergeSummary, ...]
        Detailed merge events produced by the move.
    """

    board: Board
    score_gain: int
    moved: bool
    merges: tuple[MergeSummary, ...]


@dataclass(slots=True, frozen=True)
class _CollapsedLine:
    """Internal collapsed-line result for one row or column traversal."""

    values: tuple[int, ...]
    score_gain: int
    merges: tuple[MergeSummary, ...]


def apply_move(*, board: Board, direction: MoveDirection) -> MoveSummary:
    """Apply one pure slide/merge move to a board.

    Parameters
    ----------
    board : Board
        Canonical board before the move.
    direction : MoveDirection
        Direction to apply.

    Returns
    -------
    MoveSummary
        Board after movement and merges, excluding the random tile spawn step.
    """
    size = len(board)
    mutable_rows = [list(row) for row in board]
    score_gain = 0
    merges: list[MergeSummary] = []
    moved = False

    for line_index in range(size):
        positions = _line_positions(size=size, index=line_index, direction=direction)
        values = tuple(board[row][col] for row, col in positions)
        collapsed = _collapse_line(values=values, positions=positions)
        if collapsed.values != values:
            moved = True
        score_gain += collapsed.score_gain
        merges.extend(collapsed.merges)
        for (row, col), value in zip(positions, collapsed.values, strict=True):
            mutable_rows[row][col] = value

    return MoveSummary(
        board=tuple(tuple(row) for row in mutable_rows),
        score_gain=score_gain,
        moved=moved,
        merges=tuple(merges),
    )


def legal_action_labels(*, board: Board) -> tuple[str, ...]:
    """Return the legal action labels for a board.

    Parameters
    ----------
    board : Board
        Canonical board to inspect.

    Returns
    -------
    tuple[str, ...]
        Canonical direction labels whose moves would change the board.
    """
    return tuple(
        direction.value
        for direction in ACTION_ORDER
        if apply_move(board=board, direction=direction).moved
    )


def spawn_random_tile(*, board: Board, rng: Random) -> tuple[Board, SpawnSummary]:
    """Spawn one random tile on a board using the supplied RNG.

    Parameters
    ----------
    board : Board
        Board on which a new tile should be inserted.
    rng : Random
        Random generator whose state is part of the canonical game state.

    Returns
    -------
    tuple[Board, SpawnSummary]
        Updated board and metadata describing the spawned tile.

    Raises
    ------
    ValueError
        If the supplied board has no empty cells.
    """
    available_positions = empty_positions(board=board)
    if not available_positions:
        raise ValueError("Cannot spawn a tile on a full 2048 board.")

    value = 2 if rng.random() < 0.9 else 4
    row_index, column_index = available_positions[
        rng.randrange(len(available_positions))
    ]

    mutable_rows = [list(row) for row in board]
    mutable_rows[row_index][column_index] = value
    next_board = tuple(tuple(row) for row in mutable_rows)
    return next_board, SpawnSummary(
        row=row_index,
        col=column_index,
        value=value,
    )


def max_tile(*, board: Board) -> int:
    """Return the largest tile value present on a board.

    Parameters
    ----------
    board : Board
        Board to inspect.

    Returns
    -------
    int
        Largest tile value on the board.
    """
    return max(max(row) for row in board)


def _line_positions(
    *,
    size: int,
    index: int,
    direction: MoveDirection,
) -> tuple[Position, ...]:
    """Return one traversal line in the movement order for a direction.

    Parameters
    ----------
    size : int
        Board size.
    index : int
        Row or column index for the line.
    direction : MoveDirection
        Direction whose movement order should be used.

    Returns
    -------
    tuple[Position, ...]
        Board coordinates ordered from the leading edge toward the trailing
        edge for the supplied direction.
    """
    if direction == MoveDirection.LEFT:
        return tuple((index, column_index) for column_index in range(size))
    if direction == MoveDirection.RIGHT:
        return tuple((index, column_index) for column_index in range(size - 1, -1, -1))
    if direction == MoveDirection.UP:
        return tuple((row_index, index) for row_index in range(size))
    return tuple((row_index, index) for row_index in range(size - 1, -1, -1))


def _collapse_line(
    *,
    values: tuple[int, ...],
    positions: tuple[Position, ...],
) -> _CollapsedLine:
    """Collapse one traversal line according to 2048 merge rules.

    Parameters
    ----------
    values : tuple[int, ...]
        Tile values ordered in the direction of travel.
    positions : tuple[Position, ...]
        Board coordinates corresponding to `values`.

    Returns
    -------
    _CollapsedLine
        Collapsed line values, score gain, and merge metadata.
    """
    non_zero_tiles = [
        (value, position)
        for value, position in zip(values, positions, strict=True)
        if value != 0
    ]
    collapsed_values: list[int] = []
    merges: list[MergeSummary] = []
    score_gain = 0
    tile_index = 0

    while tile_index < len(non_zero_tiles):
        value, position = non_zero_tiles[tile_index]
        target_position = positions[len(collapsed_values)]
        if (
            tile_index + 1 < len(non_zero_tiles)
            and non_zero_tiles[tile_index + 1][0] == value
        ):
            next_position = non_zero_tiles[tile_index + 1][1]
            merged_value = value * 2
            collapsed_values.append(merged_value)
            score_gain += merged_value
            merges.append(
                MergeSummary(
                    row=target_position[0],
                    col=target_position[1],
                    value=merged_value,
                    sources=(position, next_position),
                )
            )
            tile_index += 2
            continue

        collapsed_values.append(value)
        tile_index += 1

    padded_values = tuple(
        collapsed_values + [0] * (len(values) - len(collapsed_values))
    )
    return _CollapsedLine(
        values=padded_values,
        score_gain=score_gain,
        merges=tuple(merges),
    )
