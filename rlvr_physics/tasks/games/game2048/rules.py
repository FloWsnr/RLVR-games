"""Authoritative 2048 board rules."""

from random import Random

from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.games.game2048.constants import GAME2048_ACTIONS
from rlvr_physics.tasks.games.game2048.types import Board, MoveResult, SpawnTape


def move_board(board: Board, action: str) -> MoveResult:
    """Apply one 2048 move without spawning a new tile."""

    if action not in GAME2048_ACTIONS:
        raise ValueError(f"unknown 2048 action: {action}")
    if action == "left":
        moved_rows = [_merge_line(row) for row in board]
        moved_board = tuple(result[0] for result in moved_rows)
        score_delta = sum(result[1] for result in moved_rows)
    elif action == "right":
        moved_rows = [_merge_line(tuple(reversed(row))) for row in board]
        moved_board = tuple(tuple(reversed(result[0])) for result in moved_rows)
        score_delta = sum(result[1] for result in moved_rows)
    elif action == "up":
        columns = _transpose(board)
        moved_columns = [_merge_line(column) for column in columns]
        moved_board = _transpose(tuple(result[0] for result in moved_columns))
        score_delta = sum(result[1] for result in moved_columns)
    else:
        columns = _transpose(board)
        moved_columns = [_merge_line(tuple(reversed(column))) for column in columns]
        moved_board = _transpose(
            tuple(tuple(reversed(result[0])) for result in moved_columns)
        )
        score_delta = sum(result[1] for result in moved_columns)
    return MoveResult(
        board=moved_board, score_delta=score_delta, changed=moved_board != board
    )


def legal_2048_actions(board: Board) -> tuple[str, ...]:
    """Return actions that change the board."""

    legal: list[str] = []
    for action in GAME2048_ACTIONS:
        if move_board(board, action).changed:
            legal.append(action)
    return tuple(legal)


def make_spawn_tape(seed: int, length: int, size: int) -> SpawnTape:
    """Create a deterministic spawn tape for one 2048 instance.

    Parameters
    ----------
    seed:
        Seed controlling spawn positions and tile values.
    length:
        Number of spawn entries to create.
    size:
        Board width and height.
    """

    rng = Random(seed)
    tape: list[tuple[int, int]] = []
    cells = size * size
    for _ in range(length):
        empty_index = rng.randrange(cells)
        value = 4 if rng.random() < 0.1 else 2
        tape.append((empty_index, value))
    return tuple(tape)


def apply_spawn(board: Board, spawn: tuple[int, int]) -> tuple[Board, bool]:
    """Apply one spawn tape entry to a board.

    Parameters
    ----------
    board:
        Board before spawning.
    spawn:
        Pair of empty-cell index and tile value.
    """

    empties: list[tuple[int, int]] = []
    for row_index, row in enumerate(board):
        for col_index, value in enumerate(row):
            if value == 0:
                empties.append((row_index, col_index))
    if not empties:
        return board, False
    empty_index, value = spawn
    row_index, col_index = empties[empty_index % len(empties)]
    mutable = [list(row) for row in board]
    mutable[row_index][col_index] = value
    return tuple(tuple(row) for row in mutable), True


def parse_action(submission: TaskSubmission) -> str:
    """Extract a normalized 2048 action from a submission.

    Parameters
    ----------
    submission:
        Raw task submission.
    """

    if submission.kind == "action":
        parsed_action = submission.parsed.get("action")
        if isinstance(parsed_action, str):
            return parsed_action.strip().lower()
    return submission.raw.strip().lower()


def terminal_reason(
    terminal: bool, truncated: bool, max_tile: int, target_tile: int
) -> str:
    """Return the public reason for a 2048 step status."""

    if terminal and max_tile >= target_tile:
        return "target_tile_reached"
    if terminal:
        return "no_legal_moves"
    if truncated:
        return "max_turns"
    return "continue"


def require_board(value: object, name: str) -> Board:
    """Return a validated immutable 2048 board."""

    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple board")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, tuple):
            raise TypeError(f"{name} rows must be tuples")
        ints: list[int] = []
        for item in row:
            if not isinstance(item, int):
                raise TypeError(f"{name} cells must be integers")
            ints.append(item)
        rows.append(tuple(ints))
    return tuple(rows)


def require_spawn_tape(value: object, name: str) -> SpawnTape:
    """Return a validated immutable spawn tape."""

    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    tape: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} entries must be integer pairs")
        empty_index, tile_value = item
        if not isinstance(empty_index, int) or not isinstance(tile_value, int):
            raise TypeError(f"{name} entries must be integer pairs")
        tape.append((empty_index, tile_value))
    return tuple(tape)


def _merge_line(line: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    values = [value for value in line if value != 0]
    merged: list[int] = []
    score_delta = 0
    index = 0
    while index < len(values):
        value = values[index]
        if index + 1 < len(values) and values[index + 1] == value:
            merged_value = value * 2
            merged.append(merged_value)
            score_delta += merged_value
            index += 2
        else:
            merged.append(value)
            index += 1
    merged.extend(0 for _ in range(len(line) - len(merged)))
    return tuple(merged), score_delta


def _transpose(board: Board) -> Board:
    return tuple(
        tuple(row[column_index] for row in board)
        for column_index in range(len(board[0]))
    )
