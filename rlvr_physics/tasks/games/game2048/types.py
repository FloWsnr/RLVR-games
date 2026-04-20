"""Types for seeded 2048."""

from dataclasses import dataclass

Board = tuple[tuple[int, ...], ...]
SpawnTape = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Game2048State:
    """Canonical 2048 runtime state."""

    board: Board
    score: int
    turns: int
    max_tile: int
    spawn_index: int


@dataclass(frozen=True)
class MoveResult:
    """Result of applying one 2048 action before tile spawn."""

    board: Board
    score_delta: int
    changed: bool
