"""2048 scenario initializers."""

from dataclasses import dataclass
from typing import Sequence

from rlvr_games.games.game2048.chance import Game2048ChanceModel
from rlvr_games.games.game2048.engine import (
    Board,
    make_empty_board,
    normalize_board,
)
from rlvr_games.games.game2048.state import Game2048State

STANDARD_2048_SIZE = 4
STANDARD_2048_TARGET = 2048
STANDARD_START_TILE_COUNT = 2


@dataclass(slots=True)
class RandomStartScenario:
    """Scenario that initializes 2048 from a random seeded start board.

    Attributes
    ----------
    size : int
        Board dimension to initialize.
    target_value : int
        Tile value that counts as a win.
    start_tile_count : int
        Number of random tiles to spawn at reset time.
    chance_model : Game2048ChanceModel
        Chance-model helper responsible for deterministic random tile spawns.
    """

    size: int
    target_value: int
    start_tile_count: int
    chance_model: Game2048ChanceModel

    def __post_init__(self) -> None:
        """Validate scenario configuration.

        Raises
        ------
        ValueError
            If the board size or start tile count is invalid.
        """
        if self.size < 2:
            raise ValueError("2048 boards must be at least 2x2.")
        if self.start_tile_count <= 0:
            raise ValueError("2048 start_tile_count must be positive.")
        if self.start_tile_count > self.size * self.size:
            raise ValueError("2048 start_tile_count cannot exceed board capacity.")

    def reset(self, *, seed: int) -> tuple[Game2048State, dict[str, object]]:
        """Create a fresh random 2048 episode.

        Parameters
        ----------
        seed : int
            Explicit seed used for deterministic tile placement.

        Returns
        -------
        tuple[Game2048State, dict[str, object]]
            Canonical initial state and public-safe reset metadata describing
            the initial board and spawned starting tiles.
        """
        board = make_empty_board(size=self.size)
        rng_state = self.chance_model.initial_rng_state(seed=seed)
        spawned_tiles: list[dict[str, int]] = []
        for _ in range(self.start_tile_count):
            spawn_transition = self.chance_model.spawn_tile(
                board=board,
                rng_state=rng_state,
            )
            board = spawn_transition.board
            rng_state = spawn_transition.rng_state
            spawned_tile = spawn_transition.spawned_tile
            spawned_tiles.append(
                {
                    "row": spawned_tile.row,
                    "col": spawned_tile.col,
                    "value": spawned_tile.value,
                }
            )

        state = Game2048State(
            board=board,
            score=0,
            move_count=0,
            target_value=self.target_value,
            rng_state=rng_state,
        )
        return state, {
            "scenario": "random_start",
            "size": state.size,
            "target_value": state.target_value,
            "start_tile_count": self.start_tile_count,
            "initial_board": state.board,
            "spawned_tiles": tuple(spawned_tiles),
        }


@dataclass(slots=True)
class FixedBoardScenario:
    """Scenario that initializes 2048 from an explicit board position.

    Attributes
    ----------
    initial_board : Board
        Explicit starting board in row-major order.
    initial_score : int
        Score already accumulated before the scenario starts.
    initial_move_count : int
        Accepted move count already accumulated before the scenario starts.
    target_value : int
        Tile value that counts as a win.
    chance_model : Game2048ChanceModel
        Chance-model helper responsible for deterministic future tile spawns.
    """

    initial_board: Board
    initial_score: int
    initial_move_count: int
    target_value: int
    chance_model: Game2048ChanceModel

    def __post_init__(self) -> None:
        """Normalize and validate the configured starting board."""
        self.initial_board = normalize_board(rows=self.initial_board)
        if self.initial_score < 0:
            raise ValueError("2048 initial_score must be non-negative.")
        if self.initial_move_count < 0:
            raise ValueError("2048 initial_move_count must be non-negative.")

    def reset(self, *, seed: int) -> tuple[Game2048State, dict[str, object]]:
        """Create a fresh 2048 episode from the configured board.

        Parameters
        ----------
        seed : int
            Explicit seed used to initialize future random tile spawns.

        Returns
        -------
        tuple[Game2048State, dict[str, object]]
            Canonical initial state and public-safe metadata describing the
            configured board, score, and move count.
        """
        state = Game2048State(
            board=self.initial_board,
            score=self.initial_score,
            move_count=self.initial_move_count,
            target_value=self.target_value,
            rng_state=self.chance_model.initial_rng_state(seed=seed),
        )
        return state, {
            "scenario": "fixed_board",
            "size": state.size,
            "target_value": state.target_value,
            "initial_board": state.board,
            "initial_score": state.score,
            "initial_move_count": state.move_count,
        }


def normalize_initial_board(*, board: Sequence[Sequence[int]]) -> Board:
    """Normalize a programmatic or CLI-provided initial board.

    Parameters
    ----------
    board : Sequence[Sequence[int]]
        Board-like nested integer sequence.

    Returns
    -------
    Board
        Immutable normalized board.
    """
    return normalize_board(rows=board)
