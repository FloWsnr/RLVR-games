"""Canonical 2048 state types."""

from dataclasses import dataclass, field
from typing import Any

from rlvr_games.games.game2048.engine import (
    Board,
    empty_positions,
    legal_action_labels,
    max_tile,
    normalize_board,
    is_power_of_two,
)


@dataclass(slots=True, frozen=True)
class Game2048Outcome:
    """Terminal outcome summary for a canonical 2048 position.

    Attributes
    ----------
    is_terminal : bool
        Whether the state ends the episode under the configured target and move
        availability rules.
    won : bool
        Whether the terminal state reached or exceeded the configured target
        tile value.
    termination : str | None
        Structured terminal reason. Supported values are ``"target_tile"`` and
        ``"no_moves"``.
    """

    is_terminal: bool
    won: bool
    termination: str | None = None

    def __post_init__(self) -> None:
        """Validate that terminal metadata is internally coherent.

        Raises
        ------
        ValueError
            If terminal flags and terminal metadata disagree.
        """
        if self.is_terminal and self.termination is None:
            raise ValueError("Terminal 2048 outcomes require a termination reason.")
        if not self.is_terminal and (self.won or self.termination is not None):
            raise ValueError(
                "Non-terminal 2048 outcomes must not include terminal metadata."
            )
        if self.termination == "target_tile" and not self.won:
            raise ValueError("target_tile termination requires a winning outcome.")
        if self.termination == "no_moves" and self.won:
            raise ValueError("no_moves termination cannot also be a winning outcome.")

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
            "won": self.won,
            "termination": self.termination,
        }


@dataclass(init=False, slots=True, frozen=True)
class Game2048State:
    """Canonical 2048 state.

    Attributes
    ----------
    board : Board
        Immutable square grid of tile values in row-major order.
    score : int
        Cumulative score accumulated from all merge values so far.
    move_count : int
        Number of accepted moves applied in the episode so far.
    target_value : int
        Tile value that counts as a win.
    size : int
        Board dimension.
    legal_actions : tuple[str, ...]
        Canonical legal direction labels for the current board.
    empty_cell_count : int
        Number of empty cells currently on the board.
    max_tile : int
        Largest tile currently present on the board.
    outcome : Game2048Outcome
        Terminal outcome summary for the current state.
    """

    board: Board
    score: int
    move_count: int
    target_value: int
    _rng_state: tuple[Any, ...] = field(repr=False)
    size: int = field(init=False)
    legal_actions: tuple[str, ...] = field(init=False)
    empty_cell_count: int = field(init=False)
    max_tile: int = field(init=False)
    outcome: Game2048Outcome = field(init=False)

    def __init__(
        self,
        *,
        board: Board,
        score: int,
        move_count: int,
        target_value: int,
        rng_state: tuple[Any, ...],
    ) -> None:
        """Create a canonical 2048 state.

        Parameters
        ----------
        board : Board
            Immutable or board-like nested sequence representing the grid.
        score : int
            Cumulative score so far.
        move_count : int
            Number of accepted moves already applied.
        target_value : int
            Tile value that should terminate the episode as a win.
        rng_state : object
            Python `random.Random` internal state used for deterministic future
            tile spawns.

        Raises
        ------
        ValueError
            If the board, score, move count, or target value is invalid.
        """
        normalized_board = normalize_board(rows=board)
        if score < 0:
            raise ValueError("2048 scores must be non-negative.")
        if move_count < 0:
            raise ValueError("2048 move counts must be non-negative.")
        if target_value < 2 or not is_power_of_two(target_value):
            raise ValueError("2048 target_value must be a power of two >= 2.")

        current_max_tile = max_tile(board=normalized_board)
        current_empty_cell_count = len(empty_positions(board=normalized_board))
        if current_max_tile >= target_value:
            current_legal_actions = ()
            outcome = Game2048Outcome(
                is_terminal=True,
                won=True,
                termination="target_tile",
            )
        else:
            current_legal_actions = legal_action_labels(board=normalized_board)
            if not current_legal_actions:
                outcome = Game2048Outcome(
                    is_terminal=True,
                    won=False,
                    termination="no_moves",
                )
            else:
                outcome = Game2048Outcome(
                    is_terminal=False,
                    won=False,
                    termination=None,
                )

        object.__setattr__(self, "board", normalized_board)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "move_count", move_count)
        object.__setattr__(self, "target_value", target_value)
        object.__setattr__(self, "_rng_state", rng_state)
        object.__setattr__(self, "size", len(normalized_board))
        object.__setattr__(self, "legal_actions", current_legal_actions)
        object.__setattr__(self, "empty_cell_count", current_empty_cell_count)
        object.__setattr__(self, "max_tile", current_max_tile)
        object.__setattr__(self, "outcome", outcome)

    @property
    def rng_state(self) -> tuple[Any, ...]:
        """Return the deterministic RNG state for future spawns.

        Returns
        -------
        tuple[Any, ...]
            Python `random.Random` internal state tuple.
        """
        return self._rng_state

    @property
    def legal_action_count(self) -> int:
        """Return the number of legal directions for the current board.

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
            `True` when the outcome marks the board as terminal.
        """
        return self.outcome.is_terminal
