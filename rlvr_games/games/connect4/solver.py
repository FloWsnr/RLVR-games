"""BitBully-backed solver helpers for standard Connect 4."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

from bitbully import BitBully, Board as BitBullyBoard

from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.variant import (
    validate_standard_connect4_configuration,
)
from rlvr_games.games.connect4.state import Connect4State

type Connect4Perspective = Literal["x", "o"]
type BitBullyOpeningBook = Literal["default", "8-ply", "12-ply", "12-ply-dist"]

_BITBULLY_CELL_VALUES = {
    ".": 0,
    "x": 1,
    "o": 2,
}


class Connect4MoveScorer(Protocol):
    """Protocol for scoring Connect 4 moves from one side's perspective."""

    def score_action(
        self,
        *,
        state: Connect4State,
        action: Connect4Action,
        perspective: Connect4Perspective,
    ) -> float:
        """Return the scalar score for one legal move."""
        ...

    def score_actions(
        self,
        *,
        state: Connect4State,
        perspective: Connect4Perspective,
    ) -> dict[str, float]:
        """Return scalar scores for every legal move."""
        ...


@dataclass(slots=True)
class BitBullySolver:
    """Score and select standard Connect 4 moves with BitBully.

    Attributes
    ----------
    opening_book : {"default", "8-ply", "12-ply", "12-ply-dist"} | None
        Opening book to load. The default uses BitBully's packaged
        ``"12-ply-dist"`` alias for fast early-game lookups.
    max_depth : int
        Optional BitBully search depth limit in plies. ``-1`` keeps the full
        search enabled.
    """

    opening_book: BitBullyOpeningBook | None = "default"
    max_depth: int = -1
    _solver: BitBully = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Construct the backing BitBully engine.

        Raises
        ------
        ValueError
            If `max_depth` is smaller than ``-1`` or if the requested opening
            book did not load.
        """
        if self.max_depth < -1:
            raise ValueError("BitBullySolver max_depth must be >= -1.")

        self._solver = BitBully(
            opening_book=self.opening_book,
            max_depth=self.max_depth,
        )
        if self.opening_book is not None and not self._solver.is_book_loaded():
            raise ValueError(
                "BitBullySolver requested an opening book, but no opening book "
                "was loaded."
            )

    def score_action(
        self,
        *,
        state: Connect4State,
        action: Connect4Action,
        perspective: Connect4Perspective,
    ) -> float:
        """Return the BitBully score for one legal move.

        Parameters
        ----------
        state : Connect4State
            Non-terminal standard Connect 4 state to analyse.
        action : Connect4Action
            Legal move to score.
        perspective : {"x", "o"}
            Side whose point of view determines the score sign.

        Returns
        -------
        float
            BitBully move score from `perspective`.
        """
        ensure_bitbully_supported_state(state=state)
        if state.is_terminal:
            raise ValueError("BitBully move scoring requires a non-terminal state.")
        if action.label not in state.legal_actions:
            raise ValueError(f"Cannot score illegal Connect 4 move {action.label!r}.")

        score = self._solver.score_move(
            bitbully_board_from_state(state=state),
            action.column,
        )
        return float(
            _score_from_perspective(
                score=score,
                current_player=state.current_player,
                perspective=perspective,
            )
        )

    def validate_state(self, *, state: Connect4State) -> None:
        """Validate that `state` is compatible with BitBully.

        Parameters
        ----------
        state : Connect4State
            Canonical state to validate.
        """
        ensure_bitbully_supported_state(state=state)

    def score_actions(
        self,
        *,
        state: Connect4State,
        perspective: Connect4Perspective,
    ) -> dict[str, float]:
        """Return BitBully scores for every legal move.

        Parameters
        ----------
        state : Connect4State
            Standard Connect 4 state to analyse.
        perspective : {"x", "o"}
            Side whose point of view determines the score sign.

        Returns
        -------
        dict[str, float]
            One-based column labels mapped to BitBully scores in descending
            order for `perspective`.
        """
        ensure_bitbully_supported_state(state=state)
        if state.is_terminal:
            return {}

        raw_scores = self._solver.score_all_moves(
            bitbully_board_from_state(state=state)
        )
        scored_actions = {
            str(column_index + 1): float(
                _score_from_perspective(
                    score=score,
                    current_player=state.current_player,
                    perspective=perspective,
                )
            )
            for column_index, score in raw_scores.items()
        }
        return dict(
            sorted(
                scored_actions.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def select_action(
        self,
        *,
        state: Connect4State,
    ) -> Connect4Action:
        """Return BitBully's best legal move for `state`.

        Parameters
        ----------
        state : Connect4State
            Non-terminal standard Connect 4 state to analyse.

        Returns
        -------
        Connect4Action
            Selected legal move.
        """
        ensure_bitbully_supported_state(state=state)
        if state.is_terminal:
            raise ValueError("BitBully move selection requires a non-terminal state.")

        best_column = self._solver.best_move(bitbully_board_from_state(state=state))
        return Connect4Action(column=best_column)


def ensure_bitbully_supported_state(*, state: Connect4State) -> None:
    """Validate that `state` matches BitBully's supported game variant.

    Parameters
    ----------
    state : Connect4State
        Canonical state to validate.

    Raises
    ------
    ValueError
        If `state` is not the standard 7x6 connect-4 game.
    """
    validate_standard_connect4_configuration(
        rows=state.rows,
        columns=state.columns,
        connect_length=state.connect_length,
    )


def bitbully_board_from_state(*, state: Connect4State) -> BitBullyBoard:
    """Convert one canonical Connect 4 state into a BitBully board.

    Parameters
    ----------
    state : Connect4State
        Canonical state to convert.

    Returns
    -------
    BitBullyBoard
        Equivalent BitBully board representation.
    """
    ensure_bitbully_supported_state(state=state)
    row_major_board = [
        [_BITBULLY_CELL_VALUES[cell] for cell in row] for row in state.board
    ]
    return BitBullyBoard.from_array(row_major_board)


def _score_from_perspective(
    *,
    score: int,
    current_player: str,
    perspective: Connect4Perspective,
) -> int:
    """Convert one side-to-move BitBully score into a fixed perspective."""
    if current_player not in {"x", "o"}:
        raise ValueError("BitBully scores are only defined for non-terminal states.")
    if perspective == current_player:
        return score
    return -score


__all__ = [
    "BitBullyOpeningBook",
    "BitBullySolver",
    "Connect4MoveScorer",
    "Connect4Perspective",
    "bitbully_board_from_state",
    "ensure_bitbully_supported_state",
]
