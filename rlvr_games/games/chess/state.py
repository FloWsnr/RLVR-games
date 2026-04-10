"""Canonical chess state types."""

from dataclasses import dataclass, field
from typing import Any, Self

import chess


def repetition_key_from_board(board: chess.Board) -> str:
    """Return the repetition-significant part of a chess position.

    Parameters
    ----------
    board : chess.Board
        Board whose position should be reduced to the fields relevant for
        repetition detection.

    Returns
    -------
    str
        String containing piece placement, side to move, castling rights, and
        a legal en passant target square when one exists.
    """
    side_to_move = "w" if board.turn == chess.WHITE else "b"
    castling = board.castling_xfen()
    en_passant_square = "-"
    if board.has_legal_en_passant() and board.ep_square is not None:
        en_passant_square = chess.square_name(board.ep_square)
    return " ".join(
        (
            board.board_fen(),
            side_to_move,
            castling,
            en_passant_square,
        )
    )


def winner_name(winner: bool | None) -> str | None:
    """Return the structured winner label used in chess metadata.

    Parameters
    ----------
    winner : bool | None
        Winner flag returned by `python-chess`, or `None` for a draw.

    Returns
    -------
    str | None
        `"white"`, `"black"`, or `None` when the game is drawn.
    """
    if winner is None:
        return None
    return "white" if winner == chess.WHITE else "black"


@dataclass(slots=True, frozen=True)
class ChessOutcome:
    """Terminal outcome summary for a canonical chess position.

    Attributes
    ----------
    is_terminal : bool
        Whether the position ends the episode under the chess rules.
    result : str | None
        PGN-style result string such as `"1-0"` or `"1/2-1/2"`.
    winner : str | None
        Structured winner label, or `None` for drawn games.
    termination : str | None
        Lowercase termination reason, for example `"checkmate"` or
        `"threefold_repetition"`.
    """

    is_terminal: bool
    result: str | None = None
    winner: str | None = None
    termination: str | None = None

    def __post_init__(self) -> None:
        """Validate that terminal metadata is internally coherent.

        Raises
        ------
        ValueError
            If terminal metadata is missing for terminal outcomes or supplied
            for non-terminal outcomes.
        """
        has_terminal_metadata = (
            self.result is not None
            or self.winner is not None
            or self.termination is not None
        )
        if self.is_terminal and (self.result is None or self.termination is None):
            raise ValueError("Terminal chess outcomes require result metadata.")
        if not self.is_terminal and has_terminal_metadata:
            raise ValueError(
                "Non-terminal chess outcomes must not include terminal metadata."
            )

    def metadata(self) -> dict[str, object]:
        """Return dictionary metadata for terminal chess outcomes.

        Returns
        -------
        dict[str, object]
            Empty dictionary for non-terminal positions, otherwise the result,
            winner, and termination metadata fields.
        """
        if not self.is_terminal:
            return {}
        return {
            "result": self.result,
            "winner": self.winner,
            "termination": self.termination,
        }


def outcome_from_board(board: chess.Board, repetition_count: int) -> ChessOutcome:
    """Return the terminal outcome implied by the supplied board.

    Parameters
    ----------
    board : chess.Board
        Board to inspect.
    repetition_count : int
        Number of times the current repetition-significant position has been
        observed in the episode so far.

    Returns
    -------
    ChessOutcome
        Terminal outcome summary for the current position.
    """
    if repetition_count >= 3:
        return ChessOutcome(
            is_terminal=True,
            result="1/2-1/2",
            winner=None,
            termination="threefold_repetition",
        )

    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return ChessOutcome(
            is_terminal=True,
            result=board.result(claim_draw=False),
            winner=winner_name(outcome.winner),
            termination=outcome.termination.name.lower(),
        )

    if board.can_claim_fifty_moves():
        return ChessOutcome(
            is_terminal=True,
            result="1/2-1/2",
            winner=None,
            termination="fifty_moves",
        )

    return ChessOutcome(is_terminal=False)


@dataclass(slots=True)
class ChessState:
    """Canonical chess state.

    The state stores both the replayable FEN string and a cached
    `python-chess` board plus derived summary fields so the backend and
    renderer do not need to repeatedly reparse the same position during
    rollout-heavy workloads.

    Attributes
    ----------
    fen : str
        Full FEN string describing the current board position.
    repetition_counts : dict[str, int]
        Counts for repetition-significant positions encountered in the current
        episode. These counts are used to detect threefold repetition claims.
    metadata : dict[str, Any]
        Free-form game-specific metadata that should travel with the state
        without becoming part of the authoritative chess rules payload.
    board : chess.Board
        Cached `python-chess` board representing `fen`.
    repetition_key : str
        Repetition-significant position key for the current board.
    repetition_count : int
        Recorded count for the current `repetition_key`.
    side_to_move : str
        Structured label for the player whose turn it is.
    legal_actions : tuple[str, ...]
        Sorted legal UCI move strings for the current position.
    is_check : bool
        Whether the side to move is currently in check.
    outcome : ChessOutcome
        Terminal outcome summary for the current position.
    """

    fen: str
    repetition_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    board: chess.Board = field(init=False, repr=False)
    repetition_key: str = field(init=False)
    repetition_count: int = field(init=False)
    side_to_move: str = field(init=False)
    legal_actions: tuple[str, ...] = field(init=False)
    is_check: bool = field(init=False)
    outcome: ChessOutcome = field(init=False)

    def __post_init__(self) -> None:
        """Populate cached board and summary fields from `fen`."""
        self._populate_from_board(chess.Board(self.fen))

    @classmethod
    def from_board(
        cls,
        *,
        board: chess.Board,
        repetition_counts: dict[str, int],
        metadata: dict[str, Any],
    ) -> Self:
        """Create a canonical state from an existing board instance.

        Parameters
        ----------
        board : chess.Board
            Board whose current position should become the canonical state.
        repetition_counts : dict[str, int]
            Repetition counters accumulated for the episode.
        metadata : dict[str, Any]
            Free-form state metadata to carry forward unchanged.

        Returns
        -------
        Self
            Fully populated chess state whose cached board summary matches the
            supplied board.
        """
        state = object.__new__(cls)
        state.repetition_counts = repetition_counts
        state.metadata = metadata
        state._populate_from_board(board)
        return state

    @property
    def legal_action_count(self) -> int:
        """Return the number of legal moves in the current position.

        Returns
        -------
        int
            Count of legal serialized UCI actions.
        """
        return len(self.legal_actions)

    @property
    def is_terminal(self) -> bool:
        """Return whether the current position ends the episode.

        Returns
        -------
        bool
            `True` when the cached outcome marks the state as terminal.
        """
        return self.outcome.is_terminal

    def _populate_from_board(self, board: chess.Board) -> None:
        """Populate cached state fields from a `python-chess` board.

        Parameters
        ----------
        board : chess.Board
            Board to store and summarize.
        """
        self.board = board
        self.fen = board.fen()
        self.repetition_key = repetition_key_from_board(board)
        self.repetition_count = self.repetition_counts.get(self.repetition_key, 1)
        self.side_to_move = "white" if board.turn == chess.WHITE else "black"
        self.legal_actions = tuple(sorted(move.uci() for move in board.legal_moves))
        self.is_check = board.is_check()
        self.outcome = outcome_from_board(board, self.repetition_count)
