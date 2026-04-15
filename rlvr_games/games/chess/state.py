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


@dataclass(init=False, slots=True, frozen=True)
class ChessState:
    """Canonical chess state.

    The state stores a replayable FEN string, a private `python-chess` board,
    and derived summary fields. Public accessors return copies of mutable
    payloads so callers cannot accidentally desynchronize the cached board,
    legal actions, repetition data, and terminal outcome.

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
    _board: chess.Board = field(init=False, repr=False)
    _repetition_counts: dict[str, int] = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)
    repetition_key: str = field(init=False)
    repetition_count: int = field(init=False)
    side_to_move: str = field(init=False)
    legal_actions: tuple[str, ...] = field(init=False)
    is_check: bool = field(init=False)
    outcome: ChessOutcome = field(init=False)

    def __init__(self, *, fen: str) -> None:
        """Create a chess state from a FEN position.

        Parameters
        ----------
        fen : str
            Full FEN string describing the current board position.

        Raises
        ------
        ValueError
            If `fen` is not accepted by `python-chess`.
        """
        board = chess.Board(fen)
        self._populate_from_board(
            board=board,
            repetition_counts={repetition_key_from_board(board): 1},
            metadata={},
        )

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
        state._populate_from_board(
            board=board,
            repetition_counts=repetition_counts,
            metadata=metadata,
        )
        return state

    @property
    def board(self) -> chess.Board:
        """Return a copy of the current board.

        Returns
        -------
        chess.Board
            Stackless board copy representing the canonical FEN. Mutating this
            board does not mutate the state.
        """
        return self.board_copy()

    @property
    def repetition_counts(self) -> dict[str, int]:
        """Return a copy of the episode repetition counters.

        Returns
        -------
        dict[str, int]
            Copy of the repetition-significant position counts accumulated for
            this episode.
        """
        return dict(self._repetition_counts)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of free-form state metadata.

        Returns
        -------
        dict[str, Any]
            Copy of game-specific metadata carried with the state.
        """
        return dict(self._metadata)

    def board_copy(self) -> chess.Board:
        """Return a stackless copy of the canonical board.

        Returns
        -------
        chess.Board
            Board copy suitable for parsing, rendering, or applying a move
            without mutating this state.
        """
        return self._board.copy(stack=False)

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

    def _populate_from_board(
        self,
        *,
        board: chess.Board,
        repetition_counts: dict[str, int],
        metadata: dict[str, Any],
    ) -> None:
        """Populate cached state fields from a `python-chess` board.

        Parameters
        ----------
        board : chess.Board
            Board to store and summarize.
        repetition_counts : dict[str, int]
            Repetition counters accumulated for the episode.
        metadata : dict[str, Any]
            Free-form state metadata to carry forward.
        """
        board_copy = board.copy(stack=False)
        repetition_key = repetition_key_from_board(board_copy)
        next_repetition_counts = dict(repetition_counts)
        next_repetition_counts.setdefault(repetition_key, 1)
        repetition_count = next_repetition_counts[repetition_key]

        object.__setattr__(self, "_board", board_copy)
        object.__setattr__(self, "fen", board_copy.fen())
        object.__setattr__(self, "repetition_key", repetition_key)
        object.__setattr__(self, "_repetition_counts", next_repetition_counts)
        object.__setattr__(self, "repetition_count", repetition_count)
        object.__setattr__(self, "_metadata", dict(metadata))
        object.__setattr__(
            self,
            "side_to_move",
            "white" if board_copy.turn == chess.WHITE else "black",
        )
        object.__setattr__(
            self,
            "legal_actions",
            tuple(sorted(move.uci() for move in board_copy.legal_moves)),
        )
        object.__setattr__(self, "is_check", board_copy.is_check())
        object.__setattr__(
            self,
            "outcome",
            outcome_from_board(board_copy, repetition_count),
        )


def public_chess_metadata(state: ChessState) -> dict[str, object]:
    """Return model-safe observation metadata for a chess state.

    Parameters
    ----------
    state : ChessState
        Canonical chess state to summarize.

    Returns
    -------
    dict[str, object]
        Observation metadata derived from public board state only.
    """
    metadata: dict[str, object] = {
        "fen": state.fen,
        "turn": state.side_to_move,
        "side_to_move": state.side_to_move,
        "is_check": state.is_check,
        "is_terminal": state.is_terminal,
        "repetition_count": state.repetition_count,
    }
    metadata.update(state.outcome.metadata())
    return metadata


def inspect_chess_state(state: ChessState) -> dict[str, object]:
    """Return a structured canonical summary of a chess state.

    Parameters
    ----------
    state : ChessState
        Canonical chess state to inspect.

    Returns
    -------
    dict[str, object]
        Debug-oriented state summary derived from cached state fields.
    """
    summary = public_chess_metadata(state=state)
    summary.update(
        {
            "repetition_key": state.repetition_key,
            "repetition_counts": state.repetition_counts,
            "metadata": state.metadata,
            "legal_action_count": state.legal_action_count,
        }
    )
    return summary
