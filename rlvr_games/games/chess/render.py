"""Observation rendering for chess state."""

from typing import Protocol

import chess

from rlvr_games.core.types import Observation
from rlvr_games.games.chess.state import (
    ChessState,
    repetition_key_from_board,
    winner_name,
)


class ChessBoardTextFormatter(Protocol):
    """Protocol for formatting a chess board into observation text."""

    def format_board(self, board: chess.Board) -> str:
        """Return a text representation of the given board."""
        ...


class ChessBoardImageRenderer(Protocol):
    """Protocol for rendering chess board images for an observation."""

    def render_images(self, board: chess.Board) -> tuple[str, ...]:
        """Return zero or more image paths for the given board."""
        ...


class AsciiBoardFormatter:
    """Render a board as ASCII text with coordinate labels."""

    def format_board(self, board: chess.Board) -> str:
        """Return an ASCII board diagram with ranks and files.

        Parameters
        ----------
        board : chess.Board
            Board to render.

        Returns
        -------
        str
            Multi-line string showing piece placement from White's perspective.
        """
        board_rows = str(board).splitlines()
        labeled_rows = [
            f"{rank} {row}"
            for rank, row in zip(range(8, 0, -1), board_rows, strict=True)
        ]
        labeled_rows.append("  a b c d e f g h")
        return "\n".join(labeled_rows)


class UnicodeBoardFormatter:
    """Render a board with Unicode glyphs and coordinate labels."""

    def format_board(self, board: chess.Board) -> str:
        """Return a Unicode board diagram with borders and files/ranks."""
        return board.unicode(borders=True, empty_square=".")


class EmptyChessBoardImageRenderer:
    """Return no images for chess observations."""

    def render_images(self, board: chess.Board) -> tuple[str, ...]:
        """Return an empty image set for the observation."""
        return ()


class ChessObservationRenderer:
    """Render a chess observation from canonical state.

    The renderer combines text and image views over the canonical board state
    while exposing metadata derived from the verifier-backed state, such as
    side to move, legal move count, repetition state, and terminal outcomes.
    """

    def __init__(
        self,
        *,
        board_formatter: ChessBoardTextFormatter,
        image_renderer: ChessBoardImageRenderer,
    ) -> None:
        """Initialize the observation renderer with explicit view components.

        Parameters
        ----------
        board_formatter : ChessBoardTextFormatter
            Formatter used to build the text board view.
        image_renderer : ChessBoardImageRenderer
            Renderer used to produce image paths for the observation.
        """
        self.board_formatter = board_formatter
        self.image_renderer = image_renderer

    def _render_text(
        self,
        *,
        board: chess.Board,
        side_to_move: str,
        legal_action_count: int,
        repetition_count: int,
        terminal: bool,
        metadata: dict[str, object],
    ) -> str:
        """Assemble the text portion of the observation."""
        lines = [
            "Chess board:",
            self.board_formatter.format_board(board),
            f"FEN: {board.fen()}",
            f"Side to move: {side_to_move}",
            f"Legal move count: {legal_action_count}",
            f"Repetition count: {repetition_count}",
            f"In check: {'yes' if board.is_check() else 'no'}",
            f"Terminal: {'yes' if terminal else 'no'}",
        ]
        if "result" in metadata:
            lines.append(f"Result: {metadata['result']}")
            lines.append(f"Winner: {metadata['winner'] or 'draw'}")
            lines.append(f"Termination: {metadata['termination']}")
        return "\n".join(lines)

    def render(self, state: ChessState) -> Observation:
        """Render a chess state into a multimodal model observation.

        Parameters
        ----------
        state : ChessState
            Canonical chess state to render.

        Returns
        -------
        Observation
            Observation whose text and image fields are derived from the board,
            and whose metadata mirrors the rendered state summary.
        """
        board = chess.Board(state.fen)
        repetition_count = state.repetition_counts.get(
            repetition_key_from_board(board), 1
        )
        outcome = board.outcome(claim_draw=True)
        side_to_move = "white" if board.turn == chess.WHITE else "black"
        terminal = repetition_count >= 3 or outcome is not None
        legal_action_count = board.legal_moves.count()
        metadata: dict[str, object] = {
            "fen": board.fen(),
            "turn": side_to_move,
            "side_to_move": side_to_move,
            "is_check": board.is_check(),
            "is_terminal": terminal,
            "legal_action_count": legal_action_count,
            "repetition_count": repetition_count,
        }

        if repetition_count >= 3:
            metadata["result"] = "1/2-1/2"
            metadata["termination"] = "threefold_repetition"
            metadata["winner"] = None
        elif outcome is not None:
            metadata["result"] = board.result(claim_draw=True)
            metadata["termination"] = outcome.termination.name.lower()
            metadata["winner"] = winner_name(outcome.winner)

        return Observation(
            text=self._render_text(
                board=board,
                side_to_move=side_to_move,
                legal_action_count=legal_action_count,
                repetition_count=repetition_count,
                terminal=terminal,
                metadata=metadata,
            ),
            image_paths=self.image_renderer.render_images(board),
            metadata=metadata,
        )


class ChessTextRenderer(ChessObservationRenderer):
    """Compatibility wrapper for ASCII-only chess observations."""

    def __init__(self) -> None:
        """Initialize the legacy text-only renderer."""
        super().__init__(
            board_formatter=AsciiBoardFormatter(),
            image_renderer=EmptyChessBoardImageRenderer(),
        )
