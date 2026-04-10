"""Observation rendering for chess state."""

from hashlib import sha256
from pathlib import Path

import cairosvg
import chess
import chess.svg

from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.types import Observation
from rlvr_games.games.chess.state import ChessState


class AsciiBoardFormatter:
    """Render a board as ASCII text with coordinate labels."""

    def __init__(self, *, orientation: chess.Color) -> None:
        """Initialize an ASCII board formatter.

        Parameters
        ----------
        orientation : chess.Color
            Color whose perspective should be shown at the bottom of the board.
        """
        self.orientation = orientation

    def render_text(self, board: chess.Board) -> str:
        """Return an ASCII board diagram with ranks and files.

        Parameters
        ----------
        board : chess.Board
            Board to render.

        Returns
        -------
        str
            Multi-line string showing piece placement from the configured
            perspective.
        """
        if self.orientation == chess.WHITE:
            rank_indexes = range(7, -1, -1)
            file_indexes = range(8)
        else:
            rank_indexes = range(8)
            file_indexes = range(7, -1, -1)

        labeled_rows = [
            f"{rank_index + 1} {self._render_rank(board, rank_index, file_indexes)}"
            for rank_index in rank_indexes
        ]
        file_labels = " ".join(
            chess.FILE_NAMES[file_index] for file_index in file_indexes
        )
        labeled_rows.append(f"  {file_labels}")
        return "\n".join(labeled_rows)

    def _render_rank(
        self,
        board: chess.Board,
        rank_index: int,
        file_indexes: range,
    ) -> str:
        """Render one rank in the configured file order.

        Parameters
        ----------
        board : chess.Board
            Board containing the pieces to render.
        rank_index : int
            Zero-based rank index to render.
        file_indexes : range
            Zero-based file indexes in display order.

        Returns
        -------
        str
            Space-separated piece symbols for the requested rank.
        """
        symbols: list[str] = []
        for file_index in file_indexes:
            piece = board.piece_at(chess.square(file_index, rank_index))
            symbols.append("." if piece is None else piece.symbol())
        return " ".join(symbols)


class UnicodeBoardFormatter:
    """Render a board with Unicode glyphs and coordinate labels."""

    def __init__(self, *, orientation: chess.Color) -> None:
        """Initialize a Unicode board formatter.

        Parameters
        ----------
        orientation : chess.Color
            Color whose perspective should be shown at the bottom of the board.
        """
        self.orientation = orientation

    def render_text(self, board: chess.Board) -> str:
        """Return a Unicode board diagram with borders and files/ranks.

        Parameters
        ----------
        board : chess.Board
            Board to render.

        Returns
        -------
        str
            Multi-line string showing piece placement with Unicode glyphs from
            the configured perspective.
        """
        return board.unicode(
            borders=True,
            empty_square=".",
            orientation=self.orientation,
        )


class ChessRasterBoardImageRenderer:
    """Render chess boards to PNG image files."""

    def __init__(
        self,
        *,
        output_dir: Path,
        size: int,
        coordinates: bool,
        orientation: chess.Color,
    ) -> None:
        """Initialize a raster board image renderer.

        Parameters
        ----------
        output_dir : Path
            Directory where rendered PNG files should be written.
        size : int
            Width and height of the rendered PNG board in pixels.
        coordinates : bool
            Whether rank and file coordinates should be shown.
        orientation : chess.Color
            Color whose perspective should be shown at the bottom of the board.
        """
        self.output_dir = output_dir
        self.size = size
        self.coordinates = coordinates
        self.orientation = orientation

    def render_images(self, board: chess.Board) -> tuple[Path, ...]:
        """Render the board to a PNG file and return its path.

        Parameters
        ----------
        board : chess.Board
            Board to render.

        Returns
        -------
        tuple[Path, ...]
            Single-item tuple containing the rendered PNG filesystem path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        digest = sha256(
            (f"{board.fen()}|{self.size}|{self.coordinates}|{self.orientation}").encode(
                "utf-8",
            )
        ).hexdigest()[:16]
        image_path = self.output_dir / f"chess-board-{digest}.png"
        if image_path.exists():
            return (image_path,)

        check_square = board.king(board.turn) if board.is_check() else None
        svg_text = chess.svg.board(
            board=board,
            orientation=self.orientation,
            check=check_square,
            size=self.size,
            coordinates=self.coordinates,
        )
        with image_path.open("wb") as image_file:
            cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                write_to=image_file,
            )
        return (image_path,)


class ChessObservationRenderer:
    """Render a chess observation from canonical state.

    The renderer combines text and image views over the canonical board state
    while exposing metadata derived from the verifier-backed state, such as
    side to move, legal move count, repetition state, and terminal outcomes.
    """

    def __init__(
        self,
        *,
        board_formatter: TextRenderer[chess.Board],
        image_renderer: ImageRenderer[chess.Board] | None,
    ) -> None:
        """Initialize the observation renderer with explicit view components.

        Parameters
        ----------
        board_formatter : TextRenderer[chess.Board]
            Renderer used to build the text board view.
        image_renderer : ImageRenderer[chess.Board] | None
            Renderer used to produce image paths for the observation. When
            `None`, observations include no image paths.
        """
        self.board_formatter = board_formatter
        self.image_renderer = image_renderer

    def _render_text(
        self,
        *,
        state: ChessState,
    ) -> str:
        """Assemble the text portion of the observation."""
        lines = [
            "Chess board:",
            self.board_formatter.render_text(state.board),
            f"FEN: {state.fen}",
            f"Side to move: {state.side_to_move}",
            f"Legal move count: {state.legal_action_count}",
            f"Repetition count: {state.repetition_count}",
            f"In check: {'yes' if state.is_check else 'no'}",
            f"Terminal: {'yes' if state.is_terminal else 'no'}",
        ]
        if state.outcome.is_terminal:
            lines.append(f"Result: {state.outcome.result}")
            lines.append(f"Winner: {state.outcome.winner or 'draw'}")
            lines.append(f"Termination: {state.outcome.termination}")
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
        metadata: dict[str, object] = {
            "fen": state.fen,
            "turn": state.side_to_move,
            "side_to_move": state.side_to_move,
            "is_check": state.is_check,
            "is_terminal": state.is_terminal,
            "legal_action_count": state.legal_action_count,
            "repetition_count": state.repetition_count,
        }
        metadata.update(state.outcome.metadata())

        image_paths = (
            ()
            if self.image_renderer is None
            else self.image_renderer.render_images(state.board)
        )

        return Observation(
            text=self._render_text(
                state=state,
            ),
            image_paths=image_paths,
            metadata=metadata,
        )
