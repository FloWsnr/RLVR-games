"""Observation rendering for chess state."""

from functools import cached_property
from hashlib import sha256

import chess
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage

from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.types import Observation, RenderedImage
from rlvr_games.games.chess.state import ChessState, public_chess_metadata


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


class ChessFastImageRenderer:
    """Render chess boards as in-memory raster images.

    The renderer uses Pillow composition over a cached board background and
    cached piece glyph sprites. This avoids filesystem writes and SVG
    rasterization during rollout-heavy training loops while still producing
    images that the CLI can persist when needed.
    """

    _LIGHT_SQUARE = (240, 217, 181, 255)
    _DARK_SQUARE = (181, 136, 99, 255)
    _LABEL_TEXT = (54, 54, 54, 255)
    _CANVAS_BACKGROUND = (255, 255, 255, 255)
    _WHITE_PIECE_FILL = (250, 249, 245, 255)
    _WHITE_PIECE_STROKE = (48, 48, 48, 255)
    _BLACK_PIECE_FILL = (44, 44, 44, 255)
    _BLACK_PIECE_STROKE = (240, 240, 240, 255)
    _CHECK_OVERLAY = (214, 74, 61, 92)
    _PIECE_GLYPHS = {
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
    }

    def __init__(
        self,
        *,
        size: int,
        coordinates: bool,
        orientation: chess.Color,
    ) -> None:
        """Initialize a fast in-memory chess image renderer.

        Parameters
        ----------
        size : int
            Width and height of the rendered raster image in pixels.
        coordinates : bool
            Whether rank and file coordinates should be shown.
        orientation : chess.Color
            Color whose perspective should be shown at the bottom of the board.
        """
        self.size = size
        self.coordinates = coordinates
        self.orientation = orientation
        self._label_margin = 0 if not coordinates else max(18, size // 14)
        available_size = size - (2 * self._label_margin)
        self._square_size = max(8, available_size // 8)
        self._board_size = self._square_size * 8
        self._board_left = (size - self._board_size) // 2
        self._board_top = (size - self._board_size) // 2

    def render_images(self, board: chess.Board) -> tuple[RenderedImage, ...]:
        """Render the board to an in-memory raster image.

        Parameters
        ----------
        board : chess.Board
            Board to render.

        Returns
        -------
        tuple[RenderedImage, ...]
            Single-item tuple containing the rendered image payload.
        """
        image = self._background_image.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        self._draw_check_overlay(draw=draw, board=board)
        for square, piece in board.piece_map().items():
            sprite = self._piece_sprite(symbol=piece.symbol())
            if sprite is None:
                continue
            x, y = self._square_top_left(square=square)
            image.alpha_composite(sprite, dest=(x, y))

        digest = sha256(
            (f"{board.fen()}|{self.size}|{self.coordinates}|{self.orientation}").encode(
                "utf-8",
            )
        ).hexdigest()[:16]
        return (RenderedImage(key=f"chess-board-{digest}", image=image),)

    @cached_property
    def _background_image(self) -> PILImage:
        """Return the cached board background image for this configuration.

        Returns
        -------
        PILImage
            Background image containing the board squares and optional
            coordinate labels, without any pieces or check highlights.
        """
        image = Image.new(
            mode="RGBA",
            size=(self.size, self.size),
            color=self._CANVAS_BACKGROUND,
        )
        draw = ImageDraw.Draw(image, "RGBA")
        for display_rank in range(8):
            for display_file in range(8):
                color = (
                    self._LIGHT_SQUARE
                    if (display_rank + display_file) % 2 == 0
                    else self._DARK_SQUARE
                )
                x0 = self._board_left + (display_file * self._square_size)
                y0 = self._board_top + (display_rank * self._square_size)
                x1 = x0 + self._square_size
                y1 = y0 + self._square_size
                draw.rectangle((x0, y0, x1, y1), fill=color)

        if self.coordinates:
            self._draw_coordinates(draw=draw)
        return image

    @cached_property
    def _label_font(self) -> ImageFont.FreeTypeFont:
        """Return the font used for board coordinate labels.

        Returns
        -------
        ImageFont.FreeTypeFont
            Loaded font sized for coordinate labels.
        """
        return ImageFont.truetype("DejaVuSans.ttf", max(12, self._square_size // 4))

    @cached_property
    def _piece_font(self) -> ImageFont.FreeTypeFont:
        """Return the font used to draw chess piece glyph sprites.

        Returns
        -------
        ImageFont.FreeTypeFont
            Loaded font sized for piece sprites.
        """
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            max(16, int(self._square_size * 0.84)),
        )

    @cached_property
    def _piece_sprites(self) -> dict[str, PILImage]:
        """Return cached piece glyph sprites for the current square size.

        Returns
        -------
        dict[str, PILImage]
            Mapping from piece symbol to transparent RGBA sprite.
        """
        return {
            symbol: self._make_piece_sprite(symbol=symbol)
            for symbol in self._PIECE_GLYPHS
        }

    def _draw_coordinates(self, *, draw: ImageDraw.ImageDraw) -> None:
        """Draw file and rank labels around the board.

        Parameters
        ----------
        draw : ImageDraw.ImageDraw
            Drawing context targeting the background image.
        """
        if self.orientation == chess.WHITE:
            files = tuple(chess.FILE_NAMES)
            ranks = tuple(str(rank_number) for rank_number in range(8, 0, -1))
        else:
            files = tuple(reversed(chess.FILE_NAMES))
            ranks = tuple(str(rank_number) for rank_number in range(1, 9))

        for file_index, label in enumerate(files):
            x_center = (
                self._board_left
                + (file_index * self._square_size)
                + (self._square_size / 2)
            )
            label_bbox = self._label_font.getbbox(label)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            x = int(x_center - (label_width / 2))
            top_y = int(self._board_top - label_height - 4)
            bottom_y = int(self._board_top + self._board_size + 4)
            draw.text((x, top_y), label, font=self._label_font, fill=self._LABEL_TEXT)
            draw.text(
                (x, bottom_y),
                label,
                font=self._label_font,
                fill=self._LABEL_TEXT,
            )

        for rank_index, label in enumerate(ranks):
            y_center = (
                self._board_top
                + (rank_index * self._square_size)
                + (self._square_size / 2)
            )
            label_bbox = self._label_font.getbbox(label)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            x_left = int(self._board_left - label_width - 6)
            x_right = int(self._board_left + self._board_size + 6)
            y = int(y_center - (label_height / 2) - label_bbox[1])
            draw.text(
                (x_left, y),
                label,
                font=self._label_font,
                fill=self._LABEL_TEXT,
            )
            draw.text(
                (x_right, y),
                label,
                font=self._label_font,
                fill=self._LABEL_TEXT,
            )

    def _draw_check_overlay(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        board: chess.Board,
    ) -> None:
        """Highlight the checked king square when the side to move is in check.

        Parameters
        ----------
        draw : ImageDraw.ImageDraw
            Drawing context targeting the current output image.
        board : chess.Board
            Board being rendered.
        """
        if not board.is_check():
            return

        king_square = board.king(board.turn)
        if king_square is None:
            return

        x0, y0 = self._square_top_left(square=king_square)
        x1 = x0 + self._square_size
        y1 = y0 + self._square_size
        draw.rectangle((x0, y0, x1, y1), fill=self._CHECK_OVERLAY)

    def _piece_sprite(self, *, symbol: str) -> PILImage | None:
        """Return the cached sprite for a piece symbol.

        Parameters
        ----------
        symbol : str
            `python-chess` piece symbol to render.

        Returns
        -------
        PILImage | None
            Cached transparent sprite, or `None` if the symbol is unknown.
        """
        return self._piece_sprites.get(symbol)

    def _make_piece_sprite(self, *, symbol: str) -> PILImage:
        """Build one transparent glyph sprite for a chess piece.

        Parameters
        ----------
        symbol : str
            `python-chess` piece symbol to render.

        Returns
        -------
        PILImage
            Transparent RGBA sprite sized for one board square.
        """
        glyph = self._PIECE_GLYPHS[symbol]
        image = Image.new(
            mode="RGBA",
            size=(self._square_size, self._square_size),
            color=(0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(image, "RGBA")
        glyph_bbox = self._piece_font.getbbox(
            glyph, stroke_width=max(1, self._square_size // 24)
        )
        glyph_width = glyph_bbox[2] - glyph_bbox[0]
        glyph_height = glyph_bbox[3] - glyph_bbox[1]
        x = int((self._square_size - glyph_width) / 2 - glyph_bbox[0])
        y = int((self._square_size - glyph_height) / 2 - glyph_bbox[1])
        if symbol.isupper():
            fill = self._WHITE_PIECE_FILL
            stroke_fill = self._WHITE_PIECE_STROKE
        else:
            fill = self._BLACK_PIECE_FILL
            stroke_fill = self._BLACK_PIECE_STROKE
        draw.text(
            (x, y),
            glyph,
            font=self._piece_font,
            fill=fill,
            stroke_width=max(1, self._square_size // 24),
            stroke_fill=stroke_fill,
        )
        return image

    def _square_top_left(self, *, square: chess.Square) -> tuple[int, int]:
        """Return the top-left pixel for the given board square.

        Parameters
        ----------
        square : chess.Square
            Square whose pixel origin should be computed.

        Returns
        -------
        tuple[int, int]
            Top-left pixel coordinate for the square in the rendered image.
        """
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.orientation == chess.WHITE:
            display_file = file_index
            display_rank = 7 - rank_index
        else:
            display_file = 7 - file_index
            display_rank = rank_index
        return (
            self._board_left + (display_file * self._square_size),
            self._board_top + (display_rank * self._square_size),
        )


class ChessObservationRenderer:
    """Render a chess observation from canonical state.

    The renderer combines text and image views over the canonical board state
    while exposing metadata derived from the verifier-backed state, such as
    side to move, repetition state, and terminal outcomes.
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
            Renderer used to produce image payloads for the observation. When
            `None`, observations include no images.
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
        images = (
            ()
            if self.image_renderer is None
            else self.image_renderer.render_images(state.board)
        )

        return Observation(
            text=self._render_text(
                state=state,
            ),
            images=images,
            metadata=public_chess_metadata(state=state),
        )
