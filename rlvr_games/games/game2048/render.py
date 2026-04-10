"""Observation rendering for 2048 state."""

from functools import cached_property
from hashlib import sha256

from PIL import Image, ImageDraw, ImageFont

from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.types import Observation, RenderedImage
from rlvr_games.games.game2048.engine import Board
from rlvr_games.games.game2048.state import Game2048State


class Game2048AsciiBoardFormatter:
    """Render a 2048 board as ASCII text."""

    def render_text(self, board: Board) -> str:
        """Return an ASCII board diagram with fixed-width cells.

        Parameters
        ----------
        board : Board
            Board to render.

        Returns
        -------
        str
            Multi-line string containing the grid.
        """
        largest_value = max(max(row) for row in board)
        cell_width = max(6, len(str(largest_value)) + 2)
        horizontal_rule = "+" + "+".join("-" * cell_width for _ in board) + "+"
        lines = [horizontal_rule]
        for row in board:
            rendered_cells = []
            for value in row:
                label = "." if value == 0 else str(value)
                rendered_cells.append(label.rjust(cell_width))
            lines.append("|" + "|".join(rendered_cells) + "|")
            lines.append(horizontal_rule)
        return "\n".join(lines)


class Game2048ImageRenderer:
    """Render 2048 boards as in-memory raster images."""

    _BACKGROUND = (250, 248, 239, 255)
    _BOARD_BACKGROUND = (187, 173, 160, 255)
    _EMPTY_TILE = (205, 193, 180, 255)
    _DARK_TEXT = (119, 110, 101, 255)
    _LIGHT_TEXT = (249, 246, 242, 255)
    _TILE_COLORS = {
        2: (238, 228, 218, 255),
        4: (237, 224, 200, 255),
        8: (242, 177, 121, 255),
        16: (245, 149, 99, 255),
        32: (246, 124, 95, 255),
        64: (246, 94, 59, 255),
        128: (237, 207, 114, 255),
        256: (237, 204, 97, 255),
        512: (237, 200, 80, 255),
        1024: (237, 197, 63, 255),
        2048: (237, 194, 46, 255),
    }
    _LARGE_TILE = (60, 58, 50, 255)

    def __init__(self, *, size: int) -> None:
        """Initialize the image renderer.

        Parameters
        ----------
        size : int
            Width and height of the rendered raster image in pixels.
        """
        self.size = size
        self._outer_padding = max(12, size // 24)
        self._gap = max(6, size // 60)

    def render_images(self, board: Board) -> tuple[RenderedImage, ...]:
        """Render the board to an in-memory raster image.

        Parameters
        ----------
        board : Board
            Board to render.

        Returns
        -------
        tuple[RenderedImage, ...]
            Single-item tuple containing the rendered image payload.
        """
        size = len(board)
        tile_size = self._tile_size(board_size=size)
        board_span = (tile_size * size) + (self._gap * (size - 1))
        board_left = (self.size - board_span) // 2
        board_top = (self.size - board_span) // 2

        image = Image.new(
            mode="RGBA",
            size=(self.size, self.size),
            color=self._BACKGROUND,
        )
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle(
            (
                board_left - self._outer_padding,
                board_top - self._outer_padding,
                board_left + board_span + self._outer_padding,
                board_top + board_span + self._outer_padding,
            ),
            radius=min(8, self._outer_padding),
            fill=self._BOARD_BACKGROUND,
        )

        for row_index, row in enumerate(board):
            for column_index, value in enumerate(row):
                x0 = board_left + (column_index * (tile_size + self._gap))
                y0 = board_top + (row_index * (tile_size + self._gap))
                x1 = x0 + tile_size
                y1 = y0 + tile_size
                draw.rounded_rectangle(
                    (x0, y0, x1, y1),
                    radius=min(8, tile_size // 8),
                    fill=self._tile_color(value=value),
                )
                if value != 0:
                    self._draw_tile_value(
                        draw=draw,
                        value=value,
                        x0=x0,
                        y0=y0,
                        tile_size=tile_size,
                    )

        digest = sha256(f"{board}|{self.size}".encode("utf-8")).hexdigest()[:16]
        return (RenderedImage(key=f"2048-board-{digest}", image=image),)

    @cached_property
    def _dark_text_font(self) -> str:
        """Return the preferred font path hint for dark text tiles.

        Returns
        -------
        str
            Font name passed into Pillow's truetype loader.
        """
        return "DejaVuSans-Bold.ttf"

    def _tile_size(self, *, board_size: int) -> int:
        """Return the pixel size of one tile for a board dimension.

        Parameters
        ----------
        board_size : int
            Number of rows and columns in the board.

        Returns
        -------
        int
            Integer tile size in pixels.
        """
        available = (
            self.size - (2 * self._outer_padding) - (self._gap * (board_size - 1))
        )
        return max(32, available // board_size)

    def _tile_color(self, *, value: int) -> tuple[int, int, int, int]:
        """Return the fill color for a tile value.

        Parameters
        ----------
        value : int
            Tile value to color.

        Returns
        -------
        tuple[int, int, int, int]
            RGBA fill color.
        """
        if value == 0:
            return self._EMPTY_TILE
        return self._TILE_COLORS.get(value, self._LARGE_TILE)

    def _tile_text_color(self, *, value: int) -> tuple[int, int, int, int]:
        """Return the text color for a tile value.

        Parameters
        ----------
        value : int
            Tile value whose text should be colored.

        Returns
        -------
        tuple[int, int, int, int]
            RGBA text color.
        """
        if value in {2, 4}:
            return self._DARK_TEXT
        return self._LIGHT_TEXT

    def _draw_tile_value(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        value: int,
        x0: int,
        y0: int,
        tile_size: int,
    ) -> None:
        """Draw one tile value centered within a tile rectangle.

        Parameters
        ----------
        draw : ImageDraw.ImageDraw
            Drawing context targeting the output image.
        value : int
            Tile value to draw.
        x0 : int
            Left pixel of the tile rectangle.
        y0 : int
            Top pixel of the tile rectangle.
        tile_size : int
            Width and height of the tile rectangle.
        """
        font = self._value_font(value=value, tile_size=tile_size)
        text = str(value)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = int(x0 + ((tile_size - text_width) / 2) - text_bbox[0])
        y = int(y0 + ((tile_size - text_height) / 2) - text_bbox[1])
        draw.text(
            (x, y),
            text,
            font=font,
            fill=self._tile_text_color(value=value),
        )

    def _value_font(self, *, value: int, tile_size: int) -> ImageFont.FreeTypeFont:
        """Return a value-sized font for one tile.

        Parameters
        ----------
        value : int
            Tile value whose number of digits should drive the font size.
        tile_size : int
            Pixel width and height of the tile rectangle.

        Returns
        -------
        ImageFont.FreeTypeFont
            Loaded font sized for the tile.
        """
        digit_count = len(str(value))
        scale = 0.5
        if digit_count >= 4:
            scale = 0.38
        if digit_count >= 5:
            scale = 0.32
        return ImageFont.truetype(
            self._dark_text_font,
            max(14, int(tile_size * scale)),
        )


class Game2048ObservationRenderer:
    """Render a 2048 observation from canonical state."""

    def __init__(
        self,
        *,
        board_formatter: TextRenderer[Board],
        image_renderer: ImageRenderer[Board] | None,
    ) -> None:
        """Initialize the observation renderer with explicit view components.

        Parameters
        ----------
        board_formatter : TextRenderer[Board]
            Text renderer used to build the board view.
        image_renderer : ImageRenderer[Board] | None
            Optional image renderer used to produce raster observations.
        """
        self.board_formatter = board_formatter
        self.image_renderer = image_renderer

    def render(self, state: Game2048State) -> Observation:
        """Render a 2048 state into a model-facing observation.

        Parameters
        ----------
        state : Game2048State
            Canonical 2048 state to render.

        Returns
        -------
        Observation
            Observation whose text and images are derived from the canonical
            board and cached state summary.
        """
        legal_actions_text = " ".join(state.legal_actions) or "none"
        lines = [
            "2048 board:",
            self.board_formatter.render_text(state.board),
            f"Score: {state.score}",
            f"Moves: {state.move_count}",
            f"Target tile: {state.target_value}",
            f"Max tile: {state.max_tile}",
            f"Empty cells: {state.empty_cell_count}",
            f"Legal actions ({state.legal_action_count}): {legal_actions_text}",
            f"Terminal: {'yes' if state.is_terminal else 'no'}",
            f"Won: {'yes' if state.outcome.won else 'no'}",
        ]
        if state.is_terminal:
            lines.append(f"Termination: {state.outcome.termination}")

        metadata: dict[str, object] = {
            "board": state.board,
            "score": state.score,
            "move_count": state.move_count,
            "target_value": state.target_value,
            "size": state.size,
            "legal_actions": state.legal_actions,
            "legal_action_count": state.legal_action_count,
            "empty_cell_count": state.empty_cell_count,
            "max_tile": state.max_tile,
            "is_terminal": state.is_terminal,
            "won": state.outcome.won if state.is_terminal else False,
        }
        metadata.update(state.outcome.metadata())

        images: tuple[RenderedImage, ...] = ()
        if self.image_renderer is not None:
            images = self.image_renderer.render_images(state.board)

        return Observation(
            text="\n".join(lines),
            images=images,
            metadata=metadata,
        )
