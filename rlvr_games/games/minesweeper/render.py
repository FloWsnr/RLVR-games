"""Observation rendering for Minesweeper state."""

from functools import cached_property
from hashlib import sha256

from PIL import Image, ImageDraw, ImageFont

from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.types import Observation, RenderedImage
from rlvr_games.games.minesweeper.state import (
    MinesweeperState,
    public_minesweeper_board,
    public_minesweeper_metadata,
)


class MinesweeperAsciiBoardFormatter:
    """Render a Minesweeper board as labeled ASCII text."""

    def render_text(self, state: MinesweeperState) -> str:
        """Return a compact ASCII board with row and column labels.

        Parameters
        ----------
        state : MinesweeperState
            Canonical Minesweeper state to render.

        Returns
        -------
        str
            Multi-line public board diagram with one-based coordinates.
        """
        visible_board = public_minesweeper_board(state=state)
        row_label_width = len(str(state.rows))
        cell_width = 2
        header_cells = " ".join(
            str(column_index + 1).rjust(cell_width)
            for column_index in range(state.columns)
        )
        lines = [f"{' ' * (row_label_width + 1)} {header_cells}"]
        for row_index, row in enumerate(visible_board):
            rendered_cells = " ".join(cell.rjust(cell_width) for cell in row)
            lines.append(
                f"{str(row_index + 1).rjust(row_label_width)} | {rendered_cells}"
            )
        return "\n".join(lines)


class MinesweeperImageRenderer:
    """Render Minesweeper boards as in-memory raster images."""

    _BACKGROUND = (241, 237, 228, 255)
    _GRID = (94, 91, 84, 255)
    _HIDDEN = (155, 163, 170, 255)
    _REVEALED = (235, 232, 220, 255)
    _FLAGGED = (196, 74, 59, 255)
    _MINE = (64, 35, 35, 255)
    _TEXT = (43, 42, 39, 255)
    _NUMBER_COLORS = {
        "1": (44, 92, 173, 255),
        "2": (35, 124, 69, 255),
        "3": (171, 46, 46, 255),
        "4": (81, 55, 140, 255),
        "5": (130, 66, 34, 255),
        "6": (42, 118, 121, 255),
        "7": (70, 70, 70, 255),
        "8": (111, 111, 111, 255),
    }

    def __init__(self, *, size: int) -> None:
        """Initialize the image renderer.

        Parameters
        ----------
        size : int
            Width and height of the rendered raster image in pixels.
        """
        self.size = size
        self._padding = max(12, size // 30)

    def render_images(self, state: MinesweeperState) -> tuple[RenderedImage, ...]:
        """Render the public board view into one raster image.

        Parameters
        ----------
        state : MinesweeperState
            Canonical state to render.

        Returns
        -------
        tuple[RenderedImage, ...]
            Single-item tuple containing the rendered image payload.
        """
        visible_board = public_minesweeper_board(state=state)
        cell_size = self._cell_size(rows=state.rows, columns=state.columns)
        board_width = state.columns * cell_size
        board_height = state.rows * cell_size
        left = (self.size - board_width) // 2
        top = (self.size - board_height) // 2

        image = Image.new("RGBA", (self.size, self.size), self._BACKGROUND)
        draw = ImageDraw.Draw(image, "RGBA")
        for row_index, row in enumerate(visible_board):
            for col_index, label in enumerate(row):
                x0 = left + (col_index * cell_size)
                y0 = top + (row_index * cell_size)
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                draw.rectangle((x0, y0, x1, y1), fill=self._cell_fill(label=label))
                draw.rectangle((x0, y0, x1, y1), outline=self._GRID, width=1)
                self._draw_label(
                    draw=draw,
                    label=label,
                    x0=x0,
                    y0=y0,
                    cell_size=cell_size,
                )

        digest = sha256(
            f"{visible_board}|{state.rows}|{state.columns}|{self.size}".encode("utf-8")
        ).hexdigest()[:16]
        return (RenderedImage(key=f"minesweeper-board-{digest}", image=image),)

    @cached_property
    def _font_name(self) -> str:
        """Return the preferred truetype font name."""
        return "DejaVuSans-Bold.ttf"

    @cached_property
    def _font_cache(self) -> dict[tuple[int, str], ImageFont.FreeTypeFont]:
        """Return the in-memory label font cache."""
        return {}

    def _cell_size(self, *, rows: int, columns: int) -> int:
        """Return the pixel size of one cell."""
        available_width = self.size - (2 * self._padding)
        available_height = self.size - (2 * self._padding)
        return max(16, min(available_width // columns, available_height // rows))

    def _cell_fill(self, *, label: str) -> tuple[int, int, int, int]:
        """Return the fill color for one visible label."""
        if label == "#":
            return self._HIDDEN
        if label == "F":
            return self._FLAGGED
        if label in {"M", "*"}:
            return self._MINE
        return self._REVEALED

    def _draw_label(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        label: str,
        x0: int,
        y0: int,
        cell_size: int,
    ) -> None:
        """Draw one visible cell label centered inside the cell."""
        if label in {"#", "."}:
            return

        font = self._label_font(draw=draw, label=label, cell_size=cell_size)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = int(x0 + ((cell_size - text_width) / 2) - text_bbox[0])
        y = int(y0 + ((cell_size - text_height) / 2) - text_bbox[1])
        draw.text(
            (x, y),
            label,
            font=font,
            fill=self._NUMBER_COLORS.get(label, self._TEXT),
        )

    def _label_font(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        label: str,
        cell_size: int,
    ) -> ImageFont.FreeTypeFont:
        """Return a cached font sized for one cell label."""
        cache_key = (cell_size, label)
        cached_font = self._font_cache.get(cache_key)
        if cached_font is not None:
            return cached_font

        font_size = max(12, int(cell_size * 0.56))
        minimum_font_size = 10
        maximum_span = cell_size - (2 * max(3, cell_size // 10))
        while True:
            font = ImageFont.truetype(self._font_name, font_size)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if (
                text_width <= maximum_span and text_height <= maximum_span
            ) or font_size == minimum_font_size:
                self._font_cache[cache_key] = font
                return font
            font_size = max(minimum_font_size, font_size - 1)


class MinesweeperObservationRenderer:
    """Render a Minesweeper observation from canonical state."""

    def __init__(
        self,
        *,
        board_formatter: TextRenderer[MinesweeperState],
        image_renderer: ImageRenderer[MinesweeperState] | None,
    ) -> None:
        """Initialize the observation renderer with explicit view components.

        Parameters
        ----------
        board_formatter : TextRenderer[MinesweeperState]
            Text renderer used to build the board view.
        image_renderer : ImageRenderer[MinesweeperState] | None
            Optional image renderer used to produce raster observations.
        """
        self.board_formatter = board_formatter
        self.image_renderer = image_renderer

    def render(self, state: MinesweeperState) -> Observation:
        """Render a Minesweeper state into a model-facing observation.

        Parameters
        ----------
        state : MinesweeperState
            Canonical Minesweeper state to render.

        Returns
        -------
        Observation
            Observation derived from the public state view only.
        """
        lines = [
            "Minesweeper board:",
            self.board_formatter.render_text(state),
            f"Size: {state.rows}x{state.columns}",
            f"Mines: {state.mine_count}",
            f"Moves: {state.move_count}",
            f"Revealed safe cells: {state.revealed_safe_count}",
            f"Flags: {state.flagged_cell_count}",
            f"Hidden cells: {state.hidden_cell_count}",
            f"Remaining safe cells: {state.remaining_safe_cells}",
            f"Pending mine layout: {'yes' if state.has_pending_mines else 'no'}",
            f"Legal actions: {state.legal_action_count}",
            "Action format: reveal <row> <col> | flag <row> <col> | unflag <row> <col>",
            f"Terminal: {'yes' if state.is_terminal else 'no'}",
            f"Won: {'yes' if state.outcome.won else 'no'}",
        ]
        if state.is_terminal:
            lines.append(f"Termination: {state.outcome.termination}")

        images: tuple[RenderedImage, ...] = ()
        if self.image_renderer is not None:
            images = self.image_renderer.render_images(state)

        return Observation(
            text="\n".join(lines),
            images=images,
            metadata=public_minesweeper_metadata(state=state),
        )
