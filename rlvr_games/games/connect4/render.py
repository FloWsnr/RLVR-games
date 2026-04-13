"""Observation rendering for Connect 4 state."""

from hashlib import sha256

from PIL import Image, ImageDraw

from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.types import Observation, RenderedImage
from rlvr_games.games.connect4.state import (
    Board,
    Connect4State,
    EMPTY_CELL,
    PLAYER_O,
    PLAYER_X,
    inspect_connect4_state,
)


class Connect4AsciiBoardFormatter:
    """Render a Connect 4 board as ASCII text."""

    def render_text(self, board: Board) -> str:
        """Return an ASCII board diagram with column labels.

        Parameters
        ----------
        board : Board
            Board to render.

        Returns
        -------
        str
            Multi-line string containing the grid and one-based columns.
        """
        separator = "+" + ("-" * ((2 * len(board[0])) + 1)) + "+"
        lines = [separator]
        for row in board:
            lines.append("| " + " ".join(row) + " |")
        lines.append(separator)
        lines.append(
            "  " + " ".join(str(column + 1) for column in range(len(board[0])))
        )
        return "\n".join(lines)


class Connect4ImageRenderer:
    """Render Connect 4 boards as in-memory raster images."""

    _BACKGROUND = (244, 241, 232, 255)
    _BOARD = (39, 86, 196, 255)
    _EMPTY_SLOT = (236, 240, 247, 255)
    _PLAYER_X = (222, 78, 62, 255)
    _PLAYER_O = (242, 194, 63, 255)

    def __init__(self, *, size: int) -> None:
        """Initialize the image renderer.

        Parameters
        ----------
        size : int
            Width and height of the rendered raster image in pixels.
        """
        self.size = size
        self._outer_padding = max(14, size // 20)
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
        row_count = len(board)
        column_count = len(board[0])
        cell_size = self._cell_size(rows=row_count, columns=column_count)
        board_width = (column_count * cell_size) + ((column_count - 1) * self._gap)
        board_height = (row_count * cell_size) + ((row_count - 1) * self._gap)
        board_left = (self.size - board_width) // 2
        board_top = (self.size - board_height) // 2

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
                board_left + board_width + self._outer_padding,
                board_top + board_height + self._outer_padding,
            ),
            radius=min(18, self._outer_padding),
            fill=self._BOARD,
        )

        disc_radius = max(8, (cell_size // 2) - max(3, self._gap // 2))
        for row_index, row in enumerate(board):
            for column_index, cell in enumerate(row):
                center_x = board_left + (column_index * (cell_size + self._gap))
                center_y = board_top + (row_index * (cell_size + self._gap))
                center_x += cell_size // 2
                center_y += cell_size // 2
                draw.ellipse(
                    (
                        center_x - disc_radius,
                        center_y - disc_radius,
                        center_x + disc_radius,
                        center_y + disc_radius,
                    ),
                    fill=self._slot_color(cell=cell),
                )

        digest = sha256(f"{board}|{self.size}".encode("utf-8")).hexdigest()[:16]
        return (RenderedImage(key=f"connect4-board-{digest}", image=image),)

    def _cell_size(self, *, rows: int, columns: int) -> int:
        """Return the pixel size of one board cell."""
        available_width = (
            self.size - (2 * self._outer_padding) - ((columns - 1) * self._gap)
        )
        available_height = (
            self.size - (2 * self._outer_padding) - ((rows - 1) * self._gap)
        )
        return max(28, min(available_width // columns, available_height // rows))

    def _slot_color(self, *, cell: str) -> tuple[int, int, int, int]:
        """Return the fill color for one cell token."""
        if cell == PLAYER_X:
            return self._PLAYER_X
        if cell == PLAYER_O:
            return self._PLAYER_O
        if cell != EMPTY_CELL:
            raise ValueError(f"Unknown Connect 4 cell: {cell!r}.")
        return self._EMPTY_SLOT


class Connect4ObservationRenderer:
    """Render a Connect 4 observation from canonical state."""

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

    def render(self, state: Connect4State) -> Observation:
        """Render a Connect 4 state into a model-facing observation.

        Parameters
        ----------
        state : Connect4State
            Canonical Connect 4 state to render.

        Returns
        -------
        Observation
            Observation whose text and images are derived from the canonical
            board and cached state summary.
        """
        legal_actions_text = " ".join(state.legal_actions) or "none"
        lines = [
            "Connect 4 board:",
            self.board_formatter.render_text(state.board),
            f"Board size: {state.rows}x{state.columns}",
            f"Connect length: {state.connect_length}",
            f"Moves played: {state.move_count}",
            f"Current player: {state.current_player}",
            f"Legal actions ({state.legal_action_count}): {legal_actions_text}",
            f"Terminal: {'yes' if state.is_terminal else 'no'}",
        ]
        if state.is_terminal:
            lines.append(f"Winner: {state.outcome.winner or 'draw'}")
            lines.append(f"Termination: {state.outcome.termination}")

        images: tuple[RenderedImage, ...] = ()
        if self.image_renderer is not None:
            images = self.image_renderer.render_images(state.board)

        return Observation(
            text="\n".join(lines),
            images=images,
            metadata=inspect_connect4_state(state=state),
        )
