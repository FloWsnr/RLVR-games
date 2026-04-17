"""Observation rendering for Mastermind state."""

from functools import cached_property
from hashlib import sha256

from PIL import Image, ImageDraw, ImageFont

from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.types import Observation, RenderedImage
from rlvr_games.games.mastermind.engine import (
    STANDARD_MASTERMIND_MIN_IMAGE_SIZE,
    format_code,
)
from rlvr_games.games.mastermind.state import (
    MastermindGuessRecord,
    MastermindState,
    public_mastermind_metadata,
)


class MastermindAsciiBoardFormatter:
    """Render Mastermind guess history as ASCII text."""

    def render_text(self, state: MastermindState) -> str:
        """Return a compact text table of guesses and feedback."""
        lines = ["Turn | Guess   | Feedback", "---- | ------- | --------"]
        if not state.guess_history:
            lines.append("  -  | (none)  | -")
            return "\n".join(lines)

        for turn_index, record in enumerate(state.guess_history, start=1):
            lines.append(
                f"{turn_index:>4} | {format_code(code=record.guess):<7} | "
                f"{record.black_pegs}B {record.white_pegs}W"
            )
        return "\n".join(lines)


class MastermindImageRenderer:
    """Render Mastermind boards as in-memory raster images."""

    _BACKGROUND = (247, 243, 233, 255)
    _BOARD = (231, 222, 204, 255)
    _ROW_FILL = (245, 241, 232, 255)
    _ROW_OUTLINE = (156, 145, 126, 255)
    _EMPTY_PEG = (222, 214, 197, 255)
    _EMPTY_OUTLINE = (170, 160, 145, 255)
    _BLACK_FEEDBACK = (28, 28, 28, 255)
    _WHITE_FEEDBACK = (252, 252, 250, 255)
    _FEEDBACK_OUTLINE = (97, 92, 85, 255)
    _ROW_NUMBER = (85, 74, 63, 255)
    _PEG_COLORS = {
        1: (205, 70, 56, 255),
        2: (76, 121, 59, 255),
        3: (63, 99, 176, 255),
        4: (214, 171, 54, 255),
        5: (129, 74, 166, 255),
        6: (229, 137, 42, 255),
    }

    def __init__(self, *, size: int) -> None:
        """Initialize the image renderer."""
        if size < STANDARD_MASTERMIND_MIN_IMAGE_SIZE:
            raise ValueError(
                "Mastermind image_size must be at least "
                f"{STANDARD_MASTERMIND_MIN_IMAGE_SIZE}."
            )
        self.size = size
        self._outer_padding = max(8, size // 24)
        self._row_gap = max(2, size // 80)
        self._peg_gap = max(2, size // 96)

    def render_images(self, state: MastermindState) -> tuple[RenderedImage, ...]:
        """Render one Mastermind board image."""
        row_count = state.max_guesses
        board_width = self.size - (2 * self._outer_padding)
        board_height = self.size - (2 * self._outer_padding)
        row_height = (board_height - ((row_count - 1) * self._row_gap)) // row_count
        number_width = max(18, self.size // 12)
        feedback_width = max(30, self.size // 7)
        peg_area_width = (
            board_width - number_width - feedback_width - (2 * self._peg_gap)
        )
        peg_diameter = min(
            row_height - (2 * max(3, row_height // 10)),
            (peg_area_width - (3 * self._peg_gap)) // 4,
        )
        feedback_diameter = max(8, min(row_height // 4, feedback_width // 3))

        image = Image.new("RGBA", (self.size, self.size), self._BACKGROUND)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle(
            (
                self._outer_padding,
                self._outer_padding,
                self.size - self._outer_padding,
                self.size - self._outer_padding,
            ),
            radius=max(8, self._outer_padding // 2),
            fill=self._BOARD,
            outline=self._ROW_OUTLINE,
            width=2,
        )

        history_by_row = {
            row_index: record for row_index, record in enumerate(state.guess_history)
        }
        board_left = self._outer_padding + max(2, self.size // 32)
        for row_index in range(row_count):
            row_top = self._outer_padding + (row_index * (row_height + self._row_gap))
            row_bottom = row_top + row_height
            row_left = board_left
            row_right = self.size - self._outer_padding - max(2, self.size // 32)
            draw.rounded_rectangle(
                (row_left, row_top, row_right, row_bottom),
                radius=max(4, row_height // 6),
                fill=self._ROW_FILL,
                outline=self._ROW_OUTLINE,
                width=1,
            )
            self._draw_row_number(
                draw=draw,
                number=row_index + 1,
                left=row_left,
                top=row_top,
                width=number_width,
                row_height=row_height,
            )

            row_record = history_by_row.get(row_index)
            peg_left = row_left + number_width + self._peg_gap
            for peg_index in range(4):
                peg_x0 = peg_left + (peg_index * (peg_diameter + self._peg_gap))
                peg_y0 = row_top + ((row_height - peg_diameter) // 2)
                peg_x1 = peg_x0 + peg_diameter
                peg_y1 = peg_y0 + peg_diameter
                if row_record is None:
                    fill = self._EMPTY_PEG
                else:
                    fill = self._PEG_COLORS[row_record.guess[peg_index]]
                draw.ellipse(
                    (peg_x0, peg_y0, peg_x1, peg_y1),
                    fill=fill,
                    outline=self._EMPTY_OUTLINE,
                    width=2,
                )

            feedback_left = row_right - feedback_width + self._peg_gap
            feedback_tokens = self._feedback_tokens(record=row_record)
            for feedback_index in range(4):
                feedback_row = feedback_index // 2
                feedback_column = feedback_index % 2
                feedback_x0 = feedback_left + (
                    feedback_column * (feedback_diameter + self._peg_gap)
                )
                feedback_y0 = (
                    row_top
                    + max(6, row_height // 4)
                    + (feedback_row * (feedback_diameter + self._peg_gap))
                )
                feedback_x1 = feedback_x0 + feedback_diameter
                feedback_y1 = feedback_y0 + feedback_diameter
                token = feedback_tokens[feedback_index]
                if token == "black":
                    fill = self._BLACK_FEEDBACK
                elif token == "white":
                    fill = self._WHITE_FEEDBACK
                else:
                    fill = self._EMPTY_PEG
                draw.ellipse(
                    (feedback_x0, feedback_y0, feedback_x1, feedback_y1),
                    fill=fill,
                    outline=self._FEEDBACK_OUTLINE,
                    width=1,
                )

        digest = sha256(
            f"{state.guess_history}|{state.max_guesses}|{self.size}".encode("utf-8")
        ).hexdigest()[:16]
        return (RenderedImage(key=f"mastermind-board-{digest}", image=image),)

    @cached_property
    def _font_name(self) -> str:
        """Return the preferred truetype font name."""
        return "DejaVuSans-Bold.ttf"

    def _draw_row_number(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        number: int,
        left: int,
        top: int,
        width: int,
        row_height: int,
    ) -> None:
        """Draw one row number centered in its lane."""
        font = ImageFont.truetype(self._font_name, max(12, row_height // 2))
        text = str(number)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = int(left + ((width - text_width) / 2) - text_bbox[0])
        y = int(top + ((row_height - text_height) / 2) - text_bbox[1])
        draw.text((x, y), text, font=font, fill=self._ROW_NUMBER)

    def _feedback_tokens(
        self,
        *,
        record: MastermindGuessRecord | None,
    ) -> tuple[str, str, str, str]:
        """Return the 2x2 feedback token layout for one row."""
        if record is None:
            return ("empty", "empty", "empty", "empty")
        tokens = ["black"] * record.black_pegs + ["white"] * record.white_pegs
        while len(tokens) < 4:
            tokens.append("empty")
        return (tokens[0], tokens[1], tokens[2], tokens[3])


class MastermindObservationRenderer:
    """Render a Mastermind observation from canonical state."""

    def __init__(
        self,
        *,
        board_formatter: TextRenderer[MastermindState],
        image_renderer: ImageRenderer[MastermindState] | None,
    ) -> None:
        """Initialize the observation renderer with explicit view components."""
        self.board_formatter = board_formatter
        self.image_renderer = image_renderer

    def render(self, state: MastermindState) -> Observation:
        """Render a Mastermind state into a model-facing observation."""
        lines = [
            "Mastermind board:",
            self.board_formatter.render_text(state),
            "Canonical peg labels: digits 1-6",
            "Code length: 4",
            f"Guesses used: {state.move_count}",
            f"Guesses remaining: {state.guesses_remaining}",
            f"Consistent candidates: {state.candidate_count}",
            (
                "Action format: guess <d1> <d2> <d3> <d4> "
                "(for example `guess 1 1 2 2`; bare `1122` is also accepted)"
            ),
            f"Terminal: {'yes' if state.is_terminal else 'no'}",
            f"Cracked: {'yes' if state.outcome.won else 'no'}",
        ]
        if state.is_terminal:
            lines.append(f"Termination: {state.outcome.termination}")

        images: tuple[RenderedImage, ...] = ()
        if self.image_renderer is not None:
            images = self.image_renderer.render_images(state)

        return Observation(
            text="\n".join(lines),
            images=images,
            metadata=public_mastermind_metadata(state),
        )
