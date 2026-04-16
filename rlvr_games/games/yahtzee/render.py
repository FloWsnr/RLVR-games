"""Observation rendering for Yahtzee state."""

from hashlib import sha256

from PIL import Image, ImageDraw

from rlvr_games.core.protocol import ImageRenderer, TextRenderer
from rlvr_games.core.types import Observation, RenderedImage
from rlvr_games.games.yahtzee.engine import CATEGORY_ORDER, CategoryScores, Dice
from rlvr_games.games.yahtzee.state import (
    YahtzeeState,
    public_yahtzee_metadata,
)


class YahtzeeDiceFormatter:
    """Render Yahtzee dice as compact ASCII text."""

    def render_text(self, dice: Dice) -> str:
        """Return a small labeled dice summary.

        Parameters
        ----------
        dice : Dice
            Five die values to render.

        Returns
        -------
        str
            Multi-line one-based position and value summary.
        """
        positions = " ".join(str(index) for index in range(1, 6))
        values = " ".join("-" if value == 0 else str(value) for value in dice)
        return f"Positions: {positions}\nDice:      {values}"


class YahtzeeScorecardFormatter:
    """Render a Yahtzee scorecard as aligned ASCII text."""

    def render_text(self, scores: CategoryScores) -> str:
        """Return a scorecard table aligned with canonical category order.

        Parameters
        ----------
        scores : CategoryScores
            Scorecard values aligned with `CATEGORY_ORDER`.

        Returns
        -------
        str
            Multi-line scorecard summary.
        """
        label_width = max(len(category.value) for category in CATEGORY_ORDER)
        return "\n".join(
            (
                f"{category.value.ljust(label_width)} : "
                f"{'-' if scores[index] is None else scores[index]}"
            )
            for index, category in enumerate(CATEGORY_ORDER)
        )


class YahtzeeImageRenderer:
    """Render Yahtzee dice and score summary as one raster image."""

    _BACKGROUND = (245, 242, 234, 255)
    _DIE_FILL = (255, 255, 255, 255)
    _DIE_OUTLINE = (58, 55, 49, 255)
    _PIP = (41, 39, 36, 255)
    _HEADER = (94, 55, 44, 255)
    _TEXT = (41, 39, 36, 255)
    _TERMINAL = (79, 118, 81, 255)

    def __init__(self, *, size: int) -> None:
        """Initialize the image renderer.

        Parameters
        ----------
        size : int
            Width and height of the rendered raster image in pixels.
        """
        self.size = size

    def render_images(self, state: YahtzeeState) -> tuple[RenderedImage, ...]:
        """Render the supplied state into one raster summary image.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state to render.

        Returns
        -------
        tuple[RenderedImage, ...]
            Single-item tuple containing the rendered image payload.
        """
        image = Image.new("RGBA", (self.size, self.size), self._BACKGROUND)
        draw = ImageDraw.Draw(image, "RGBA")
        self._draw_header(draw=draw, state=state)
        self._draw_dice(draw=draw, state=state)

        digest = sha256(
            (
                f"{state.dice}|{state.total_score}|{state.turn_number}|"
                f"{state.rolls_used_in_turn}|{state.is_terminal}|{self.size}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        return (RenderedImage(key=f"yahtzee-state-{digest}", image=image),)

    def _draw_header(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        state: YahtzeeState,
    ) -> None:
        """Draw the top summary header."""
        draw.rounded_rectangle(
            (12, 12, self.size - 12, 76),
            radius=16,
            fill=self._TERMINAL if state.is_terminal else self._HEADER,
        )
        draw.text(
            (26, 24),
            f"Turn {state.turn_number}/13",
            fill=(255, 255, 255, 255),
        )
        draw.text(
            (26, 48),
            f"Score {state.total_score}",
            fill=(255, 255, 255, 255),
        )

    def _draw_dice(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        state: YahtzeeState,
    ) -> None:
        """Draw five dice centered in the image."""
        die_size = max(40, self.size // 7)
        gap = max(10, self.size // 36)
        total_width = (5 * die_size) + (4 * gap)
        left = (self.size - total_width) // 2
        top = max(108, self.size // 3)

        for index, value in enumerate(state.dice):
            x0 = left + (index * (die_size + gap))
            y0 = top
            x1 = x0 + die_size
            y1 = y0 + die_size
            draw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=max(8, die_size // 8),
                fill=self._DIE_FILL,
                outline=self._DIE_OUTLINE,
                width=2,
            )
            if value == 0:
                draw.text(
                    (x0 + (die_size // 2) - 3, y0 + (die_size // 2) - 6),
                    "?",
                    fill=self._TEXT,
                )
                continue
            self._draw_pips(draw=draw, value=value, left=x0, top=y0, size=die_size)
            draw.text(
                (x0 + (die_size // 2) - 3, y1 + 8),
                str(index + 1),
                fill=self._TEXT,
            )

    def _draw_pips(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        value: int,
        left: int,
        top: int,
        size: int,
    ) -> None:
        """Draw the pips for one die value."""
        pip_radius = max(4, size // 14)
        low = left + int(size * 0.28)
        mid = left + (size // 2)
        high = left + int(size * 0.72)
        upper = top + int(size * 0.28)
        center = top + (size // 2)
        lower = top + int(size * 0.72)

        pip_positions = {
            1: ((mid, center),),
            2: ((low, upper), (high, lower)),
            3: ((low, upper), (mid, center), (high, lower)),
            4: ((low, upper), (high, upper), (low, lower), (high, lower)),
            5: (
                (low, upper),
                (high, upper),
                (mid, center),
                (low, lower),
                (high, lower),
            ),
            6: (
                (low, upper),
                (high, upper),
                (low, center),
                (high, center),
                (low, lower),
                (high, lower),
            ),
        }
        for x, y in pip_positions[value]:
            draw.ellipse(
                (x - pip_radius, y - pip_radius, x + pip_radius, y + pip_radius),
                fill=self._PIP,
            )


class YahtzeeObservationRenderer:
    """Render a Yahtzee observation from canonical state."""

    def __init__(
        self,
        *,
        dice_formatter: TextRenderer[Dice],
        scorecard_formatter: TextRenderer[CategoryScores],
        image_renderer: ImageRenderer[YahtzeeState] | None,
    ) -> None:
        """Initialize the observation renderer with explicit view components.

        Parameters
        ----------
        dice_formatter : TextRenderer[Dice]
            Text renderer used to build the dice view.
        scorecard_formatter : TextRenderer[CategoryScores]
            Text renderer used to build the scorecard view.
        image_renderer : ImageRenderer[YahtzeeState] | None
            Optional image renderer used to produce raster observations.
        """
        self.dice_formatter = dice_formatter
        self.scorecard_formatter = scorecard_formatter
        self.image_renderer = image_renderer

    def render(self, state: YahtzeeState) -> Observation:
        """Render a Yahtzee state into a model-facing observation.

        Parameters
        ----------
        state : YahtzeeState
            Canonical Yahtzee state to render.

        Returns
        -------
        Observation
            Observation derived from the canonical dice, scorecard, and public
            metadata.
        """
        lines = [
            "Yahtzee state:",
            (
                "Opening roll pending."
                if state.awaiting_roll
                else self.dice_formatter.render_text(state.dice)
            ),
            f"Turn: {state.turn_number}/13",
            f"Rolls used this turn: {state.rolls_used_in_turn}",
            f"Rerolls remaining: {state.rerolls_remaining}",
            f"Turns completed: {state.turns_completed}",
            f"Total score: {state.total_score}",
            (
                "Available score options: "
                + (
                    "none"
                    if not state.available_score_options
                    else ", ".join(
                        (
                            f"{category}={score}"
                            for category, score in state.available_score_options.items()
                        )
                    )
                )
            ),
            "Scorecard:",
            self.scorecard_formatter.render_text(state.category_scores),
            "Action format: score <category> | reroll <positions>",
            f"Terminal: {'yes' if state.is_terminal else 'no'}",
        ]
        if state.is_terminal:
            lines.append(f"Final score: {state.outcome.final_score}")
            lines.append(f"Termination: {state.outcome.termination}")

        images: tuple[RenderedImage, ...] = ()
        if self.image_renderer is not None:
            images = self.image_renderer.render_images(state)

        return Observation(
            text="\n".join(lines),
            images=images,
            metadata=public_yahtzee_metadata(state=state),
        )
