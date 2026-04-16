"""Yahtzee renderer tests."""

from PIL import Image

from rlvr_games.core.types import RenderedImage
from rlvr_games.games.yahtzee import (
    YahtzeeDiceFormatter,
    YahtzeeImageRenderer,
    YahtzeeObservationRenderer,
    YahtzeeScorecardFormatter,
    YahtzeeState,
)
from rlvr_games.games.yahtzee.chance import YahtzeeChanceModel


class StubYahtzeeImageRenderer:
    """Small stub image renderer for observation tests."""

    def render_images(
        self,
        state: YahtzeeState,
    ) -> tuple[RenderedImage, ...]:
        """Return a single fixed image payload."""
        del state
        return (
            RenderedImage(
                key="stub-yahtzee-state",
                image=Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
            ),
        )


def make_state() -> YahtzeeState:
    """Return a representative non-terminal Yahtzee state."""
    return YahtzeeState(
        dice=(4, 4, 1, 3, 5),
        rolls_used_in_turn=1,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=YahtzeeChanceModel().initial_rng_state(seed=0),
    )


def test_observation_renderer_includes_text_metadata_and_images() -> None:
    renderer = YahtzeeObservationRenderer(
        dice_formatter=YahtzeeDiceFormatter(),
        scorecard_formatter=YahtzeeScorecardFormatter(),
        image_renderer=StubYahtzeeImageRenderer(),
    )

    observation = renderer.render(make_state())

    assert observation.text is not None
    assert "Yahtzee state:" in observation.text
    assert "Action format: score <category>" in observation.text
    assert observation.metadata["dice"] == (4, 4, 1, 3, 5)
    assert "legal_actions" not in observation.metadata
    assert len(observation.images) == 1
    assert observation.images[0].key == "stub-yahtzee-state"


def test_image_renderer_is_deterministic_for_the_same_state() -> None:
    renderer = YahtzeeImageRenderer(size=320)
    state = make_state()

    first_render = renderer.render_images(state)[0]
    second_render = renderer.render_images(state)[0]

    assert second_render.key == first_render.key
    assert second_render.image.tobytes() == first_render.image.tobytes()
