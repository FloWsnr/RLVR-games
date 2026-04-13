"""2048 renderer tests."""

from random import Random

from PIL import Image

from rlvr_games.core.types import RenderedImage
from rlvr_games.games.game2048 import (
    Game2048AsciiBoardFormatter,
    Game2048ImageRenderer,
    Game2048ObservationRenderer,
    Game2048State,
)

MERGE_BOARD = (
    (2, 2, 2, 2),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)


class Stub2048ImageRenderer:
    """Small stub image renderer for observation tests."""

    def render_images(
        self,
        board: tuple[tuple[int, ...], ...],
    ) -> tuple[RenderedImage, ...]:
        """Return a single fixed image payload."""
        del board
        return (
            RenderedImage(
                key="stub-2048-board",
                image=Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
            ),
        )


def test_observation_renderer_can_emit_text_and_images() -> None:
    renderer = Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=Stub2048ImageRenderer(),
    )

    observation = renderer.render(
        Game2048State(
            board=MERGE_BOARD,
            score=0,
            move_count=0,
            target_value=2048,
            rng_state=Random(0).getstate(),
        )
    )

    assert observation.text is not None
    assert "2048 board:" in observation.text
    assert len(observation.images) == 1
    assert observation.images[0].key == "stub-2048-board"
    assert observation.images[0].image.size == (32, 32)


def test_image_renderer_emits_single_raster_image() -> None:
    renderer = Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=Game2048ImageRenderer(size=360),
    )

    observation = renderer.render(
        Game2048State(
            board=MERGE_BOARD,
            score=8,
            move_count=1,
            target_value=2048,
            rng_state=Random(0).getstate(),
        )
    )

    assert len(observation.images) == 1
    rendered_image = observation.images[0]
    assert rendered_image.key.startswith("2048-board-")
    assert rendered_image.image.size == (360, 360)
    assert rendered_image.image.mode == "RGBA"


def test_image_renderer_is_deterministic_for_the_same_position() -> None:
    image_renderer = Game2048ImageRenderer(size=360)

    first_render = image_renderer.render_images(MERGE_BOARD)[0]
    second_render = image_renderer.render_images(MERGE_BOARD)[0]

    assert second_render.key == first_render.key
    assert second_render.image.tobytes() == first_render.image.tobytes()


def test_image_renderer_supports_large_tile_labels() -> None:
    renderer = Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=Game2048ImageRenderer(size=360),
    )

    observation = renderer.render(
        Game2048State(
            board=((131072, 0, 0, 0), (0, 65536, 0, 0), (0, 0, 32768, 0), (0, 0, 0, 0)),
            score=0,
            move_count=0,
            target_value=262144,
            rng_state=Random(0).getstate(),
        )
    )

    assert len(observation.images) == 1
    assert observation.images[0].image.size == (360, 360)
