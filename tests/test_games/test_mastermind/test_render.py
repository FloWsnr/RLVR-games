"""Mastermind renderer tests."""

from rlvr_games.games.mastermind import (
    MastermindAsciiBoardFormatter,
    MastermindGuessRecord,
    MastermindImageRenderer,
    MastermindObservationRenderer,
    MastermindState,
)


def test_observation_renderer_hides_secret_code_in_text_and_metadata() -> None:
    renderer = MastermindObservationRenderer(
        board_formatter=MastermindAsciiBoardFormatter(),
        image_renderer=MastermindImageRenderer(size=320),
    )
    state = MastermindState(
        secret_code=(6, 6, 5, 5),
        guess_history=(
            MastermindGuessRecord(
                guess=(6, 5, 6, 5),
                black_pegs=2,
                white_pegs=2,
            ),
        ),
    )

    observation = renderer.render(state)

    assert "Mastermind board:" in (observation.text or "")
    assert "6 6 5 5" not in (observation.text or "")
    assert "secret_code" not in observation.metadata
    assert observation.metadata["candidate_count"] == state.candidate_count
    assert len(observation.images) == 1
    assert observation.images[0].key.startswith("mastermind-board-")


def test_image_renderer_is_deterministic_for_the_same_state() -> None:
    renderer = MastermindImageRenderer(size=320)
    state = MastermindState(
        secret_code=(1, 1, 2, 2),
        guess_history=(
            MastermindGuessRecord(
                guess=(1, 2, 1, 2),
                black_pegs=2,
                white_pegs=2,
            ),
        ),
    )

    first_render = renderer.render_images(state)[0]
    second_render = renderer.render_images(state)[0]

    assert second_render.key == first_render.key
    assert second_render.image.tobytes() == first_render.image.tobytes()


def test_image_renderer_supports_smallest_allowed_size() -> None:
    renderer = MastermindImageRenderer(size=128)
    state = MastermindState(secret_code=(1, 1, 2, 2))

    rendered_image = renderer.render_images(state)[0]

    assert rendered_image.image.size == (128, 128)
