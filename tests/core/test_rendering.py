"""Tests for observation content helpers."""

from rlvr_physics.core.rendering import (
    ImageContent,
    TextContent,
    image_observation,
    text_observation,
)


def test_text_observation_concatenates_text_blocks() -> None:
    observation = text_observation("text", "hello")

    assert observation.renderer_name == "text"
    assert observation.text() == "hello"
    assert len(observation.content_digests()) == 1


def test_image_observation_includes_digest_and_alt_text() -> None:
    observation = image_observation(
        renderer_name="image",
        data=b"\x89PNG\r\n\x1a\nfake",
        mime_type="image/png",
        alt_text="image fallback",
        text="model text",
    )

    assert isinstance(observation.contents[0], ImageContent)
    assert isinstance(observation.contents[1], TextContent)
    assert observation.contents[0].mime_type == "image/png"
    assert observation.contents[0].alt_text == "image fallback"
    assert observation.text() == "model text"
    assert observation.content_digests()[0] == observation.contents[0].digest()


def test_image_observation_accepts_non_png_mime_type() -> None:
    observation = image_observation(
        renderer_name="svg",
        data=b"<svg></svg>",
        mime_type="image/svg+xml",
        alt_text="fallback",
        text="model text",
    )

    assert isinstance(observation.contents[0], ImageContent)
    assert observation.contents[0].mime_type == "image/svg+xml"


def test_image_content_digest_includes_visible_metadata() -> None:
    first = ImageContent(data=b"same-bytes", mime_type="image/png", alt_text="first")
    second = ImageContent(data=b"same-bytes", mime_type="image/png", alt_text="second")
    third = ImageContent(
        data=b"same-bytes", mime_type="image/svg+xml", alt_text="first"
    )

    assert first.digest() != second.digest()
    assert first.digest() != third.digest()
