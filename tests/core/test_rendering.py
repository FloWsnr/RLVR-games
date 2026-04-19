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
    observation = image_observation("image", b"\x89PNG\r\n\x1a\nfake", "fallback")

    assert isinstance(observation.contents[0], ImageContent)
    assert isinstance(observation.contents[1], TextContent)
    assert observation.contents[0].mime_type == "image/png"
    assert observation.text() == "fallback"
    assert observation.content_digests()[0] == observation.contents[0].digest()
