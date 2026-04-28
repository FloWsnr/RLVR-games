"""Tests for observation content helpers."""

import pytest
from zlib import compress, crc32

from rlvr_physics.core.rendering import (
    ImageContent,
    TextContent,
    image_observation,
    text_observation,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Return one PNG chunk with a valid CRC."""

    checksum = crc32(chunk_type)
    checksum = crc32(data, checksum) & 0xFFFFFFFF
    return (
        len(data).to_bytes(4, "big") + chunk_type + data + checksum.to_bytes(4, "big")
    )


VALID_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00")
    + _png_chunk(b"IDAT", compress(b"\x00\xff\xff\xff"))
    + _png_chunk(b"IEND", b"")
)


def test_text_observation_concatenates_text_blocks() -> None:
    observation = text_observation("text", "hello")

    assert observation.renderer_name == "text"
    assert observation.text() == "hello"
    assert len(observation.content_digests()) == 1


def test_image_observation_includes_digest_and_alt_text() -> None:
    observation = image_observation(
        renderer_name="image",
        data=VALID_PNG,
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


def test_image_observation_rejects_svg_mime_type() -> None:
    with pytest.raises(ValueError, match="image/png"):
        image_observation(
            renderer_name="svg",
            data=b"<svg></svg>",
            mime_type="image/svg+xml",
            alt_text="fallback",
            text="model text",
        )


def test_image_content_rejects_svg_mime_type() -> None:
    with pytest.raises(ValueError, match="image/png"):
        ImageContent(
            data=b"<svg></svg>",
            mime_type="image/svg+xml",
            alt_text="fallback",
        )


def test_image_observation_rejects_gif_mime_type() -> None:
    with pytest.raises(ValueError, match="image/png"):
        image_observation(
            renderer_name="gif",
            data=b"GIF89a",
            mime_type="image/gif",
            alt_text="fallback",
            text="model text",
        )


def test_image_content_digest_includes_visible_metadata() -> None:
    first = ImageContent(
        data=VALID_PNG,
        mime_type="image/png",
        alt_text="first",
    )
    second = ImageContent(
        data=VALID_PNG,
        mime_type="image/png",
        alt_text="second",
    )

    assert first.digest() != second.digest()


def test_image_content_rejects_mislabeled_svg_png_data() -> None:
    with pytest.raises(ValueError, match="PNG signature"):
        ImageContent(
            data=b"<svg></svg>",
            mime_type="image/png",
            alt_text="fallback",
        )


def test_image_content_rejects_signature_only_png_data() -> None:
    with pytest.raises(ValueError, match="truncated"):
        ImageContent(
            data=b"\x89PNG\r\n\x1a\nfake",
            mime_type="image/png",
            alt_text="fallback",
        )
