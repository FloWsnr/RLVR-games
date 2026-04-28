"""Observation content and renderer helpers."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal
from zlib import crc32

PNG_MIME_TYPE = "image/png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IHDR_CHUNK = b"IHDR"
PNG_IDAT_CHUNK = b"IDAT"
PNG_IEND_CHUNK = b"IEND"
PNG_IHDR_LENGTH = 13
PNG_CHUNK_HEADER_LENGTH = 8
PNG_CHUNK_CRC_LENGTH = 4


@dataclass(frozen=True)
class TextContent:
    """A text observation block.

    Attributes
    ----------
    text:
        Model-facing text content.
    kind:
        Discriminator for text content blocks.
    """

    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class ImageContent:
    """An image observation block.

    Attributes
    ----------
    data:
        Encoded image bytes.
    mime_type:
        Media type for the encoded bytes. Must be ``image/png``.
    alt_text:
        Short model-facing text fallback.
    kind:
        Discriminator for image content blocks.
    """

    data: bytes
    mime_type: str
    alt_text: str
    kind: Literal["image"] = "image"

    def __post_init__(self) -> None:
        """Validate that image observations use PNG multimodal payloads."""

        validate_png_image_data(self.data, self.mime_type)

    def digest(self) -> str:
        """Return a stable digest for compact logging.

        Returns
        -------
        str
            SHA-256 hex digest of the encoded image bytes and model-visible
            image metadata.
        """

        digest = sha256()
        digest.update(self.mime_type.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.alt_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.data)
        return digest.hexdigest()


ObservationContent = TextContent | ImageContent


@dataclass(frozen=True)
class RenderedObservation:
    """A deterministic observation emitted by a renderer.

    Attributes
    ----------
    renderer_name:
        Renderer identifier, such as ``text`` or ``image``.
    contents:
        Ordered content blocks shown to the model.
    """

    renderer_name: str
    contents: tuple[ObservationContent, ...]

    def text(self) -> str:
        """Return all text blocks concatenated with blank lines.

        Returns
        -------
        str
            Text content blocks joined by two newline characters. Image-only
            observations return an empty string.
        """

        return "\n\n".join(
            content.text
            for content in self.contents
            if isinstance(content, TextContent)
        )

    def content_digests(self) -> tuple[str, ...]:
        """Return compact content hashes for observation content.

        Returns
        -------
        tuple[str, ...]
            SHA-256 hex digests for each content block in order. Text blocks
            are hashed from UTF-8 text; image blocks use their encoded bytes.
        """

        digests: list[str] = []
        for content in self.contents:
            if isinstance(content, TextContent):
                digests.append(sha256(content.text.encode("utf-8")).hexdigest())
            else:
                digests.append(content.digest())
        return tuple(digests)


def text_observation(renderer_name: str, text: str) -> RenderedObservation:
    """Build a text-only observation.

    Parameters
    ----------
    renderer_name:
        Name to store on the rendered observation.
    text:
        Model-facing text content.

    Returns
    -------
    RenderedObservation
        Observation containing one ``TextContent`` block.
    """

    return RenderedObservation(
        renderer_name=renderer_name, contents=(TextContent(text=text),)
    )


def image_observation(
    renderer_name: str, data: bytes, mime_type: str, alt_text: str, text: str
) -> RenderedObservation:
    """Build an image observation with a separate text block.

    Parameters
    ----------
    renderer_name:
        Name to store on the rendered observation.
    data:
        Encoded image bytes.
    mime_type:
        Media type for the encoded image bytes. Must be ``image/png``.
    alt_text:
        Concise text alternative for the image content block.
    text:
        Model-facing text content appended as a separate ``TextContent`` block.

    Returns
    -------
    RenderedObservation
        Observation containing an ``ImageContent`` block followed by a text
        block.
    """

    return RenderedObservation(
        renderer_name=renderer_name,
        contents=(
            ImageContent(data=data, mime_type=mime_type, alt_text=alt_text),
            TextContent(text=text),
        ),
    )


def validate_png_image_mime_type(mime_type: str) -> None:
    """Validate that an image content block is PNG.

    Parameters
    ----------
    mime_type:
        Media type for encoded image bytes.

    Raises
    ------
    ValueError
        Raised when the media type is not PNG.
    """

    if mime_type != PNG_MIME_TYPE:
        raise ValueError(f"image observations must use {PNG_MIME_TYPE}")


def validate_png_image_data(data: bytes, mime_type: str) -> None:
    """Validate that encoded image bytes match PNG.

    Parameters
    ----------
    data:
        Encoded image bytes.
    mime_type:
        Media type for encoded image bytes.

    Raises
    ------
    ValueError
        Raised when the media type is unsupported or the bytes do not match PNG.
    """

    validate_png_image_mime_type(mime_type)
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("image/png data must start with the PNG signature")
    _validate_png_chunks(data)


def _validate_png_chunks(data: bytes) -> None:
    """Validate basic PNG chunk structure and CRCs."""

    offset = len(PNG_SIGNATURE)
    seen_ihdr = False
    seen_idat = False
    while True:
        if offset + PNG_CHUNK_HEADER_LENGTH > len(data):
            raise ValueError("image/png data is truncated")

        chunk_length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + PNG_CHUNK_HEADER_LENGTH]
        chunk_data_start = offset + PNG_CHUNK_HEADER_LENGTH
        chunk_data_end = chunk_data_start + chunk_length
        chunk_crc_end = chunk_data_end + PNG_CHUNK_CRC_LENGTH
        if chunk_crc_end > len(data):
            raise ValueError("image/png data is truncated")

        chunk_data = data[chunk_data_start:chunk_data_end]
        expected_crc = int.from_bytes(data[chunk_data_end:chunk_crc_end], "big")
        actual_crc = crc32(chunk_type)
        actual_crc = crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("image/png data has invalid chunk CRC")

        if not seen_ihdr:
            if chunk_type != PNG_IHDR_CHUNK or chunk_length != PNG_IHDR_LENGTH:
                raise ValueError("image/png data must start with an IHDR chunk")
            seen_ihdr = True
        elif chunk_type == PNG_IHDR_CHUNK:
            raise ValueError("image/png data contains multiple IHDR chunks")

        if chunk_type == PNG_IDAT_CHUNK:
            seen_idat = True
        if chunk_type == PNG_IEND_CHUNK:
            if chunk_length != 0:
                raise ValueError("image/png IEND chunk must be empty")
            if not seen_idat:
                raise ValueError("image/png data must contain an IDAT chunk")
            if chunk_crc_end != len(data):
                raise ValueError("image/png data has bytes after IEND")
            return

        offset = chunk_crc_end
