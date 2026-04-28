"""Observation content and renderer helpers."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal


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
        Media type for the encoded bytes.
    alt_text:
        Short model-facing text fallback.
    kind:
        Discriminator for image content blocks.
    """

    data: bytes
    mime_type: str
    alt_text: str
    kind: Literal["image"] = "image"

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
        Media type for the encoded image bytes.
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
