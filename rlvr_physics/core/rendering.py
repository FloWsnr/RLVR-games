"""Observation content and renderer helpers."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal


@dataclass(frozen=True)
class TextContent:
    """A text observation block.

    Parameters
    ----------
    text:
        Model-facing text content.
    """

    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class ImageContent:
    """An image observation block.

    Parameters
    ----------
    data:
        Encoded image bytes.
    mime_type:
        Media type for the encoded bytes.
    alt_text:
        Short model-facing text fallback.
    """

    data: bytes
    mime_type: str
    alt_text: str
    kind: Literal["image"] = "image"

    def digest(self) -> str:
        """Return a stable digest for trajectory logging."""

        return sha256(self.data).hexdigest()


ObservationContent = TextContent | ImageContent


@dataclass(frozen=True)
class RenderedObservation:
    """A deterministic observation emitted by a renderer.

    Parameters
    ----------
    renderer_name:
        Renderer identifier, such as ``text`` or ``image``.
    contents:
        Ordered content blocks shown to the model.
    """

    renderer_name: str
    contents: tuple[ObservationContent, ...]

    def text(self) -> str:
        """Return all text blocks concatenated with blank lines."""

        return "\n\n".join(
            content.text
            for content in self.contents
            if isinstance(content, TextContent)
        )

    def content_digests(self) -> tuple[str, ...]:
        """Return compact content hashes suitable for public trajectory logs."""

        digests: list[str] = []
        for content in self.contents:
            if isinstance(content, TextContent):
                digests.append(sha256(content.text.encode("utf-8")).hexdigest())
            else:
                digests.append(content.digest())
        return tuple(digests)


def text_observation(renderer_name: str, text: str) -> RenderedObservation:
    """Build a text-only observation."""

    return RenderedObservation(
        renderer_name=renderer_name, contents=(TextContent(text=text),)
    )


def image_observation(
    renderer_name: str, data: bytes, alt_text: str
) -> RenderedObservation:
    """Build an image observation with a text fallback."""

    return RenderedObservation(
        renderer_name=renderer_name,
        contents=(
            ImageContent(data=data, mime_type="image/png", alt_text=alt_text),
            TextContent(text=alt_text),
        ),
    )
