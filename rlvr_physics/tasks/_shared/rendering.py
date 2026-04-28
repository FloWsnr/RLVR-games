"""Shared rendering helpers for task packages."""

from dataclasses import dataclass

import cairosvg

from rlvr_physics.core.rendering import PNG_MIME_TYPE


@dataclass(frozen=True)
class RasterImage:
    """Encoded raster image data.

    Parameters
    ----------
    data:
        Encoded PNG bytes.
    mime_type:
        Media type matching ``data``.

    Attributes
    ----------
    data:
        Encoded PNG bytes.
    mime_type:
        Media type matching ``data``.
    """

    data: bytes
    mime_type: str


def rasterize_svg(svg_text: str) -> RasterImage:
    """Rasterize an SVG document into PNG bytes.

    Parameters
    ----------
    svg_text:
        Complete SVG document text.

    Returns
    -------
    RasterImage
        Encoded PNG image and its MIME type.

    Raises
    ------
    ValueError
        Raised when the SVG cannot be rasterized.
    """

    png_data = _render_svg_to_png(svg_text)
    return RasterImage(data=png_data, mime_type=PNG_MIME_TYPE)


def _render_svg_to_png(svg_text: str) -> bytes:
    """Render SVG text to PNG bytes with CairoSVG."""

    try:
        png_data = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
    except Exception as exc:
        raise ValueError("could not rasterize SVG") from exc
    if not isinstance(png_data, bytes):
        raise ValueError("could not rasterize SVG to bytes")
    return png_data
