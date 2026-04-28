"""Tests for shared task rendering helpers."""

from struct import unpack

import pytest

from rlvr_physics.core.rendering import PNG_MIME_TYPE
from rlvr_physics.tasks._shared.rendering import rasterize_svg

MINIMAL_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8" '
    'viewBox="0 0 12 8">'
    '<rect x="0" y="0" width="12" height="8" fill="#ffffff"/>'
    '<circle cx="6" cy="4" r="3" fill="#1f6feb"/>'
    "</svg>"
)


def test_rasterize_svg_to_png_returns_png_bytes() -> None:
    raster_image = rasterize_svg(MINIMAL_SVG)

    assert raster_image.mime_type == PNG_MIME_TYPE
    assert raster_image.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert _png_size(raster_image.data) == (12, 8)


def test_rasterize_svg_rejects_invalid_svg() -> None:
    with pytest.raises(ValueError, match="could not rasterize SVG"):
        rasterize_svg("<svg><missing>")


def _png_size(data: bytes) -> tuple[int, int]:
    """Return the width and height from a PNG IHDR chunk."""

    return unpack(">II", data[16:24])
