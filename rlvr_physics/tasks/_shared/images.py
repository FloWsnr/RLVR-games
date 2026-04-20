"""Image rendering helpers shared by task renderers."""

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


def encode_png(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes.

    Parameters
    ----------
    image:
        Image to encode.
    """

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: str,
    y_offset: float,
) -> None:
    """Draw text centered within a rectangular box.

    Parameters
    ----------
    draw:
        Active PIL drawing context.
    box:
        Rectangle as ``(left, top, right, bottom)``.
    text:
        Text to draw.
    font:
        Font used for measurement and rendering.
    fill:
        Text color.
    y_offset:
        Vertical adjustment applied after centering.
    """

    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        (
            left + ((right - left) - text_width) / 2,
            top + ((bottom - top) - text_height) / 2 + y_offset,
        ),
        text,
        fill=fill,
        font=font,
    )
