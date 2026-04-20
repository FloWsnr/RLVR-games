"""Renderers for Countdown observations."""

from PIL import Image, ImageDraw, ImageFont

from rlvr_physics.core.instances import TaskInstance, require_int, require_tuple_of_ints
from rlvr_physics.core.rendering import (
    RenderedObservation,
    image_observation,
    text_observation,
)
from rlvr_physics.tasks._shared.images import draw_centered_text, encode_png


def render_countdown_text(instance: TaskInstance) -> RenderedObservation:
    """Render a Countdown instance as plain text."""

    numbers = require_tuple_of_ints(instance.public_payload["numbers"], "numbers")
    target = require_int(instance.public_payload["target"], "target")
    prompt = (
        "Countdown numbers game\n"
        f"Target: {target}\n"
        f"Numbers: {', '.join(str(number) for number in numbers)}\n\n"
        "Submit one arithmetic expression that uses every listed number exactly once "
        "and evaluates to the target. Allowed operators: +, -, *, /, and parentheses."
    )
    return text_observation("text", prompt)


def render_countdown_image(instance: TaskInstance) -> RenderedObservation:
    """Render a Countdown instance as a PNG image observation."""

    numbers = require_tuple_of_ints(instance.public_payload["numbers"], "numbers")
    target = require_int(instance.public_payload["target"], "target")
    image = Image.new("RGB", (720, 420), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default(size=34)
    font_large = ImageFont.load_default(size=54)
    font_body = ImageFont.load_default(size=24)

    draw.rectangle((0, 0, 720, 420), fill="#f7f8fa")
    draw.text((40, 34), "Countdown", fill="#1c2331", font=font_title)
    draw.text((40, 92), f"Target {target}", fill="#0c4a6e", font=font_large)

    tile_width = 96
    tile_height = 72
    gap = 18
    total_width = len(numbers) * tile_width + (len(numbers) - 1) * gap
    start_x = (720 - total_width) // 2
    y = 210
    for index, number in enumerate(numbers):
        x = start_x + index * (tile_width + gap)
        box = (x, y, x + tile_width, y + tile_height)
        draw.rounded_rectangle(
            box,
            radius=8,
            fill="#ffffff",
            outline="#8aa0b4",
            width=2,
        )
        draw_centered_text(draw, box, str(number), font_large, "#1f2937", -4)

    draw.text(
        (40, 340), "Use each number once with + - * /", fill="#364152", font=font_body
    )
    alt_text = render_countdown_text(instance).text()
    return image_observation("image", encode_png(image), alt_text)
