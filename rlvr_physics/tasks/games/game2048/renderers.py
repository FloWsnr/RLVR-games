"""Renderers for seeded 2048."""

from PIL import Image, ImageDraw, ImageFont

from rlvr_physics.core.rendering import (
    RenderedObservation,
    image_observation,
    text_observation,
)
from rlvr_physics.tasks._shared.images import draw_centered_text, encode_png
from rlvr_physics.tasks.games.game2048.types import Game2048State


def render_2048_text(state: Game2048State, target_tile: int) -> RenderedObservation:
    """Render a 2048 state as plain text."""

    rows = []
    for row in state.board:
        rows.append(" ".join(f"{value:4d}" if value else "   ." for value in row))
    prompt = (
        "2048\n"
        f"Score: {state.score} | Moves: {state.turns} | Max tile: {state.max_tile} | Target: {target_tile}\n\n"
        + "\n".join(rows)
        + "\n\nSubmit one action: up, down, left, or right."
    )
    return text_observation("text", prompt)


def render_2048_image(state: Game2048State, target_tile: int) -> RenderedObservation:
    """Render a 2048 state as a PNG image observation."""

    tile = 104
    gap = 12
    margin = 32
    header = 92
    board_px = 4 * tile + 3 * gap
    width = board_px + 2 * margin
    height = board_px + 2 * margin + header
    image = Image.new("RGB", (width, height), "#f3f5f7")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default(size=30)
    font_tile = ImageFont.load_default(size=36)
    font_body = ImageFont.load_default(size=20)
    draw.text((margin, 24), "2048", fill="#1f2937", font=font_title)
    draw.text(
        (margin + 104, 31),
        f"Score {state.score}  Moves {state.turns}  Target {target_tile}",
        fill="#334155",
        font=font_body,
    )
    palette = {
        0: ("#d7dde3", "#d7dde3"),
        2: ("#eef2ff", "#1f2937"),
        4: ("#e0f2fe", "#1f2937"),
        8: ("#bbf7d0", "#14532d"),
        16: ("#fde68a", "#713f12"),
        32: ("#fed7aa", "#7c2d12"),
        64: ("#fecaca", "#7f1d1d"),
        128: ("#ddd6fe", "#312e81"),
        256: ("#c7d2fe", "#1e3a8a"),
        512: ("#bfdbfe", "#172554"),
        1024: ("#a7f3d0", "#064e3b"),
        2048: ("#fef08a", "#713f12"),
    }
    for row_index, row in enumerate(state.board):
        for col_index, value in enumerate(row):
            x = margin + col_index * (tile + gap)
            y = margin + header + row_index * (tile + gap)
            fill, text_color = palette.get(value, ("#99f6e4", "#134e4a"))
            box = (x, y, x + tile, y + tile)
            draw.rounded_rectangle(box, radius=8, fill=fill)
            if value:
                draw_centered_text(draw, box, str(value), font_tile, text_color, -4)
    return image_observation(
        "image", encode_png(image), render_2048_text(state, target_tile).text()
    )
