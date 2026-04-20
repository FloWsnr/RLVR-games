"""Renderers for chess tactics."""

import chess
import chess.svg

from rlvr_physics.core.instances import TaskInstance, require_int, require_str
from rlvr_physics.core.rendering import (
    ImageContent,
    RenderedObservation,
    TextContent,
    text_observation,
)


def render_chess_tactics_text(instance: TaskInstance) -> RenderedObservation:
    """Render a chess tactic as plain text."""

    fen = require_str(instance.public_payload["fen"], "fen")
    side_to_move = require_str(instance.public_payload["side_to_move"], "side_to_move")
    mate_depth = require_int(instance.public_payload["mate_depth"], "mate_depth")
    board = chess.Board(fen)
    prompt = (
        f"Chess tactic: mate in {mate_depth}\n"
        f"Side to move: {side_to_move}\n"
        f"FEN: {fen}\n\n"
        f"{board}\n\n"
        "Submit the winning move in SAN or UCI notation."
    )
    return text_observation("text", prompt)


def render_chess_tactics_image(instance: TaskInstance) -> RenderedObservation:
    """Render a chess tactic as an SVG image observation."""

    fen = require_str(instance.public_payload["fen"], "fen")
    board = chess.Board(fen)
    alt_text = render_chess_tactics_text(instance).text()
    svg = chess.svg.board(board=board, orientation=chess.WHITE, size=640)
    return RenderedObservation(
        renderer_name="image",
        contents=(
            ImageContent(
                data=svg.encode("utf-8"),
                mime_type="image/svg+xml",
                alt_text=alt_text,
            ),
            TextContent(text=alt_text),
        ),
    )
