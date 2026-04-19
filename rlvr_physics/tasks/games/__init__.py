"""Small game and puzzle tasks used as architecture probes."""

from rlvr_physics.tasks.games.chess_tactics import (
    ChessTacticsSession,
    chess_tactics_task_spec,
    make_chess_tactics_instance,
    render_chess_tactics_image,
    render_chess_tactics_text,
)
from rlvr_physics.tasks.games.countdown import (
    CountdownSession,
    countdown_task_spec,
    make_countdown_instance,
    render_countdown_image,
    render_countdown_text,
)
from rlvr_physics.tasks.games.game2048 import (
    Game2048Session,
    game2048_task_spec,
    make_2048_instance,
    render_2048_image,
    render_2048_text,
)

__all__ = [
    "ChessTacticsSession",
    "CountdownSession",
    "Game2048Session",
    "chess_tactics_task_spec",
    "countdown_task_spec",
    "game2048_task_spec",
    "make_2048_instance",
    "make_chess_tactics_instance",
    "make_countdown_instance",
    "render_2048_image",
    "render_2048_text",
    "render_chess_tactics_image",
    "render_chess_tactics_text",
    "render_countdown_image",
    "render_countdown_text",
]
