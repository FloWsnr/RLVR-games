"""Seeded 2048 task package."""

from rlvr_physics.tasks.games.game2048.constants import (
    GAME2048_ACTIONS,
    GAME2048_DOMAIN,
    GAME2048_KIND,
)
from rlvr_physics.tasks.games.game2048.instances import (
    initial_2048_state,
    make_2048_instance,
)
from rlvr_physics.tasks.games.game2048.renderers import (
    render_2048_image,
    render_2048_text,
)
from rlvr_physics.tasks.games.game2048.rules import legal_2048_actions, move_board
from rlvr_physics.tasks.games.game2048.session import Game2048Session
from rlvr_physics.tasks.games.game2048.spec import game2048_task_spec
from rlvr_physics.tasks.games.game2048.types import (
    Board,
    Game2048State,
    MoveResult,
    SpawnTape,
)

__all__ = [
    "Board",
    "GAME2048_ACTIONS",
    "GAME2048_DOMAIN",
    "GAME2048_KIND",
    "Game2048Session",
    "Game2048State",
    "MoveResult",
    "SpawnTape",
    "game2048_task_spec",
    "initial_2048_state",
    "legal_2048_actions",
    "make_2048_instance",
    "move_board",
    "render_2048_image",
    "render_2048_text",
]
