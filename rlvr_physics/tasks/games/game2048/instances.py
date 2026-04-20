"""Instance construction for seeded 2048."""

from rlvr_physics.core.instances import (
    TaskInstance,
    TaskLimits,
    require_int,
    stable_hash,
)
from rlvr_physics.tasks.games.game2048.constants import (
    GAME2048_DOMAIN,
    GAME2048_KIND,
)
from rlvr_physics.tasks.games.game2048.rules import (
    apply_spawn,
    make_spawn_tape,
    require_board,
)
from rlvr_physics.tasks.games.game2048.types import Board, Game2048State


def make_2048_instance(seed: int, max_turns: int, target_tile: int) -> TaskInstance:
    """Create one deterministic 2048 task instance."""

    size = 4
    spawn_tape = make_spawn_tape(seed, max_turns + 16, size)
    board: Board = tuple(tuple(0 for _ in range(size)) for _ in range(size))
    board, first_used = apply_spawn(board, spawn_tape[0])
    board, second_used = apply_spawn(board, spawn_tape[1])
    initial_spawn_count = int(first_used) + int(second_used)
    task_id = (
        "2048-"
        + stable_hash(
            {
                "seed": seed,
                "max_turns": max_turns,
                "target_tile": target_tile,
                "spawn_tape": spawn_tape,
                "initial_board": board,
            }
        )[:16]
    )
    return TaskInstance(
        task_id=task_id,
        kind=GAME2048_KIND,
        domain=GAME2048_DOMAIN,
        seed=seed,
        public_payload={
            "size": size,
            "initial_board": board,
            "target_tile": target_tile,
        },
        privileged_payload={
            "spawn_tape": spawn_tape,
            "initial_spawn_count": initial_spawn_count,
        },
        limits=TaskLimits(max_turns=max_turns, action_budget=max_turns),
        metadata={"source": "procedural.spawn_tape"},
    )


def initial_2048_state(instance: TaskInstance) -> Game2048State:
    """Build the canonical initial state for a 2048 instance."""

    board = require_board(instance.public_payload["initial_board"], "initial_board")
    max_tile = max(max(row) for row in board)
    spawn_index = require_int(
        instance.privileged_payload["initial_spawn_count"], "initial_spawn_count"
    )
    return Game2048State(
        board=board, score=0, turns=0, max_tile=max_tile, spawn_index=spawn_index
    )
