"""Task spec construction for seeded 2048."""

from rlvr_physics.core.instances import TaskLimits
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.tasks.games.game2048.constants import (
    GAME2048_ACTIONS,
    GAME2048_DOMAIN,
    GAME2048_KIND,
)


def game2048_task_spec(seed: int, max_turns: int, target_tile: int) -> TaskSpec:
    """Return the Python task spec for seeded 2048.

    Parameters
    ----------
    seed:
        Seed used to generate the spawn tape.
    max_turns:
        Maximum valid moves before truncation.
    target_tile:
        Tile value that ends the task successfully.
    """

    return TaskSpec(
        kind=GAME2048_KIND,
        domain=GAME2048_DOMAIN,
        source=SourceSpec(
            source_type="procedural.spawn_tape",
            seed=seed,
            parameters={"max_turns": max_turns, "target_tile": target_tile},
        ),
        renderers=(
            RendererSpec(renderer_type="text"),
            RendererSpec(renderer_type="image"),
        ),
        verifier=VerifierSpec(
            verifier_type="deterministic_2048_rules",
            parameters={"actions": GAME2048_ACTIONS},
        ),
        reward=RewardSpec(
            reward_type="score_delta", parameters={"invalid_action": -1.0}
        ),
        limits=TaskLimits(max_turns=max_turns, action_budget=max_turns),
        metadata={"exports": {"environment": {"action_space": GAME2048_ACTIONS}}},
    )
