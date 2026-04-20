"""Task spec construction for chess tactics."""

from rlvr_physics.core.instances import TaskLimits
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.tasks.games.chess_tactics.constants import CHESS_DOMAIN, CHESS_KIND


def chess_tactics_task_spec(seed: int) -> TaskSpec:
    """Return the Python task spec for chess tactics.

    Parameters
    ----------
    seed:
        Source seed for deterministic puzzle selection.
    """

    return TaskSpec(
        kind=CHESS_KIND,
        domain=CHESS_DOMAIN,
        source=SourceSpec(
            source_type="builtin.chess_tactics", seed=seed, parameters={"mate_depth": 1}
        ),
        renderers=(
            RendererSpec(renderer_type="text"),
            RendererSpec(renderer_type="image"),
        ),
        verifier=VerifierSpec(
            verifier_type="python_chess_mate_in_one",
            parameters={"accepted_notation": ("SAN", "UCI")},
        ),
        reward=RewardSpec(
            reward_type="mate_in_one",
            parameters={
                "correct": 1.0,
                "legal_non_solution": 0.2,
                "illegal_or_parse_failure": 0.0,
            },
        ),
        limits=TaskLimits(max_turns=1, token_budget=64),
        metadata={"exports": {"dataset": {"ability": "chess_tactics"}}},
    )
