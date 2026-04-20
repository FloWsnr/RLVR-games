"""Task spec construction for Countdown."""

from rlvr_physics.core.instances import TaskLimits
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.tasks.games.countdown.constants import (
    COUNTDOWN_DOMAIN,
    COUNTDOWN_KIND,
)


def countdown_task_spec(seed: int, size: int) -> TaskSpec:
    """Return the Python task spec for Countdown instances.

    Parameters
    ----------
    seed:
        Procedural Reasoning Gym seed.
    size:
        Virtual source dataset size.
    """

    return TaskSpec(
        kind=COUNTDOWN_KIND,
        domain=COUNTDOWN_DOMAIN,
        source=SourceSpec(
            source_type="reasoning_gym.countdown", seed=seed, parameters={"size": size}
        ),
        renderers=(
            RendererSpec(renderer_type="text"),
            RendererSpec(renderer_type="image"),
        ),
        verifier=VerifierSpec(
            verifier_type="arithmetic_expression",
            parameters={"uses_all_numbers_once": True, "exact_target": True},
        ),
        reward=RewardSpec(
            reward_type="graded_countdown",
            parameters={
                "correct": 1.0,
                "wrong_numbers": 0.05,
                "wrong_value": 0.05,
                "invalid": 0.01,
            },
        ),
        limits=TaskLimits(max_turns=1, token_budget=256),
        metadata={"exports": {"dataset": {"ability": "countdown"}}},
    )
