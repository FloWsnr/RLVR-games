"""Task spec construction for physics discovery."""

from rlvr_physics.core.instances import TaskLimits
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.tasks.physics.discovery.constants import (
    PHYSICS_DISCOVERY_KIND,
    PHYSICS_DISCOVERY_SOURCE,
    PHYSICS_DOMAIN,
)
from rlvr_physics.tasks.physics.discovery.records import physics_discovery_records
from rlvr_physics.tasks.physics.discovery.utils import (
    validate_positive_quota,
    validate_prior_mode,
)


def physics_discovery_task_spec(
    seed: int,
    sample_quota: int,
    hypothesis_quota: int,
    prior_mode: str,
) -> TaskSpec:
    """Return the task spec for interactive physics discovery.

    Parameters
    ----------
    seed:
        Seed used for deterministic hidden verification points.
    sample_quota:
        Maximum accepted experiments before hypotheses must rely on history.
    hypothesis_quota:
        Maximum accepted hypothesis tests.
    prior_mode:
        Information exposure mode.
    """

    validate_prior_mode(prior_mode)
    validate_positive_quota(sample_quota, "sample_quota")
    validate_positive_quota(hypothesis_quota, "hypothesis_quota")
    return TaskSpec(
        kind=PHYSICS_DISCOVERY_KIND,
        domain=PHYSICS_DOMAIN,
        source=SourceSpec(
            source_type=PHYSICS_DISCOVERY_SOURCE,
            seed=seed,
            parameters={
                "sample_quota": sample_quota,
                "hypothesis_quota": hypothesis_quota,
                "prior_mode": prior_mode,
                "records": len(physics_discovery_records()),
            },
        ),
        renderers=(RendererSpec(renderer_type="text"),),
        verifier=VerifierSpec(
            verifier_type="hidden_numeric_equivalence",
            parameters={"hidden_points": 24, "relative_tolerance": 1e-5},
        ),
        reward=RewardSpec(
            reward_type="hypothesis_fit_with_experiment_cost",
            parameters={
                "correct": 1.0,
                "experiment_cost": -0.01,
                "invalid": -0.05,
            },
        ),
        limits=TaskLimits(
            max_turns=sample_quota + hypothesis_quota,
            action_budget=sample_quota + hypothesis_quota,
        ),
        metadata={
            "exports": {
                "environment": {"actions": ("run_experiment", "submit_hypothesis")}
            },
            "source": "PhysGym full records",
        },
    )
