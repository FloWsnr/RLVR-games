"""Tests for task factory helpers."""

from rlvr_physics.core.factory import ConfiguredTaskFactory
from rlvr_physics.core.instances import TaskInstance, TaskLimits
from rlvr_physics.core.rendering import text_observation
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskSession,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
)
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.core.trajectory import TaskTrajectory


class FactoryTestSession:
    """Minimal scalar session used by factory tests."""

    def __init__(self, instance: TaskInstance) -> None:
        self._instance = instance
        self._trajectory = TaskTrajectory(
            task_id=instance.task_id,
            session_id="factory-test-session",
        )
        self._turn: TaskTurn | None = None

    def reset(self, seed: int) -> TaskResetResult:
        """Start the minimal test session."""

        _ = seed
        self._turn = TaskTurn(
            turn_index=0,
            observation=text_observation("text", "factory prompt"),
            submission_modes=("final_text",),
            action_schema={},
            public_limits=self._instance.limits.as_public_dict(),
            public_info={"task_id": self._instance.task_id},
        )
        return TaskResetResult(
            session_id=self._trajectory.session_id,
            turn=self._turn,
            trajectory=self._trajectory,
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current turn."""

        return self._turn

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Reject submissions because this fixture only tests factory reset."""

        _ = submission
        raise NotImplementedError("factory test session does not score submissions")

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the fixture trajectory."""

        return self._trajectory


def test_configured_task_factory_creates_scalar_sessions() -> None:
    instance = TaskInstance(
        task_id="factory-test",
        kind="tests.factory.v1",
        domain="tests",
        seed=17,
        public_payload={},
        privileged_payload={},
        limits=TaskLimits(max_turns=1),
    )
    spec = TaskSpec(
        kind=instance.kind,
        domain=instance.domain,
        source=SourceSpec(source_type="tests.factory", seed=17),
        renderers=(RendererSpec(renderer_type="text"),),
        verifier=VerifierSpec(verifier_type="fixture"),
        reward=RewardSpec(reward_type="fixture", parameters={}),
        limits=instance.limits,
    )
    factory = ConfiguredTaskFactory(
        spec=spec,
        session_builder=_factory_test_session,
    )

    session = factory.create_session(instance)
    reset = session.reset(seed=3)

    assert factory.spec.kind == instance.kind
    assert reset.turn.public_info["task_id"] == instance.task_id


def _factory_test_session(instance: TaskInstance) -> TaskSession:
    """Create a minimal session for factory tests."""

    return FactoryTestSession(instance)
