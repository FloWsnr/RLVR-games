"""Tests for configured task helpers in core.factory."""

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
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


class ConfiguredTaskTestSession:
    """Minimal scalar session used by configured task tests.

    Parameters
    ----------
    instance:
        Immutable test task instance whose identifiers and limits are exposed
        through reset results.
    """

    def __init__(self, instance: TaskInstance) -> None:
        self._instance = instance
        self._trajectory = TaskTrajectory(
            task_id=instance.task_id,
            session_id="configured-task-test-session",
        )
        self._turn: TaskTurn | None = None

    def reset(self, seed: int) -> TaskResetResult:
        """Start the minimal test session.

        Parameters
        ----------
        seed:
            Reset seed accepted to satisfy the session protocol. This fixture
            does not use it because the task instance is already fixed.

        Returns
        -------
        TaskResetResult
            Reset payload containing the first turn and fixture trajectory.
        """

        _ = seed
        self._turn = TaskTurn(
            turn_index=0,
            observation=text_observation("text", "configured task prompt"),
            submission_modes=("final_text",),
            action_schema={},
            public_limits=self._instance.public_limits(),
            public_info={"task_id": self._instance.task_id},
        )
        return TaskResetResult(
            session_id=self._trajectory.session_id,
            turn=self._turn,
            trajectory=self._trajectory,
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current turn.

        Returns
        -------
        TaskTurn | None
            Current test turn, or ``None`` before reset.
        """

        return self._turn

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Reject submissions because this fixture only tests session reset.

        Parameters
        ----------
        submission:
            Submission accepted to satisfy the session protocol.

        Raises
        ------
        NotImplementedError
            Always raised because the fixture does not score submissions.
        """

        _ = submission
        raise NotImplementedError("configured task test session does not score")

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the fixture trajectory.

        Returns
        -------
        TaskTrajectory
            Append-only trajectory attached to this fixture session.
        """

        return self._trajectory


def test_configured_task_builds_instances_and_creates_scalar_sessions() -> None:
    spec = TaskSpec(
        kind="tests.configured_task.v1",
        domain="tests",
        source=SourceSpec(source_type="tests.configured_task", seed=17),
        renderers=(RendererSpec(renderer_type="text"),),
        verifier=VerifierSpec(verifier_type="fixture"),
        reward=RewardSpec(reward_type="fixture", parameters={}),
        max_turns=1,
    )
    task = ConfiguredTask(
        spec=spec,
        instance_builder=_build_configured_task_test_instance,
        session_builder=_configured_task_test_session,
    )

    instance = task.build_instance(seed=17)
    session = task.create_session(instance)
    reset = session.reset(seed=3)

    assert task.spec.kind == instance.kind
    assert instance.task_id == "configured-task-test-17"
    assert reset.turn.public_info["task_id"] == instance.task_id


def test_configured_task_rejects_instances_from_other_tasks() -> None:
    spec = TaskSpec(
        kind="tests.configured_task.v1",
        domain="tests",
        source=SourceSpec(source_type="tests.configured_task", seed=17),
        renderers=(RendererSpec(renderer_type="text"),),
        verifier=VerifierSpec(verifier_type="fixture"),
        reward=RewardSpec(reward_type="fixture", parameters={}),
        max_turns=1,
    )
    task = ConfiguredTask(
        spec=spec,
        instance_builder=_build_mismatched_configured_task_test_instance,
        session_builder=_configured_task_test_session,
    )

    with pytest.raises(ValueError, match="instance kind"):
        task.build_instance(seed=17)


def _build_configured_task_test_instance(seed: int) -> TaskInstance:
    """Build an immutable test instance.

    Parameters
    ----------
    seed:
        Deterministic test instance seed.

    Returns
    -------
    TaskInstance
        Immutable test task instance.
    """

    return TaskInstance(
        task_id=f"configured-task-test-{seed}",
        kind="tests.configured_task.v1",
        domain="tests",
        seed=seed,
        public_payload={},
        privileged_payload={},
        max_turns=1,
    )


def _build_mismatched_configured_task_test_instance(seed: int) -> TaskInstance:
    """Build an instance whose kind does not match the configured spec.

    Parameters
    ----------
    seed:
        Deterministic test instance seed.

    Returns
    -------
    TaskInstance
        Immutable mismatched task instance.
    """

    return TaskInstance(
        task_id=f"configured-task-test-{seed}",
        kind="tests.other_task.v1",
        domain="tests",
        seed=seed,
        public_payload={},
        privileged_payload={},
        max_turns=1,
    )


def _configured_task_test_session(instance: TaskInstance) -> TaskSession:
    """Create a minimal session for configured task tests.

    Parameters
    ----------
    instance:
        Immutable test task instance for the session.

    Returns
    -------
    TaskSession
        Fresh scalar session backed by ``instance``.
    """

    return ConfiguredTaskTestSession(instance)
