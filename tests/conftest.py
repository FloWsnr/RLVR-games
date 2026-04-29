"""Shared pytest fixtures for package-level tests."""

from dataclasses import dataclass

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.rendering import text_observation
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskSession,
    TaskStepResult,
    TaskTurn,
)
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.core.submissions import TaskSubmission

EXAMPLE_TASK_KIND = "tests.configured_task.v1"
EXAMPLE_TASK_DOMAIN = "tests"


@dataclass(frozen=True)
class ExampleTaskFixture:
    """Reusable configured task setup for core tests."""

    task: ConfiguredTask
    instance: TaskInstance
    session: TaskSession
    reset: TaskResetResult


class ExampleScalarSession:
    """Minimal scalar session used by configured task tests."""

    def __init__(self, instance: TaskInstance) -> None:
        self._instance = instance
        self._session_id = "configured-task-test-session"
        self._turn: TaskTurn | None = None

    def reset(self, seed: int) -> TaskResetResult:
        _ = seed
        self._turn = TaskTurn(
            turn_index=0,
            observation=text_observation("text", "configured task prompt"),
            submission_modes=("final_text",),
            submission_format={},
            action_schema={},
            invalid_submission_policies={},
            public_limits=self._instance.public_limits(),
            public_info={"task_id": self._instance.task_id},
        )
        return TaskResetResult(
            session_id=self._session_id,
            turn=self._turn,
            public_info={"task_id": self._instance.task_id},
            debug_info={},
        )

    @property
    def turn(self) -> TaskTurn | None:
        return self._turn

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        _ = submission
        raise NotImplementedError("example scalar session does not score")


@pytest.fixture
def example_task_spec() -> TaskSpec:
    """Return a public task spec for core tests."""

    return TaskSpec(
        kind=EXAMPLE_TASK_KIND,
        domain=EXAMPLE_TASK_DOMAIN,
        source=SourceSpec(source_type="tests.configured_task", seed=17),
        renderers=(RendererSpec(renderer_type="text"),),
        verifier=VerifierSpec(verifier_type="fixture"),
        reward=RewardSpec(reward_type="fixture", parameters={}),
        budget_limits={"turns": 1},
        metadata={"exports": {"dataset": {"ability": "example"}}},
    )


@pytest.fixture
def example_task_instance() -> TaskInstance:
    """Return an immutable task instance for core tests."""

    return build_example_task_instance(seed=17)


@pytest.fixture
def example_turn(example_task_instance: TaskInstance) -> TaskTurn:
    """Return a model-facing turn for core session tests."""

    return TaskTurn(
        turn_index=0,
        observation=text_observation("text", "prompt"),
        submission_modes=("final_text",),
        submission_format={},
        action_schema={},
        invalid_submission_policies={},
        public_limits=example_task_instance.public_limits(),
        public_info={"task_id": example_task_instance.task_id},
    )


@pytest.fixture
def example_configured_task(example_task_spec: TaskSpec) -> ConfiguredTask:
    """Return a configured task backed by the example fixture session."""

    return ConfiguredTask(
        spec=example_task_spec,
        instance_builder=build_example_task_instance,
        session_builder=ExampleScalarSession,
    )


@pytest.fixture
def mismatched_configured_task(example_task_spec: TaskSpec) -> ConfiguredTask:
    """Return a configured task whose instance builder emits the wrong kind."""

    return ConfiguredTask(
        spec=example_task_spec,
        instance_builder=build_other_kind_task_instance,
        session_builder=ExampleScalarSession,
    )


@pytest.fixture
def example_task_fixture(
    example_configured_task: ConfiguredTask,
) -> ExampleTaskFixture:
    """Return a configured task with a built instance and reset session."""

    instance = example_configured_task.build_instance(seed=17)
    session = example_configured_task.create_session(instance)
    reset = session.reset(seed=3)
    return ExampleTaskFixture(
        task=example_configured_task,
        instance=instance,
        session=session,
        reset=reset,
    )


def build_example_task_instance(seed: int) -> TaskInstance:
    """Build an immutable example task instance."""

    return TaskInstance(
        task_id=f"configured-task-test-{seed}",
        kind=EXAMPLE_TASK_KIND,
        domain=EXAMPLE_TASK_DOMAIN,
        seed=seed,
        public_payload={"prompt": "configured task prompt"},
        privileged_payload={"answer": 42},
        budget_limits={"turns": 1},
    )


def build_other_kind_task_instance(seed: int) -> TaskInstance:
    """Build an instance whose kind does not match the example spec."""

    return TaskInstance(
        task_id=f"configured-task-test-{seed}",
        kind="tests.other_task.v1",
        domain=EXAMPLE_TASK_DOMAIN,
        seed=seed,
        public_payload={},
        privileged_payload={},
        budget_limits={"turns": 1},
    )
