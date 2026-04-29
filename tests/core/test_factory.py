"""Tests for configured task helpers in core.factory."""

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from tests.conftest import EXAMPLE_TASK_KIND, ExampleTaskFixture


def test_configured_task_builds_instances_and_creates_scalar_sessions(
    example_task_fixture: ExampleTaskFixture,
) -> None:
    task = example_task_fixture.task
    instance = example_task_fixture.instance
    reset = example_task_fixture.reset

    assert task.spec.kind == instance.kind
    assert instance.task_id == "configured-task-test-17"
    assert reset.turn.public_info["task_id"] == instance.task_id
    assert reset.turn.public_limits == instance.public_limits()


def test_configured_task_rejects_instances_from_other_tasks(
    mismatched_configured_task: ConfiguredTask,
) -> None:
    with pytest.raises(ValueError, match="instance kind"):
        mismatched_configured_task.build_instance(seed=17)


def test_configured_task_rejects_external_domain_mismatch(
    example_configured_task: ConfiguredTask,
    example_task_fixture: ExampleTaskFixture,
) -> None:
    mismatched_instance = TaskInstance(
        task_id="configured-task-test-domain",
        kind=EXAMPLE_TASK_KIND,
        domain="other",
        seed=17,
        public_payload={},
        privileged_payload={},
        budget_limits={"turns": 1},
    )

    with pytest.raises(ValueError, match="instance domain"):
        example_configured_task.create_session(mismatched_instance)
