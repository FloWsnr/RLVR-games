"""Tests for task factory helpers."""

from rlvr_physics.core.factory import ConfiguredTaskFactory
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.tasks.games.countdown import (
    CountdownSession,
    countdown_task_spec,
    make_countdown_instance,
)


def _countdown_text_session(instance: TaskInstance) -> TaskSession:
    return CountdownSession(instance, "text")


def test_configured_task_factory_creates_scalar_sessions() -> None:
    factory = ConfiguredTaskFactory(
        spec=countdown_task_spec(seed=17, size=1),
        session_builder=_countdown_text_session,
    )
    instance = make_countdown_instance(seed=17, source_index=0)

    session = factory.create_session(instance)
    reset = session.reset(seed=3)

    assert factory.spec.kind == instance.kind
    assert reset.turn.public_info["task_id"] == instance.task_id
