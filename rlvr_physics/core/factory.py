"""Task factory protocols and helper implementations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.core.specs import TaskSpec


TaskSessionBuilder = Callable[[TaskInstance], TaskSession]


class TaskFactory(Protocol):
    """Factory for configured scalar task sessions."""

    @property
    def spec(self) -> TaskSpec:
        """Return the public task specification."""
        ...

    def create_session(self, instance: TaskInstance) -> TaskSession:
        """Create one fresh scalar session for ``instance``."""
        ...


@dataclass(frozen=True)
class ConfiguredTaskFactory:
    """Task factory backed by a task spec and callable session builder.

    Parameters
    ----------
    spec:
        Public task specification for the configured task family.
    session_builder:
        Callable that creates one fresh scalar session for an immutable task
        instance. Renderer choices and other session configuration should be
        captured in this callable.
    """

    spec: TaskSpec
    session_builder: TaskSessionBuilder

    def create_session(self, instance: TaskInstance) -> TaskSession:
        """Create one fresh scalar session for ``instance``."""

        return self.session_builder(instance)
