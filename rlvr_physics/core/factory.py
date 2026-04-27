"""Task factory protocols and helper implementations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.core.specs import TaskSpec


TaskSessionBuilder = Callable[[TaskInstance], TaskSession]


class TaskFactory(Protocol):
    """Factory for configured scalar task sessions.

    Attributes
    ----------
    spec:
        Public task specification for sessions created by the factory.
    """

    @property
    def spec(self) -> TaskSpec:
        """Return the public task specification.

        Returns
        -------
        TaskSpec
            Specification describing the configured task family.
        """
        ...

    def create_session(self, instance: TaskInstance) -> TaskSession:
        """Create one fresh scalar session for ``instance``.

        Parameters
        ----------
        instance:
            Immutable task instance used to initialize the session.

        Returns
        -------
        TaskSession
            Fresh session for the supplied task instance.
        """
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

    Attributes
    ----------
    spec:
        Public task specification for the configured task family.
    session_builder:
        Callable used to create one fresh scalar session for an immutable task
        instance.
    """

    spec: TaskSpec
    session_builder: TaskSessionBuilder

    def create_session(self, instance: TaskInstance) -> TaskSession:
        """Create one fresh scalar session for ``instance``.

        Parameters
        ----------
        instance:
            Immutable task instance passed to the configured session builder.

        Returns
        -------
        TaskSession
            Fresh session returned by ``session_builder``.
        """

        return self.session_builder(instance)
