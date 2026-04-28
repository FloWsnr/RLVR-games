"""Configured task helpers for instance and session construction."""

from collections.abc import Callable
from dataclasses import dataclass

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.core.specs import TaskSpec


TaskInstanceBuilder = Callable[[int], TaskInstance]
TaskSessionBuilder = Callable[[TaskInstance], TaskSession]


@dataclass(frozen=True)
class ConfiguredTask:
    """Configured task family with spec, instance source, and session builder.

    Parameters
    ----------
    spec:
        Public task specification for this configured task family.
    instance_builder:
        Callable that builds one immutable task instance from a deterministic
        seed. Record-backed tasks may close over source configuration in this
        callable.
    session_builder:
        Callable that creates one fresh scalar session for an immutable task
        instance. Renderer choices and other session configuration should be
        captured in this callable.

    Attributes
    ----------
    spec:
        Public task specification for this configured task family.
    instance_builder:
        Callable used to build immutable task instances from deterministic
        seeds.
    session_builder:
        Callable used to create one fresh scalar session for an immutable task
        instance.
    """

    spec: TaskSpec
    instance_builder: TaskInstanceBuilder
    session_builder: TaskSessionBuilder

    def build_instance(self, seed: int) -> TaskInstance:
        """Build one immutable instance for this configured task.

        Parameters
        ----------
        seed:
            Deterministic seed passed to the configured instance builder.

        Returns
        -------
        TaskInstance
            Immutable task instance produced by ``instance_builder``.

        Raises
        ------
        ValueError
            Raised when the produced instance does not match the configured
            task kind or domain.
        """

        instance = self.instance_builder(seed)
        self._validate_instance(instance)
        return instance

    def create_session(self, instance: TaskInstance) -> TaskSession:
        """Create one fresh scalar session for ``instance``.

        Parameters
        ----------
        instance:
            Immutable task instance used to initialize the session.

        Returns
        -------
        TaskSession
            Fresh session returned by ``session_builder``.

        Raises
        ------
        ValueError
            Raised when ``instance`` does not match the configured task kind or
            domain.
        """

        self._validate_instance(instance)
        return self.session_builder(instance)

    def _validate_instance(self, instance: TaskInstance) -> None:
        """Validate that an instance belongs to this configured task.

        Parameters
        ----------
        instance:
            Immutable task instance to validate.

        Raises
        ------
        ValueError
            Raised when the instance kind or domain differs from ``spec``.
        """

        if instance.kind != self.spec.kind:
            raise ValueError(
                f"instance kind {instance.kind!r} does not match "
                f"configured task kind {self.spec.kind!r}"
            )
        if instance.domain != self.spec.domain:
            raise ValueError(
                f"instance domain {instance.domain!r} does not match "
                f"configured task domain {self.spec.domain!r}"
            )
