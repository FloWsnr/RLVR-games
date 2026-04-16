"""Agent-facing action context types."""

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

StateT = TypeVar("StateT", contravariant=True)


def _snapshot_agent_metadata(
    value: Any,
    *,
    _active_container_ids: set[int] | None = None,
) -> Any:
    """Return a detached JSON-like snapshot of agent-visible metadata.

    Parameters
    ----------
    value : Any
        Metadata value to validate and snapshot.

    Returns
    -------
    Any
        Detached metadata containing only JSON-like scalar, list, tuple, and
        dict values with string keys.

    Raises
    ------
    TypeError
        If `value` contains a non-JSON-like object.
    """
    if _active_container_ids is None:
        _active_container_ids = set()

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        container_id = id(value)
        if container_id in _active_container_ids:
            raise TypeError("Agent-visible metadata must not contain cycles.")
        _active_container_ids.add(container_id)
        try:
            return [
                _snapshot_agent_metadata(
                    item,
                    _active_container_ids=_active_container_ids,
                )
                for item in value
            ]
        finally:
            _active_container_ids.remove(container_id)
    if isinstance(value, tuple):
        container_id = id(value)
        if container_id in _active_container_ids:
            raise TypeError("Agent-visible metadata must not contain cycles.")
        _active_container_ids.add(container_id)
        try:
            return tuple(
                _snapshot_agent_metadata(
                    item,
                    _active_container_ids=_active_container_ids,
                )
                for item in value
            )
        finally:
            _active_container_ids.remove(container_id)
    if isinstance(value, dict):
        container_id = id(value)
        if container_id in _active_container_ids:
            raise TypeError("Agent-visible metadata must not contain cycles.")
        _active_container_ids.add(container_id)
        snapshot: dict[str, Any] = {}
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("Agent-visible metadata keys must be strings.")
                snapshot[key] = _snapshot_agent_metadata(
                    item,
                    _active_container_ids=_active_container_ids,
                )
            return snapshot
        finally:
            _active_container_ids.remove(container_id)
    raise TypeError("Agent-visible metadata must contain only JSON-like values.")


def _normalize_opening_events(
    opening_events: tuple["AgentVisibleEvent", ...],
    *,
    snapshot: bool,
) -> tuple["AgentVisibleEvent", ...]:
    """Validate and normalize projected opening events.

    Parameters
    ----------
    opening_events : tuple[AgentVisibleEvent, ...]
        Projected opening events to validate.
    snapshot : bool
        Whether to deep-copy validated events before returning them.

    Returns
    -------
    tuple[AgentVisibleEvent, ...]
        Normalized tuple containing only `AgentVisibleEvent` instances.

    Raises
    ------
    TypeError
        If any projected event is not an `AgentVisibleEvent`.
    """
    normalized_events: list[AgentVisibleEvent] = []
    for event in opening_events:
        if not isinstance(event, AgentVisibleEvent):
            raise TypeError(
                "opening_events must contain only AgentVisibleEvent instances."
            )
        normalized_events.append(
            AgentVisibleEvent(
                kind=event.kind,
                source=event.source,
                text=event.text,
                metadata=event.metadata,
            )
            if snapshot
            else event
        )
    return tuple(normalized_events)


@dataclass(slots=True)
class AgentVisibleEvent:
    """Structured event projected into agent-facing context.

    Attributes
    ----------
    kind : str
        Structured event kind such as ``"opening_deal"`` or
        ``"opening_roll"``.
    source : str
        Structured source label describing who produced the event.
    text : str | None
        Optional human-readable event summary suitable for later prompt
        adapters.
    metadata : dict[str, Any]
        Structured public-safe metadata exposed to the agent.
    """

    kind: str
    source: str
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the projected event is coherent."""
        if not isinstance(self.kind, str):
            raise TypeError("AgentVisibleEvent kind must be a string.")
        if not self.kind:
            raise ValueError("AgentVisibleEvent kind must be non-empty.")
        if not isinstance(self.source, str):
            raise TypeError("AgentVisibleEvent source must be a string.")
        if not self.source:
            raise ValueError("AgentVisibleEvent source must be non-empty.")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("AgentVisibleEvent text must be a string or None.")
        if not isinstance(self.metadata, dict):
            raise TypeError("AgentVisibleEvent metadata must be a dict.")
        self.metadata = _snapshot_agent_metadata(self.metadata)


@dataclass(slots=True)
class PublicResetEvent:
    """Public reset-time history exposed to an agent-context projector.

    Attributes
    ----------
    source : str
        Structured label describing who produced the event.
    label : str
        Structured serialized label describing the event.
    info : dict[str, Any]
        Detached public-safe metadata describing the event.
    """

    source: str
    label: str
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the reset event can be exposed to a projector."""
        if not isinstance(self.source, str):
            raise TypeError("PublicResetEvent source must be a string.")
        if not self.source:
            raise ValueError("PublicResetEvent source must be non-empty.")
        if not isinstance(self.label, str):
            raise TypeError("PublicResetEvent label must be a string.")
        if not self.label:
            raise ValueError("PublicResetEvent label must be non-empty.")
        if not isinstance(self.info, dict):
            raise TypeError("PublicResetEvent info must be a dict.")
        self.info = _snapshot_agent_metadata(self.info)


@dataclass(slots=True)
class ProjectedActionContext:
    """Structured agent-visible context contributed by a projector.

    Attributes
    ----------
    opening_events : tuple[AgentVisibleEvent, ...]
        Optional reset-time events projected into the agent-visible action
        context.
    """

    opening_events: tuple[AgentVisibleEvent, ...] = ()

    def __post_init__(self) -> None:
        """Validate projected context fields contributed by a projector."""
        self.opening_events = _normalize_opening_events(
            self.opening_events,
            snapshot=False,
        )


@dataclass(slots=True)
class ActionContext:
    """Agent-facing context for choosing the next action.

    Attributes
    ----------
    turn_index : int
        Zero-based turn index for the next action to be taken.
    opening_events : tuple[AgentVisibleEvent, ...]
        Optional reset-time events projected into the agent-visible action
        context.
    """

    turn_index: int
    opening_events: tuple[AgentVisibleEvent, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the action context is coherent."""
        if self.turn_index < 0:
            raise ValueError("ActionContext turn_index must be non-negative.")
        self.opening_events = _normalize_opening_events(
            self.opening_events,
            snapshot=True,
        )


class AgentContextProjector(Protocol[StateT]):
    """Protocol for projecting structured agent-facing context.

    Projectors expose selected public-safe reset history or canonical-state
    details to the agent without modifying the observation or reset info.
    """

    def project_action_context(
        self,
        *,
        state: StateT,
        reset_events: tuple[PublicResetEvent, ...],
    ) -> ProjectedActionContext:
        """Project agent-facing context from public-safe env state.

        Parameters
        ----------
        state : StateT
            Current canonical state after reset-time resolution and any prior
            accepted steps.
        reset_events : tuple[PublicResetEvent, ...]
            Detached public reset-time history recorded before the first agent
            action.

        Returns
        -------
        ProjectedActionContext
            Structured context projected for the agent. The environment still
            owns turn numbering and any other generic action-context fields.
        """
        ...
