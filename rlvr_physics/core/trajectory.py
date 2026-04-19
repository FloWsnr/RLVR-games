"""Append-only trajectory records."""

from dataclasses import dataclass, field
from typing import Mapping

from rlvr_physics.core.instances import freeze_mapping


@dataclass(frozen=True)
class TrajectoryEvent:
    """One verified event in a scalar task rollout.

    Parameters
    ----------
    event_type:
        Event category, such as ``reset``, ``observation``, or ``reward``.
    turn_index:
        Turn associated with the event.
    public:
        Trainer-safe payload.
    debug:
        Privileged local payload for evaluation and debugging.
    """

    event_type: str
    turn_index: int
    public: Mapping[str, object]
    debug: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze payloads after construction."""

        object.__setattr__(self, "public", freeze_mapping(self.public))
        object.__setattr__(self, "debug", freeze_mapping(self.debug))


@dataclass
class TaskTrajectory:
    """Append-only trajectory for one task session.

    Parameters
    ----------
    task_id:
        Stable task identifier.
    session_id:
        Stable session identifier.
    """

    task_id: str
    session_id: str
    _events: list[TrajectoryEvent] = field(default_factory=list, repr=False)

    @property
    def events(self) -> tuple[TrajectoryEvent, ...]:
        """Return a read-only tuple of recorded events."""

        return tuple(self._events)

    def append(
        self,
        event_type: str,
        turn_index: int,
        public: Mapping[str, object],
        debug: Mapping[str, object],
    ) -> TrajectoryEvent:
        """Append and return a trajectory event."""

        event = TrajectoryEvent(
            event_type=event_type, turn_index=turn_index, public=public, debug=debug
        )
        self._events.append(event)
        return event

    def snapshot(self) -> tuple[TrajectoryEvent, ...]:
        """Return an immutable view of recorded events."""

        return self.events
