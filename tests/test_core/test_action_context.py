"""Action context tests."""

from dataclasses import dataclass, field

import pytest

from rlvr_games.core import (
    AgentVisibleEvent,
    AppliedResetEvent,
    EpisodeConfig,
    PublicResetEvent,
    ProjectedActionContext,
)
from rlvr_games.core.rollout import build_action_context

from tests.test_core.support import (
    CounterBackend,
    CounterState,
    make_counter_env,
)


@dataclass(slots=True)
class CounterOpeningResetEventPolicy:
    """Record one reset-time opening event for action-context tests."""

    _emitted: bool = field(init=False, default=False, repr=False)

    def reset(self, *, initial_state: CounterState) -> None:
        """Start a fresh reset-time event sequence."""
        del initial_state
        self._emitted = False

    def apply_next_event(
        self,
        *,
        state: CounterState,
    ) -> AppliedResetEvent[CounterState] | None:
        """Emit one reset-time event and then stop."""
        if self._emitted:
            return None
        self._emitted = True
        next_state = CounterState(value=state.value + 1)
        return AppliedResetEvent(
            source="dealer",
            label="opening bonus",
            next_state=next_state,
            info={"value": next_state.value},
        )


class CounterOpeningProjector:
    """Project reset events into structured opening context."""

    def project_action_context(
        self,
        *,
        state: CounterState,
        reset_events: tuple[PublicResetEvent, ...],
    ) -> ProjectedActionContext:
        """Project action context from reset-event trajectory history."""
        del state
        return ProjectedActionContext(
            opening_events=tuple(
                AgentVisibleEvent(
                    kind="opening_event",
                    source=event.source,
                    text=event.label,
                    metadata=event.info,
                )
                for event in reset_events
            ),
        )


class MutatingCounterProjector:
    """Mutate projection inputs to verify the env passes snapshots."""

    def project_action_context(
        self,
        *,
        state: CounterState,
        reset_events: tuple[PublicResetEvent, ...],
    ) -> ProjectedActionContext:
        """Mutate the provided inputs before projecting one opening event."""
        state.value = 123
        reset_events[0].info["value"] = 999
        return ProjectedActionContext(
            opening_events=(
                AgentVisibleEvent(
                    kind="opening_event",
                    source="dealer",
                    text="mutated copy",
                    metadata={
                        "state_value": state.value,
                        "event_value": reset_events[0].info["value"],
                    },
                ),
            ),
        )


class LeakyCounterProjector:
    """Return the wrong event type to verify runtime validation."""

    def project_action_context(
        self,
        *,
        state: CounterState,
        reset_events: tuple[PublicResetEvent, ...],
    ) -> ProjectedActionContext:
        """Return raw reset events instead of agent-visible events."""
        del state
        return ProjectedActionContext(
            opening_events=reset_events,  # pyright: ignore[reportArgumentType]
        )


class MetadataSmugglingCounterProjector:
    """Embed a non-JSON-like object in metadata to verify validation."""

    def project_action_context(
        self,
        *,
        state: CounterState,
        reset_events: tuple[PublicResetEvent, ...],
    ) -> ProjectedActionContext:
        """Return one event with an invalid metadata payload."""
        del state
        return ProjectedActionContext(
            opening_events=(
                AgentVisibleEvent(
                    kind="opening_event",
                    source="dealer",
                    metadata={"leak": reset_events[0]},  # pyright: ignore[reportArgumentType]
                ),
            ),
        )


class WrongReturnCounterProjector:
    """Return the wrong top-level type to verify runtime validation."""

    def project_action_context(
        self,
        *,
        state: CounterState,
        reset_events: tuple[PublicResetEvent, ...],
    ) -> ProjectedActionContext:
        """Return a full ActionContext instead of a projection."""
        del state
        del reset_events
        return AgentVisibleEvent(  # pyright: ignore[reportReturnType]
            kind="opening_event",
            source="dealer",
        )


def test_build_action_context_uses_current_turn_index() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
    )
    env.reset(seed=7)

    initial_context = build_action_context(env=env)
    env.step("1")
    next_context = build_action_context(env=env)

    assert initial_context.turn_index == 0
    assert next_context.turn_index == 1
    assert initial_context.opening_events == ()
    assert next_context.opening_events == ()


def test_build_action_context_can_include_projected_opening_events() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
        agent_context_projector=CounterOpeningProjector(),
        reset_event_policy=CounterOpeningResetEventPolicy(),
    )
    env.reset(seed=11)

    context = build_action_context(env=env)
    env.step("1")
    next_context = build_action_context(env=env)

    assert context.turn_index == 0
    assert next_context.turn_index == 1
    assert context.opening_events == (
        AgentVisibleEvent(
            kind="opening_event",
            source="dealer",
            text="opening bonus",
            metadata={"value": 1},
        ),
    )
    assert next_context.opening_events == context.opening_events


def test_projected_opening_events_do_not_alias_trajectory_reset_events() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
        agent_context_projector=CounterOpeningProjector(),
        reset_event_policy=CounterOpeningResetEventPolicy(),
    )
    env.reset(seed=11)

    context = build_action_context(env=env)
    context.opening_events[0].metadata["value"] = 999

    assert env.trajectory.reset_events[0].info["value"] == 1


def test_projector_receives_snapshots_of_state_and_trajectory() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
        agent_context_projector=MutatingCounterProjector(),
        reset_event_policy=CounterOpeningResetEventPolicy(),
    )
    env.reset(seed=11)

    context = build_action_context(env=env)

    assert context.opening_events == (
        AgentVisibleEvent(
            kind="opening_event",
            source="dealer",
            text="mutated copy",
            metadata={"state_value": 123, "event_value": 999},
        ),
    )
    assert env.state.value == 1
    assert env.trajectory.reset_events[0].info["value"] == 1


def test_build_action_context_rejects_non_agent_visible_events() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
        agent_context_projector=LeakyCounterProjector(),
        reset_event_policy=CounterOpeningResetEventPolicy(),
    )
    env.reset(seed=11)

    with pytest.raises(TypeError, match="opening_events must contain only"):
        build_action_context(env=env)


def test_build_action_context_rejects_non_json_like_metadata() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
        agent_context_projector=MetadataSmugglingCounterProjector(),
        reset_event_policy=CounterOpeningResetEventPolicy(),
    )
    env.reset(seed=11)

    with pytest.raises(
        TypeError,
        match="Agent-visible metadata must contain only JSON-like values.",
    ):
        build_action_context(env=env)


def test_agent_visible_event_rejects_cyclic_metadata() -> None:
    metadata: dict[str, object] = {}
    metadata["self"] = metadata

    with pytest.raises(
        TypeError,
        match="Agent-visible metadata must not contain cycles.",
    ):
        AgentVisibleEvent(
            kind="opening_event",
            source="dealer",
            metadata=metadata,
        )


def test_build_action_context_rejects_invalid_projector_return_type() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
        agent_context_projector=WrongReturnCounterProjector(),
        reset_event_policy=CounterOpeningResetEventPolicy(),
    )
    env.reset(seed=11)

    with pytest.raises(
        TypeError,
        match="must return ProjectedActionContext",
    ):
        build_action_context(env=env)


@pytest.mark.parametrize(
    ("event", "error_message"),
    [
        (
            {"kind": 123, "source": "dealer"},
            "AgentVisibleEvent kind must be a string.",
        ),
        (
            {"kind": "opening_event", "source": object()},
            "AgentVisibleEvent source must be a string.",
        ),
        (
            {"kind": "opening_event", "source": "dealer", "text": 123},
            "AgentVisibleEvent text must be a string or None.",
        ),
    ],
)
def test_agent_visible_event_rejects_invalid_string_fields(
    event: dict[str, object],
    error_message: str,
) -> None:
    with pytest.raises(TypeError, match=error_message):
        AgentVisibleEvent(**event)  # pyright: ignore[reportArgumentType]
