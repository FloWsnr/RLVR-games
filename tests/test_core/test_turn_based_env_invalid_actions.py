"""Generic turn-based env invalid-action and limit tests."""

from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)

from tests.test_core.support import (
    ApplyRejectingCounterBackend,
    CounterBackend,
    make_counter_env,
)


def test_penalize_continue_records_rejected_attempt_without_state_transition() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-2.0,
            )
        ),
    )
    env.reset(seed=1)

    rejected = env.step("bad")
    accepted = env.step("1")

    assert rejected.accepted is False
    assert rejected.reward == -2.0
    assert rejected.info["attempt_count"] == 1
    assert rejected.info["transition_count"] == 0
    assert rejected.observation.metadata["value"] == 0
    assert accepted.accepted is True
    assert accepted.info["attempt_count"] == 2
    assert accepted.info["transition_count"] == 1
    assert len(env.trajectory.steps) == 2
    assert env.trajectory.steps[0].action is None
    assert env.trajectory.steps[1].action is not None


def test_max_attempts_counts_penalized_invalid_attempts() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(
            max_attempts=2,
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
    )
    env.reset(seed=2)

    first = env.step("bad")
    second = env.step("1")

    assert first.truncated is False
    assert second.accepted is True
    assert second.truncated is True
    assert second.info["truncated_reason"] == "max_attempts"


def test_max_transitions_only_counts_accepted_state_changes() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(
            max_transitions=1,
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
    )
    env.reset(seed=3)

    rejected = env.step("bad")
    accepted = env.step("1")

    assert rejected.truncated is False
    assert accepted.accepted is True
    assert accepted.truncated is True
    assert accepted.info["truncated_reason"] == "max_transitions"


def test_apply_time_invalid_action_uses_env_policy() -> None:
    env = make_counter_env(
        backend=ApplyRejectingCounterBackend(),
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-3.0,
            )
        ),
    )
    env.reset(seed=8)

    rejected = env.step("zero")

    assert rejected.accepted is False
    assert rejected.reward == -3.0
    assert rejected.info["invalid_action"] is True
    assert rejected.info["transition_count"] == 0
    assert env.state.value == 0
    assert len(env.trajectory.steps) == 1
