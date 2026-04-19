"""Generic task-session rollout tests."""

import pytest

from rlvr_games.core import (
    EnvironmentTaskSession,
    EpisodeConfig,
    TaskTurn,
    rollout_task_session,
)
from rlvr_games.tasks.arithmetic import (
    ArithmeticOperation,
    ArithmeticTaskSource,
    make_arithmetic_session,
)
from tests.test_core.support import CounterBackend, make_counter_env


def test_rollout_task_session_drives_environment_backed_session() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = EnvironmentTaskSession(env=env, task_kind="counter")

    trajectory = rollout_task_session(
        session=session,
        seed=7,
        policy=lambda turn: "1",
        max_submissions=3,
    )

    assert trajectory.task_instance_id == "counter:seed=7:episode=0"
    assert len(trajectory.submissions) == 3
    assert trajectory.total_reward == 3.0
    assert trajectory.done is True
    assert session.done is True


def test_rollout_task_session_drives_single_step_session() -> None:
    session = make_arithmetic_session(
        task_source=ArithmeticTaskSource(
            min_value=2,
            max_value=2,
            operations=(ArithmeticOperation.ADD,),
        )
    )

    trajectory = rollout_task_session(
        session=session,
        seed=0,
        policy=lambda turn: "4",
    )

    assert trajectory.task_instance_id.startswith("arithmetic:")
    assert len(trajectory.submissions) == 1
    assert trajectory.submissions[0].valid_submission is True
    assert trajectory.total_reward == 1.0
    assert trajectory.done is True


def test_rollout_task_session_guard_rejects_non_positive_max_submissions() -> None:
    session = make_arithmetic_session()

    with pytest.raises(ValueError, match="max_submissions"):
        rollout_task_session(
            session=session, seed=0, policy=lambda turn: "0", max_submissions=0
        )


def test_rollout_task_session_guard_stops_non_terminating_sessions() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = EnvironmentTaskSession(env=env, task_kind="counter")

    with pytest.raises(RuntimeError, match="max_submissions"):
        rollout_task_session(
            session=session,
            seed=7,
            policy=lambda turn: "1",
            max_submissions=2,
        )


def test_rollout_task_session_policy_receives_task_turn() -> None:
    session = make_arithmetic_session(
        task_source=ArithmeticTaskSource(
            min_value=2,
            max_value=2,
            operations=(ArithmeticOperation.ADD,),
        )
    )
    seen_turns: list[TaskTurn] = []

    def policy(turn: TaskTurn) -> str:
        seen_turns.append(turn)
        return "4"

    rollout_task_session(session=session, seed=0, policy=policy)

    assert len(seen_turns) == 1
    assert seen_turns[0].action_context.turn_index == 0
