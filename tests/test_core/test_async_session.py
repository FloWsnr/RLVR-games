"""Tests for the async task-session pool."""

from pathlib import Path

import pytest

from rlvr_games.core import (
    AsyncSessionPool,
    AsyncTaskResetResult,
    AsyncTaskSession,
    AsyncTaskSubmissionResult,
    EnvironmentNotResetError,
    EnvironmentTaskSession,
    EpisodeConfig,
    EpisodeFinishedError,
    TaskSessionProtocol,
    TurnBasedEnv,
)
from rlvr_games.tasks.arithmetic import ArithmeticOperation, ArithmeticTaskSource
from rlvr_games.tasks.arithmetic import make_arithmetic_session
from rlvr_games.task_specs import load_task_spec

from tests.test_core.support import (
    CounterAction,
    CounterBackend,
    CounterState,
    make_counter_env,
)


def _build_async_arithmetic_session() -> TaskSessionProtocol:
    """Return one spawn-safe arithmetic verifier session."""
    return make_arithmetic_session(
        task_source=ArithmeticTaskSource(
            min_value=2,
            max_value=2,
            operations=(ArithmeticOperation.ADD,),
        )
    )


def _build_async_counter_env() -> TurnBasedEnv[CounterState, CounterAction]:
    """Return one spawn-safe counter env."""
    return make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )


def _build_async_counter_task_session() -> TaskSessionProtocol:
    """Return one spawn-safe environment-backed task session."""
    return EnvironmentTaskSession(
        env=_build_async_counter_env(),
        task_kind="counter",
    )


def _build_failing_session() -> TaskSessionProtocol:
    """Raise during worker construction."""
    raise RuntimeError("session factory failed")


def test_async_session_pool_runs_single_step_verifier_session() -> None:
    with AsyncSessionPool(session_factories=(_build_async_arithmetic_session,)) as pool:
        pool.reset(slot_id=0, seed=0)

        reset_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(reset_result, AsyncTaskResetResult)
        assert reset_result.slot_id == 0
        assert reset_result.episode_index == 0
        assert reset_result.episode_finished is False
        assert reset_result.reset_result.turn is not None
        assert reset_result.reset_result.observation is not None
        observation_text = reset_result.reset_result.observation.text
        assert observation_text is not None
        assert "2 + 2" in observation_text

        pool.submit(slot_id=0, assistant_output="4")
        submission_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(submission_result, AsyncTaskSubmissionResult)
        assert submission_result.slot_id == 0
        assert submission_result.episode_index == 0
        assert submission_result.episode_finished is True
        assert submission_result.submission_result.valid_submission is True
        assert submission_result.submission_result.reward == 1.0
        assert submission_result.submission_result.parsed_output == 4


def test_async_session_pool_can_wrap_environment_factories() -> None:
    with AsyncSessionPool.from_env_factories(
        env_factories=(_build_async_counter_env,),
        task_kind="counter",
    ) as pool:
        pool.reset(slot_id=0, seed=3)
        reset_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(reset_result, AsyncTaskResetResult)
        assert reset_result.reset_result.turn is not None
        assert reset_result.reset_result.turn.action_context.turn_index == 0
        assert reset_result.reset_result.observation is not None
        assert reset_result.reset_result.observation.metadata["value"] == 0

        pool.submit(slot_id=0, assistant_output="1")
        submission_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(submission_result, AsyncTaskSubmissionResult)
        assert submission_result.submission_result.valid_submission is True
        assert submission_result.submission_result.reward == 1.0
        assert submission_result.submission_result.turn is not None
        assert submission_result.submission_result.turn.action_context.turn_index == 1
        assert submission_result.submission_result.observation is not None
        assert submission_result.submission_result.observation.metadata["value"] == 1


def test_async_session_pool_can_build_workers_from_task_spec_paths() -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "tasks"
        / "arithmetic"
        / "simple_addition.yaml"
    )

    with AsyncSessionPool.from_task_spec_paths(
        task_spec_paths=(task_spec_path,)
    ) as pool:
        pool.reset(slot_id=0, seed=0)
        assert isinstance(pool.recv(timeout_seconds=5.0), AsyncTaskResetResult)

        pool.submit(slot_id=0, assistant_output="4")
        submission_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(submission_result, AsyncTaskSubmissionResult)
        assert submission_result.submission_result.reward == 1.0


def test_async_session_pool_can_build_workers_from_task_specs() -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "tasks"
        / "arithmetic"
        / "simple_addition.yaml"
    )
    task_spec = load_task_spec(path=task_spec_path)

    with AsyncSessionPool.from_task_specs(task_specs=(task_spec,)) as pool:
        pool.reset(slot_id=0, seed=0)
        assert isinstance(pool.recv(timeout_seconds=5.0), AsyncTaskResetResult)

        pool.submit(slot_id=0, assistant_output="4")
        submission_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(submission_result, AsyncTaskSubmissionResult)
        assert submission_result.submission_result.reward == 1.0


def test_async_task_session_leases_slot_and_records_trajectory() -> None:
    with AsyncSessionPool(session_factories=(_build_async_arithmetic_session,)) as pool:
        session = pool.session(slot_id=0)

        assert isinstance(session, AsyncTaskSession)
        with pytest.raises(RuntimeError, match="leased"):
            pool.reset(slot_id=0, seed=0)

        reset_result = session.reset(seed=0)
        assert reset_result.turn is not None
        assert session.task_instance_id == reset_result.task_instance_id
        assert session.trajectory.initial_turn is reset_result.turn

        submission_result = session.submit("4")
        assert submission_result.done is True
        assert session.done is True
        assert session.turn is None
        assert session.episode_return == 1.0
        assert len(session.trajectory.submissions) == 1
        assert session.trajectory.total_reward == 1.0

        with pytest.raises(EpisodeFinishedError):
            session.submit("4")

        session.close()
        pool.reset(slot_id=0, seed=1)
        assert isinstance(pool.recv(timeout_seconds=5.0), AsyncTaskResetResult)


def test_async_task_session_rejects_access_before_reset() -> None:
    with AsyncSessionPool(session_factories=(_build_async_arithmetic_session,)) as pool:
        session = pool.session(slot_id=0)

        with pytest.raises(EnvironmentNotResetError):
            _ = session.task_instance_id
        with pytest.raises(EnvironmentNotResetError):
            _ = session.turn
        with pytest.raises(EnvironmentNotResetError):
            _ = session.trajectory
        with pytest.raises(EnvironmentNotResetError):
            session.submit("4")


def test_async_session_pool_propagates_worker_command_exceptions() -> None:
    with AsyncSessionPool(session_factories=(_build_async_arithmetic_session,)) as pool:
        pool.submit(slot_id=0, assistant_output="4")

        with pytest.raises(EnvironmentNotResetError):
            pool.recv(timeout_seconds=5.0)

        pool.reset(slot_id=0, seed=0)
        assert isinstance(pool.recv(timeout_seconds=5.0), AsyncTaskResetResult)


def test_async_session_pool_propagates_worker_startup_exceptions() -> None:
    with pytest.raises(RuntimeError, match="session factory failed"):
        AsyncSessionPool(
            session_factories=(_build_failing_session,),
            startup_timeout_seconds=5.0,
        )


def test_async_session_pool_rejects_busy_and_buffered_slot_dispatch() -> None:
    with AsyncSessionPool(
        session_factories=(
            _build_async_counter_task_session,
            _build_async_counter_task_session,
        )
    ) as pool:
        pool.reset(slot_id=0, seed=0)
        with pytest.raises(RuntimeError, match="pending"):
            pool.submit(slot_id=0, assistant_output="1")

        pool.reset(slot_id=1, seed=1)
        ready_results = []
        while len(ready_results) < 2:
            ready_results.extend(pool.recv_ready(max_results=2, timeout_seconds=5.0))
        assert len(ready_results) == 2

        pool.submit(slot_id=0, assistant_output="1")
        pool.submit(slot_id=1, assistant_output="1")
        assert pool.pending_slot_ids == (0, 1)


def test_async_session_pool_close_is_idempotent() -> None:
    pool = AsyncSessionPool(session_factories=(_build_async_arithmetic_session,))
    pool.close()
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        pool.reset(slot_id=0, seed=0)


def test_async_counter_task_session_factory_is_spawn_safe() -> None:
    session = _build_async_counter_task_session()

    reset_result = session.reset(seed=0)

    assert reset_result.turn is not None
    assert reset_result.turn.observation.metadata["value"] == 0
