"""Asynchronous process-backed task-session pool helpers."""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
import multiprocessing
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
import time
import traceback
from typing import TYPE_CHECKING, Any, cast

from rlvr_games.core.exceptions import EpisodeFinishedError, EnvironmentNotResetError
from rlvr_games.core.protocol import Environment
from rlvr_games.core.session import (
    EnvironmentTaskSession,
    TaskResetResult,
    TaskSessionProtocol,
    TaskSubmissionResult,
    TaskTrajectory,
    TaskTurn,
)

_DEFAULT_START_METHOD = "spawn"
_DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
_DEFAULT_CLOSE_JOIN_TIMEOUT_SECONDS = 5.0
_DEFAULT_TERMINATE_JOIN_TIMEOUT_SECONDS = 1.0

SessionFactory = Callable[[], TaskSessionProtocol]
EnvFactory = Callable[[], Environment[Any, Any]]

if TYPE_CHECKING:
    from rlvr_games.core.task_spec_base import TaskSpec


@dataclass(slots=True)
class AsyncTaskResetResult:
    """Result returned when one async task-session worker finishes reset."""

    slot_id: int
    episode_index: int
    reset_result: TaskResetResult

    @property
    def task_instance_id(self) -> str:
        """Return the reset task-instance id."""
        return self.reset_result.task_instance_id

    @property
    def episode_finished(self) -> bool:
        """Return whether reset produced no actionable turn."""
        return self.reset_result.turn is None


@dataclass(slots=True)
class AsyncTaskSubmissionResult:
    """Result returned when one async task-session worker finishes submit."""

    slot_id: int
    episode_index: int
    submission_result: TaskSubmissionResult

    @property
    def task_instance_id(self) -> str:
        """Return the submission task-instance id."""
        return self.submission_result.task_instance_id

    @property
    def episode_finished(self) -> bool:
        """Return whether the submission finished the task session."""
        return self.submission_result.done


@dataclass(slots=True, frozen=True)
class _WorkerStarted:
    """Successful worker-start handshake."""


@dataclass(slots=True, frozen=True)
class _ResetCommand:
    """Reset one worker-owned task session."""

    seed: int


@dataclass(slots=True, frozen=True)
class _SubmitCommand:
    """Submit one assistant output to a worker-owned task session."""

    assistant_output: str


@dataclass(slots=True, frozen=True)
class _CloseCommand:
    """Request graceful worker shutdown."""


@dataclass(slots=True)
class _WorkerResetResult:
    """Internal reset response sent from worker to parent."""

    episode_index: int
    reset_result: TaskResetResult


@dataclass(slots=True)
class _WorkerSubmissionResult:
    """Internal submission response sent from worker to parent."""

    episode_index: int
    submission_result: TaskSubmissionResult


@dataclass(slots=True)
class _WorkerException:
    """Serializable exception payload sent from worker to parent."""

    exception_type: type[BaseException]
    message: str
    traceback_text: str


def _build_env_task_session(
    *,
    env_factory: EnvFactory,
    task_kind: str,
) -> TaskSessionProtocol:
    """Build an environment-backed task session from an env factory."""
    return EnvironmentTaskSession(env=env_factory(), task_kind=task_kind)


def _build_session_from_task_spec(*, task_spec: "TaskSpec") -> TaskSessionProtocol:
    """Build one task session from a validated task spec."""
    from rlvr_games.task_specs import build_task_session_from_task_spec

    return build_task_session_from_task_spec(task_spec=task_spec)


def _load_session_from_task_spec_path(*, path: Path) -> TaskSessionProtocol:
    """Load and build one task session from a task-spec path."""
    from rlvr_games.task_specs import load_task_session_from_task_spec_path

    return load_task_session_from_task_spec_path(path=path)


def _safe_send(*, connection: Connection, payload: object) -> bool:
    """Send one payload to the parent if the worker pipe is still open."""
    try:
        connection.send(payload)
    except (BrokenPipeError, EOFError, OSError):
        return False
    return True


def _build_worker_exception(*, exc: Exception) -> _WorkerException:
    """Convert one caught exception into a serializable payload."""
    return _WorkerException(
        exception_type=type(exc),
        message=str(exc),
        traceback_text=traceback.format_exc(),
    )


def _worker_main(
    *,
    connection: Connection,
    session_factory: SessionFactory,
) -> None:
    """Build one task session inside a worker process and serve commands."""
    session = None
    try:
        session = session_factory()
        if not _safe_send(connection=connection, payload=_WorkerStarted()):
            return

        episode_index = -1
        while True:
            try:
                command = connection.recv()
            except EOFError:
                break

            if isinstance(command, _CloseCommand):
                break

            try:
                if isinstance(command, _ResetCommand):
                    next_episode_index = episode_index + 1
                    reset_result = session.reset(seed=command.seed)
                    response = _WorkerResetResult(
                        episode_index=next_episode_index,
                        reset_result=reset_result,
                    )
                    episode_index = next_episode_index
                    if not _safe_send(connection=connection, payload=response):
                        break
                    continue

                if isinstance(command, _SubmitCommand):
                    submission_result = session.submit(command.assistant_output)
                    response = _WorkerSubmissionResult(
                        episode_index=episode_index,
                        submission_result=submission_result,
                    )
                    if not _safe_send(connection=connection, payload=response):
                        break
                    continue

                raise RuntimeError(
                    f"Worker received unsupported command type: {type(command)!r}."
                )
            except Exception as exc:
                if not _safe_send(
                    connection=connection,
                    payload=_build_worker_exception(exc=exc),
                ):
                    break
    except Exception as exc:
        _safe_send(connection=connection, payload=_build_worker_exception(exc=exc))
    finally:
        if session is not None:
            session.close()
        connection.close()


class AsyncSessionPool:
    """Process-backed async pool that owns one task session per slot."""

    def __init__(
        self,
        *,
        session_factories: Sequence[SessionFactory],
        start_method: str = _DEFAULT_START_METHOD,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        """Start one worker per supplied task-session factory."""
        if not session_factories:
            raise ValueError("AsyncSessionPool requires at least one session factory.")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive.")

        self._closed = False
        self._connections_by_slot: dict[int, Connection] = {}
        self._processes_by_slot: dict[int, BaseProcess] = {}
        self._slot_ids_by_fileno: dict[int, int] = {}
        self._busy_slot_ids: set[int] = set()
        self._lease_token_by_slot: dict[int, int] = {}
        self._buffered_results: deque[
            tuple[int, int, AsyncTaskResetResult | AsyncTaskSubmissionResult]
        ] = deque()
        self._buffered_exceptions: deque[tuple[int, int, BaseException]] = deque()
        self._next_command_id_by_slot: dict[int, int] = {}
        self._pending_command_id_by_slot: dict[int, int] = {}
        self._next_lease_token = 0

        context = multiprocessing.get_context(start_method)
        try:
            process_factory = cast(
                Callable[..., BaseProcess],
                getattr(context, "Process"),
            )
            for slot_id, session_factory in enumerate(session_factories):
                parent_connection, child_connection = context.Pipe()
                process = process_factory(
                    target=_worker_main,
                    kwargs={
                        "connection": child_connection,
                        "session_factory": session_factory,
                    },
                    name=f"rlvr-async-session-{slot_id}",
                )
                process.start()
                child_connection.close()

                self._connections_by_slot[slot_id] = parent_connection
                self._processes_by_slot[slot_id] = process
                self._slot_ids_by_fileno[parent_connection.fileno()] = slot_id
                self._next_command_id_by_slot[slot_id] = 0

            self._wait_for_worker_startup(timeout_seconds=startup_timeout_seconds)
        except Exception:
            self.close()
            raise

    @classmethod
    def from_env_factories(
        cls,
        *,
        env_factories: Sequence[EnvFactory],
        task_kind: str = "environment",
        start_method: str = _DEFAULT_START_METHOD,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> "AsyncSessionPool":
        """Build a task-session pool from environment factories."""
        return cls(
            session_factories=tuple(
                partial(
                    _build_env_task_session,
                    env_factory=env_factory,
                    task_kind=task_kind,
                )
                for env_factory in env_factories
            ),
            start_method=start_method,
            startup_timeout_seconds=startup_timeout_seconds,
        )

    @classmethod
    def from_task_specs(
        cls,
        *,
        task_specs: Sequence["TaskSpec"],
        start_method: str = _DEFAULT_START_METHOD,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> "AsyncSessionPool":
        """Build a pool whose workers materialize sessions from task specs."""
        return cls(
            session_factories=tuple(
                partial(_build_session_from_task_spec, task_spec=task_spec)
                for task_spec in task_specs
            ),
            start_method=start_method,
            startup_timeout_seconds=startup_timeout_seconds,
        )

    @classmethod
    def from_task_spec_paths(
        cls,
        *,
        task_spec_paths: Sequence[Path],
        start_method: str = _DEFAULT_START_METHOD,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> "AsyncSessionPool":
        """Load task specs from disk, then build a session pool from them."""
        return cls(
            session_factories=tuple(
                partial(_load_session_from_task_spec_path, path=path)
                for path in task_spec_paths
            ),
            start_method=start_method,
            startup_timeout_seconds=startup_timeout_seconds,
        )

    @property
    def slot_count(self) -> int:
        """Return the number of worker slots owned by the pool."""
        return len(self._connections_by_slot)

    @property
    def pending_slot_ids(self) -> tuple[int, ...]:
        """Return unleased slot ids whose most recent command is still in flight."""
        return self._receivable_slot_ids(allow_leased=False)

    def reset(self, *, slot_id: int, seed: int) -> None:
        """Enqueue a reset command for one slot and return immediately."""
        self._enqueue_reset(slot_id=slot_id, seed=seed)

    def reset_all(self, *, seeds: Sequence[int]) -> None:
        """Enqueue one reset command per slot."""
        if len(seeds) != self.slot_count:
            raise ValueError("reset_all() requires exactly one seed per worker slot.")
        for slot_id, seed in enumerate(seeds):
            self.reset(slot_id=slot_id, seed=seed)

    def submit(self, *, slot_id: int, assistant_output: str) -> None:
        """Enqueue one assistant output for a slot and return immediately."""
        self._enqueue_submit(slot_id=slot_id, assistant_output=assistant_output)

    def session(self, *, slot_id: int, close_pool: bool = False) -> "AsyncTaskSession":
        """Return a task-session wrapper for one leased pool slot."""
        lease_token = self._lease_slot(slot_id=slot_id)
        try:
            return AsyncTaskSession(
                pool=self,
                slot_id=slot_id,
                lease_token=lease_token,
                close_pool=close_pool,
            )
        except Exception:
            self._release_slot(slot_id=slot_id, lease_token=lease_token)
            raise

    def recv(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncTaskResetResult | AsyncTaskSubmissionResult:
        """Wait for one slot result and return it."""
        results = self.recv_ready(max_results=1, timeout_seconds=timeout_seconds)
        return results[0]

    def recv_slot(
        self,
        *,
        slot_id: int,
        command_id: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncTaskResetResult | AsyncTaskSubmissionResult:
        """Wait for the next result produced by one specific unleased slot."""
        return self._recv_slot(
            slot_id=slot_id,
            command_id=command_id,
            timeout_seconds=timeout_seconds,
            allow_leased=False,
            lease_token=None,
        )

    def recv_ready(
        self,
        *,
        max_results: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[AsyncTaskResetResult | AsyncTaskSubmissionResult, ...]:
        """Wait for one or more ready slot results."""
        self._ensure_open()
        if max_results is not None and max_results <= 0:
            raise ValueError("max_results must be positive when provided.")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative when provided.")

        buffered_results = self._pop_buffered_results(
            max_results=max_results,
            allow_leased=False,
        )
        if buffered_results:
            return buffered_results
        buffered_exception = self._pop_buffered_exception(allow_leased=False)
        if buffered_exception is not None:
            raise buffered_exception
        pending_slot_ids = self._receivable_slot_ids(allow_leased=False)
        if not pending_slot_ids:
            raise RuntimeError("AsyncSessionPool has no unleased commands to receive.")

        ready_connections = cast(
            list[Connection],
            wait(
                [self._connections_by_slot[slot_id] for slot_id in pending_slot_ids],
                timeout=timeout_seconds,
            ),
        )
        if not ready_connections:
            raise TimeoutError("Timed out waiting for async task-session results.")

        if max_results is not None:
            ready_connections = ready_connections[:max_results]

        for connection in ready_connections:
            self._buffer_response(connection=connection)

        buffered_results = self._pop_buffered_results(
            max_results=max_results,
            allow_leased=False,
        )
        if buffered_results:
            return buffered_results
        buffered_exception = self._pop_buffered_exception(allow_leased=False)
        if buffered_exception is not None:
            raise buffered_exception
        raise RuntimeError("AsyncSessionPool received no buffered results.")

    def close(self) -> None:
        """Shut down worker processes and close their pipes."""
        if self._closed:
            return

        self._closed = True
        close_command = _CloseCommand()
        for connection in self._connections_by_slot.values():
            _safe_send(connection=connection, payload=close_command)

        for process in self._processes_by_slot.values():
            process.join(timeout=_DEFAULT_CLOSE_JOIN_TIMEOUT_SECONDS)

        for process in self._processes_by_slot.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=_DEFAULT_TERMINATE_JOIN_TIMEOUT_SECONDS)

        for connection in self._connections_by_slot.values():
            connection.close()

        self._connections_by_slot.clear()
        self._processes_by_slot.clear()
        self._slot_ids_by_fileno.clear()
        self._busy_slot_ids.clear()
        self._lease_token_by_slot.clear()
        self._buffered_results.clear()
        self._buffered_exceptions.clear()
        self._next_command_id_by_slot.clear()
        self._pending_command_id_by_slot.clear()

    def __enter__(self) -> "AsyncSessionPool":
        """Return the pool for context-manager use."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback_obj: object) -> None:
        """Close the pool when leaving a context manager."""
        del exc_type
        del exc
        del traceback_obj
        self.close()

    def _wait_for_worker_startup(self, *, timeout_seconds: float) -> None:
        """Wait for every worker to acknowledge successful startup."""
        for slot_id, connection in self._connections_by_slot.items():
            if not connection.poll(timeout_seconds):
                raise TimeoutError(
                    f"Timed out waiting for async session worker {slot_id} to start."
                )
            try:
                response = connection.recv()
            except EOFError as exc:
                process = self._processes_by_slot[slot_id]
                raise RuntimeError(
                    "Async session worker exited during startup "
                    f"(slot {slot_id}, exitcode={process.exitcode})."
                ) from exc

            if isinstance(response, _WorkerStarted):
                continue
            if isinstance(response, _WorkerException):
                raise self._materialize_worker_exception(
                    slot_id=slot_id,
                    response=response,
                )
            raise RuntimeError(
                f"Async session worker {slot_id} sent an unexpected startup response."
            )

    def _enqueue_reset(
        self,
        *,
        slot_id: int,
        seed: int,
        allow_leased: bool = False,
        lease_token: int | None = None,
    ) -> int:
        """Enqueue a reset command and return its per-slot command id."""
        return self._dispatch(
            slot_id=slot_id,
            command=_ResetCommand(seed=seed),
            allow_leased=allow_leased,
            lease_token=lease_token,
        )

    def _enqueue_submit(
        self,
        *,
        slot_id: int,
        assistant_output: str,
        allow_leased: bool = False,
        lease_token: int | None = None,
    ) -> int:
        """Enqueue a submit command and return its per-slot command id."""
        return self._dispatch(
            slot_id=slot_id,
            command=_SubmitCommand(assistant_output=assistant_output),
            allow_leased=allow_leased,
            lease_token=lease_token,
        )

    def _dispatch(
        self,
        *,
        slot_id: int,
        command: _ResetCommand | _SubmitCommand,
        allow_leased: bool,
        lease_token: int | None,
    ) -> int:
        """Send one command to an idle slot."""
        self._ensure_open()
        connection = self._connection_for_slot(slot_id=slot_id)
        self._ensure_slot_is_accessible(
            slot_id=slot_id,
            allow_leased=allow_leased,
            lease_token=lease_token,
        )
        if slot_id in self._busy_slot_ids:
            raise RuntimeError(
                f"Async session slot {slot_id} already has a pending task."
            )
        if self._slot_has_buffered_response(slot_id=slot_id):
            raise RuntimeError(
                f"Async session slot {slot_id} has an unread buffered result."
            )
        command_id = self._next_command_id_by_slot[slot_id]
        self._next_command_id_by_slot[slot_id] = command_id + 1
        try:
            connection.send(command)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"Async session worker for slot {slot_id} is not available."
            ) from exc
        self._busy_slot_ids.add(slot_id)
        self._pending_command_id_by_slot[slot_id] = command_id
        return command_id

    def _recv_slot(
        self,
        *,
        slot_id: int,
        command_id: int | None,
        timeout_seconds: float | None,
        allow_leased: bool,
        lease_token: int | None,
    ) -> AsyncTaskResetResult | AsyncTaskSubmissionResult:
        """Wait for the next result produced by one specific slot."""
        self._ensure_open()
        self._connection_for_slot(slot_id=slot_id)
        self._ensure_slot_is_accessible(
            slot_id=slot_id,
            allow_leased=allow_leased,
            lease_token=lease_token,
        )
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative when provided.")

        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds

        while True:
            buffered_result = self._pop_buffered_result_for_slot(
                slot_id=slot_id,
                command_id=command_id,
            )
            if buffered_result is not None:
                return buffered_result

            buffered_exception = self._pop_buffered_exception_for_slot(
                slot_id=slot_id,
                command_id=command_id,
            )
            if buffered_exception is not None:
                raise buffered_exception

            if slot_id not in self._busy_slot_ids:
                raise RuntimeError(
                    f"Async session slot {slot_id} has no pending command to receive."
                )
            if command_id is not None:
                pending_command_id = self._pending_command_id_by_slot.get(slot_id)
                if pending_command_id != command_id:
                    raise RuntimeError(
                        f"Async session slot {slot_id} has no pending command "
                        f"with id {command_id}."
                    )

            remaining_timeout = None
            if deadline is not None:
                remaining_timeout = max(0.0, deadline - time.monotonic())

            ready_connections = cast(
                list[Connection],
                wait(
                    [
                        self._connections_by_slot[pending_slot_id]
                        for pending_slot_id in self._busy_slot_ids
                    ],
                    timeout=remaining_timeout,
                ),
            )
            if not ready_connections:
                raise TimeoutError("Timed out waiting for async task-session results.")

            for connection in ready_connections:
                self._buffer_response(connection=connection)

    def _buffer_response(self, *, connection: Connection) -> None:
        """Receive one worker response and buffer its translated outcome."""
        fileno = connection.fileno()
        slot_id = self._slot_ids_by_fileno[fileno]
        self._busy_slot_ids.remove(slot_id)
        command_id = self._pending_command_id_by_slot.pop(slot_id)

        try:
            response = connection.recv()
        except EOFError:
            process = self._processes_by_slot[slot_id]
            self._buffered_exceptions.append(
                (
                    slot_id,
                    command_id,
                    RuntimeError(
                        "Async session worker exited while a command was in flight "
                        f"(slot {slot_id}, exitcode={process.exitcode})."
                    ),
                )
            )
            return

        if isinstance(response, _WorkerException):
            self._buffered_exceptions.append(
                (
                    slot_id,
                    command_id,
                    self._materialize_worker_exception(
                        slot_id=slot_id,
                        response=response,
                    ),
                )
            )
            return
        if isinstance(response, _WorkerResetResult):
            self._buffered_results.append(
                (
                    slot_id,
                    command_id,
                    AsyncTaskResetResult(
                        slot_id=slot_id,
                        episode_index=response.episode_index,
                        reset_result=response.reset_result,
                    ),
                )
            )
            return
        if isinstance(response, _WorkerSubmissionResult):
            self._buffered_results.append(
                (
                    slot_id,
                    command_id,
                    AsyncTaskSubmissionResult(
                        slot_id=slot_id,
                        episode_index=response.episode_index,
                        submission_result=response.submission_result,
                    ),
                )
            )
            return
        self._buffered_exceptions.append(
            (
                slot_id,
                command_id,
                RuntimeError(
                    f"Async session worker {slot_id} returned an unknown response."
                ),
            )
        )

    def _materialize_worker_exception(
        self,
        *,
        slot_id: int,
        response: _WorkerException,
    ) -> BaseException:
        """Reconstruct one worker-side exception."""
        try:
            exc = response.exception_type(response.message)
        except Exception:
            exc = RuntimeError(
                f"{response.exception_type.__name__}: {response.message}"
            )
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(f"Raised by async session worker slot {slot_id}.")
            add_note(response.traceback_text.rstrip())
        return exc

    def _pop_buffered_results(
        self,
        *,
        max_results: int | None,
        allow_leased: bool,
    ) -> tuple[AsyncTaskResetResult | AsyncTaskSubmissionResult, ...]:
        """Pop and return up to ``max_results`` buffered successful results."""
        if not self._buffered_results:
            return ()

        if max_results is None:
            max_results = len(self._buffered_results)

        results: list[AsyncTaskResetResult | AsyncTaskSubmissionResult] = []
        remaining_results: deque[
            tuple[int, int, AsyncTaskResetResult | AsyncTaskSubmissionResult]
        ] = deque()
        while self._buffered_results:
            slot_id, command_id, result = self._buffered_results.popleft()
            if len(results) < max_results and (
                allow_leased or slot_id not in self._lease_token_by_slot
            ):
                results.append(result)
                continue
            remaining_results.append((slot_id, command_id, result))
        self._buffered_results = remaining_results
        return tuple(results)

    def _pop_buffered_result_for_slot(
        self,
        *,
        slot_id: int,
        command_id: int | None,
    ) -> AsyncTaskResetResult | AsyncTaskSubmissionResult | None:
        """Pop and return one buffered successful result for ``slot_id``."""
        buffered_result = None
        remaining_results: deque[
            tuple[int, int, AsyncTaskResetResult | AsyncTaskSubmissionResult]
        ] = deque()
        while self._buffered_results:
            result_slot_id, result_command_id, result = self._buffered_results.popleft()
            if (
                buffered_result is None
                and result_slot_id == slot_id
                and (command_id is None or result_command_id == command_id)
            ):
                buffered_result = result
                continue
            remaining_results.append((result_slot_id, result_command_id, result))
        self._buffered_results = remaining_results
        return buffered_result

    def _pop_buffered_exception_for_slot(
        self,
        *,
        slot_id: int,
        command_id: int | None,
    ) -> BaseException | None:
        """Pop and return one buffered exception for ``slot_id``."""
        buffered_exception = None
        remaining_exceptions: deque[tuple[int, int, BaseException]] = deque()
        while self._buffered_exceptions:
            exception_slot_id, exception_command_id, exception = (
                self._buffered_exceptions.popleft()
            )
            if (
                buffered_exception is None
                and exception_slot_id == slot_id
                and (command_id is None or exception_command_id == command_id)
            ):
                buffered_exception = exception
                continue
            remaining_exceptions.append(
                (exception_slot_id, exception_command_id, exception)
            )
        self._buffered_exceptions = remaining_exceptions
        return buffered_exception

    def _pop_buffered_exception(self, *, allow_leased: bool) -> BaseException | None:
        """Pop and return one buffered exception visible to the caller."""
        remaining_exceptions: deque[tuple[int, int, BaseException]] = deque()
        buffered_exception = None
        while self._buffered_exceptions:
            slot_id, command_id, exception = self._buffered_exceptions.popleft()
            if buffered_exception is None and (
                allow_leased or slot_id not in self._lease_token_by_slot
            ):
                buffered_exception = exception
                continue
            remaining_exceptions.append((slot_id, command_id, exception))
        self._buffered_exceptions = remaining_exceptions
        return buffered_exception

    def _connection_for_slot(self, *, slot_id: int) -> Connection:
        """Return the parent connection for a validated slot id."""
        connection = self._connections_by_slot.get(slot_id)
        if connection is None:
            raise IndexError(f"Async session slot {slot_id} does not exist.")
        return connection

    def _lease_slot(self, *, slot_id: int) -> int:
        """Mark one slot as exclusively owned by an async task session."""
        self._ensure_open()
        self._connection_for_slot(slot_id=slot_id)
        if slot_id in self._lease_token_by_slot:
            raise RuntimeError(
                f"Async session slot {slot_id} is already leased to a task session."
            )
        if slot_id in self._busy_slot_ids:
            raise RuntimeError(
                f"Async session slot {slot_id} cannot be leased while a command is pending."
            )
        if self._slot_has_buffered_response(slot_id=slot_id):
            raise RuntimeError(
                f"Async session slot {slot_id} has an unread buffered result."
            )
        lease_token = self._next_lease_token
        self._next_lease_token += 1
        self._lease_token_by_slot[slot_id] = lease_token
        return lease_token

    def _release_slot(self, *, slot_id: int, lease_token: int) -> None:
        """Release a task-session lease on one slot."""
        if self._closed:
            return
        current_lease_token = self._lease_token_by_slot.get(slot_id)
        if current_lease_token != lease_token:
            raise RuntimeError(
                f"Async session slot {slot_id} is not owned by this task session."
            )
        if slot_id in self._busy_slot_ids or self._slot_has_buffered_response(
            slot_id=slot_id
        ):
            raise RuntimeError(
                f"Async session slot {slot_id} still has in-flight or unread work."
            )
        del self._lease_token_by_slot[slot_id]

    def _receivable_slot_ids(self, *, allow_leased: bool) -> tuple[int, ...]:
        """Return pending slot ids visible to one receive caller."""
        return tuple(
            slot_id
            for slot_id in sorted(self._busy_slot_ids)
            if allow_leased or slot_id not in self._lease_token_by_slot
        )

    def _slot_has_buffered_response(self, *, slot_id: int) -> bool:
        """Return whether one slot has an unread buffered result or exception."""
        return any(
            result_slot_id == slot_id for result_slot_id, _, _ in self._buffered_results
        ) or any(
            exception_slot_id == slot_id
            for exception_slot_id, _, _ in self._buffered_exceptions
        )

    def _ensure_slot_is_accessible(
        self,
        *,
        slot_id: int,
        allow_leased: bool,
        lease_token: int | None,
    ) -> None:
        """Raise when a caller tries to access a slot leased to a session."""
        current_lease_token = self._lease_token_by_slot.get(slot_id)
        if allow_leased:
            if lease_token is None or current_lease_token != lease_token:
                raise RuntimeError(
                    f"Async session slot {slot_id} is not owned by this task session."
                )
            return
        if current_lease_token is not None:
            raise RuntimeError(
                f"Async session slot {slot_id} is leased to a task session."
            )

    def _ensure_open(self) -> None:
        """Raise if the pool has already been closed."""
        if self._closed:
            raise RuntimeError("AsyncSessionPool has already been closed.")


class AsyncTaskSession:
    """Task-session wrapper backed by one leased async pool slot."""

    def __init__(
        self,
        *,
        pool: AsyncSessionPool,
        slot_id: int,
        lease_token: int,
        close_pool: bool = False,
    ) -> None:
        """Initialize the leased async task-session wrapper."""
        self._pool = pool
        self._slot_id = slot_id
        self._lease_token: int | None = lease_token
        self._close_pool = close_pool
        self._task_instance_id: str | None = None
        self._turn: TaskTurn | None = None
        self._trajectory: TaskTrajectory | None = None
        self._episode_return = 0.0
        self._done = False

    @property
    def done(self) -> bool:
        """Return whether the current scalar session has finished."""
        return self._done

    @property
    def task_instance_id(self) -> str:
        """Return the active task-instance id."""
        if self._task_instance_id is None:
            raise EnvironmentNotResetError(
                "Call AsyncTaskSession.reset() before task_instance_id."
            )
        return self._task_instance_id

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current actionable turn, if one exists."""
        if self._trajectory is None:
            raise EnvironmentNotResetError("Call AsyncTaskSession.reset() before turn.")
        return self._turn

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the locally mirrored task trajectory."""
        if self._trajectory is None:
            raise EnvironmentNotResetError(
                "Call AsyncTaskSession.reset() before trajectory."
            )
        return self._trajectory

    @property
    def episode_return(self) -> float:
        """Return cumulative reward for the active scalar session."""
        return self._episode_return

    def reset(self, *, seed: int) -> TaskResetResult:
        """Reset the leased remote task session."""
        self._ensure_open_lease()
        command_id = self._pool._enqueue_reset(
            slot_id=self._slot_id,
            seed=seed,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        result = self._pool._recv_slot(
            slot_id=self._slot_id,
            command_id=command_id,
            timeout_seconds=None,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        if not isinstance(result, AsyncTaskResetResult):
            raise RuntimeError("Expected AsyncTaskResetResult during reset.")
        reset_result = result.reset_result
        self._task_instance_id = reset_result.task_instance_id
        self._turn = reset_result.turn
        self._done = reset_result.turn is None
        self._episode_return = 0.0
        self._trajectory = TaskTrajectory(
            task_instance_id=reset_result.task_instance_id,
            initial_turn=reset_result.turn,
            reset_info=reset_result.info,
        )
        return reset_result

    def submit(self, assistant_output: str) -> TaskSubmissionResult:
        """Submit one assistant output to the leased remote task session."""
        self._ensure_open_lease()
        if self._trajectory is None:
            raise EnvironmentNotResetError(
                "Call AsyncTaskSession.reset() before submit()."
            )
        if self._turn is None:
            raise EpisodeFinishedError(
                "The current task session has finished. Call reset() first."
            )
        command_id = self._pool._enqueue_submit(
            slot_id=self._slot_id,
            assistant_output=assistant_output,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        result = self._pool._recv_slot(
            slot_id=self._slot_id,
            command_id=command_id,
            timeout_seconds=None,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        if not isinstance(result, AsyncTaskSubmissionResult):
            raise RuntimeError("Expected AsyncTaskSubmissionResult during submit.")
        submission_result = result.submission_result
        self._turn = submission_result.turn
        self._done = submission_result.done
        self._episode_return += submission_result.reward
        self.trajectory.append(submission_result, details={"adapter": "async_session"})
        return submission_result

    def close(self) -> None:
        """Release the leased pool slot or close the pool."""
        if self._close_pool:
            self._pool.close()
            self._lease_token = None
            return
        if self._lease_token is None:
            return
        self._pool._release_slot(
            slot_id=self._slot_id,
            lease_token=self._lease_token,
        )
        self._lease_token = None

    def _ensure_open_lease(self) -> None:
        """Raise when this wrapper no longer owns a pool slot."""
        if self._lease_token is None:
            raise RuntimeError("AsyncTaskSession is closed.")


__all__ = [
    "AsyncSessionPool",
    "AsyncTaskResetResult",
    "AsyncTaskSession",
    "AsyncTaskSubmissionResult",
    "EnvFactory",
    "SessionFactory",
]
