"""Asynchronous process-backed environment pool helpers."""

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

from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import PreparedTurn, prepare_turn
from rlvr_games.core.types import Observation, StepResult

if TYPE_CHECKING:
    from rlvr_games.core.task_spec_base import TaskSpec
    from rlvr_games.core.workflow import AsyncWorkflowSession

_DEFAULT_START_METHOD = "spawn"
_DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
_DEFAULT_CLOSE_JOIN_TIMEOUT_SECONDS = 5.0
_DEFAULT_TERMINATE_JOIN_TIMEOUT_SECONDS = 1.0

EnvFactory = Callable[[], Environment[Any, Any]]


@dataclass(slots=True)
class AsyncResetResult:
    """Result returned when one async worker finishes a reset command.

    Attributes
    ----------
    slot_id : int
        Pool slot that produced the reset result.
    episode_index : int
        Zero-based episode index for this slot.
    observation : Observation
        Initial observation returned by the environment reset.
    reset_info : dict[str, object]
        Public-safe reset metadata emitted by the environment.
    turn : PreparedTurn | None
        Prepared next action turn, or `None` when the episode already finished
        during reset-time resolution.
    """

    slot_id: int
    episode_index: int
    observation: Observation
    reset_info: dict[str, object]
    turn: PreparedTurn | None

    @property
    def episode_finished(self) -> bool:
        """Return whether the slot has already finished the episode."""
        return self.turn is None


@dataclass(slots=True)
class AsyncStepResult:
    """Result returned when one async worker finishes a step command.

    Attributes
    ----------
    slot_id : int
        Pool slot that produced the step result.
    episode_index : int
        Zero-based episode index for this slot.
    step_result : StepResult
        Step transition outcome returned by the underlying environment.
    turn : PreparedTurn | None
        Prepared next action turn, or `None` when the episode finished after
        this step.
    """

    slot_id: int
    episode_index: int
    step_result: StepResult
    turn: PreparedTurn | None

    def __post_init__(self) -> None:
        """Validate that terminal step results do not expose a next turn."""
        if self.episode_finished and self.turn is not None:
            raise ValueError("Finished async step results must not contain a turn.")
        if not self.episode_finished and self.turn is None:
            raise ValueError("Non-terminal async step results require a turn.")

    @property
    def episode_finished(self) -> bool:
        """Return whether the slot finished after this step."""
        return self.step_result.terminated or self.step_result.truncated


@dataclass(slots=True, frozen=True)
class _WorkerStarted:
    """Successful worker-start handshake."""


@dataclass(slots=True, frozen=True)
class _ResetCommand:
    """Reset one worker-owned environment."""

    seed: int


@dataclass(slots=True, frozen=True)
class _StepCommand:
    """Step one worker-owned environment."""

    raw_action: str


@dataclass(slots=True, frozen=True)
class _CloseCommand:
    """Request graceful worker shutdown."""


@dataclass(slots=True)
class _WorkerResetResult:
    """Internal reset response sent from worker to parent."""

    episode_index: int
    observation: Observation
    reset_info: dict[str, object]
    turn: PreparedTurn | None


@dataclass(slots=True)
class _WorkerStepResult:
    """Internal step response sent from worker to parent."""

    episode_index: int
    step_result: StepResult
    turn: PreparedTurn | None


@dataclass(slots=True)
class _WorkerException:
    """Serializable exception payload sent from worker to parent."""

    exception_type: type[BaseException]
    message: str
    traceback_text: str


def _build_env_from_task_spec(*, task_spec: "TaskSpec") -> Environment[Any, Any]:
    """Build one environment from a validated task spec."""
    from rlvr_games.task_specs import build_environment_from_task_spec

    return build_environment_from_task_spec(task_spec=task_spec)


def _load_env_from_task_spec_path(*, path: Path) -> Environment[Any, Any]:
    """Load and build one environment from a task-spec path."""
    from rlvr_games.task_specs import load_environment_from_task_spec_path

    return load_environment_from_task_spec_path(path=path)


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
    env_factory: EnvFactory,
) -> None:
    """Build one environment inside a worker process and serve commands."""
    env = None
    try:
        env = env_factory()
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
                    observation, reset_info = env.reset(seed=command.seed)
                    turn = None
                    if not env.episode_finished:
                        turn = prepare_turn(env=env, observation=observation)
                    response = _WorkerResetResult(
                        episode_index=next_episode_index,
                        observation=observation,
                        reset_info=reset_info,
                        turn=turn,
                    )
                    episode_index = next_episode_index
                    if not _safe_send(connection=connection, payload=response):
                        break
                    continue

                if isinstance(command, _StepCommand):
                    step_result = env.step(command.raw_action)
                    turn = None
                    if not env.episode_finished:
                        turn = prepare_turn(
                            env=env,
                            observation=step_result.observation,
                        )
                    response = _WorkerStepResult(
                        episode_index=episode_index,
                        step_result=step_result,
                        turn=turn,
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
        if env is not None:
            env.close()
        connection.close()


class AsyncEnvPool:
    """Process-backed async pool that owns one live env per slot.

    The pool keeps one environment instance per worker process. Callers
    enqueue reset or step commands per slot, then receive results as soon as
    individual workers become ready.
    """

    def __init__(
        self,
        *,
        env_factories: Sequence[EnvFactory],
        start_method: str = _DEFAULT_START_METHOD,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        """Start one worker per supplied environment factory."""
        if not env_factories:
            raise ValueError("AsyncEnvPool requires at least one environment factory.")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive.")

        self._closed = False
        self._connections_by_slot: dict[int, Connection] = {}
        self._processes_by_slot: dict[int, BaseProcess] = {}
        self._slot_ids_by_fileno: dict[int, int] = {}
        self._busy_slot_ids: set[int] = set()
        self._lease_token_by_slot: dict[int, int] = {}
        self._buffered_results: deque[
            tuple[int, int, AsyncResetResult | AsyncStepResult]
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
            for slot_id, env_factory in enumerate(env_factories):
                parent_connection, child_connection = context.Pipe()
                process = process_factory(
                    target=_worker_main,
                    kwargs={
                        "connection": child_connection,
                        "env_factory": env_factory,
                    },
                    name=f"rlvr-async-env-{slot_id}",
                )
                process.start()
                child_connection.close()

                self._connections_by_slot[slot_id] = parent_connection
                self._processes_by_slot[slot_id] = process
                self._slot_ids_by_fileno[parent_connection.fileno()] = slot_id
                self._next_command_id_by_slot[slot_id] = 0

            self._wait_for_worker_startup(
                timeout_seconds=startup_timeout_seconds,
            )
        except Exception:
            self.close()
            raise

    @classmethod
    def from_task_specs(
        cls,
        *,
        task_specs: Sequence["TaskSpec"],
        start_method: str = _DEFAULT_START_METHOD,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> "AsyncEnvPool":
        """Build a pool whose workers materialize environments from task specs."""
        return cls(
            env_factories=tuple(
                partial(_build_env_from_task_spec, task_spec=task_spec)
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
    ) -> "AsyncEnvPool":
        """Load task specs from disk, then build a pool from them."""
        return cls(
            env_factories=tuple(
                partial(_load_env_from_task_spec_path, path=path)
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

    def step(self, *, slot_id: int, raw_action: str) -> None:
        """Enqueue one step command for a slot and return immediately."""
        self._enqueue_step(slot_id=slot_id, raw_action=raw_action)

    def session(
        self,
        *,
        slot_id: int,
        action_extractor: Callable[[str], str] | None = None,
        close_pool: bool = False,
    ) -> "AsyncWorkflowSession":
        """Return an async workflow-session wrapper for one pool slot."""
        from rlvr_games.core.workflow import AsyncWorkflowSession

        lease_token = self._lease_slot(slot_id=slot_id)
        try:
            return AsyncWorkflowSession(
                pool=self,
                slot_id=slot_id,
                lease_token=lease_token,
                action_extractor=action_extractor,
                close_pool=close_pool,
            )
        except Exception:
            self._release_slot(slot_id=slot_id, lease_token=lease_token)
            raise

    def workflow_session(
        self,
        *,
        slot_id: int,
        action_extractor: Callable[[str], str] | None = None,
        close_pool: bool = False,
    ) -> "AsyncWorkflowSession":
        """Return a workflow-session wrapper for one pool slot."""
        return self.session(
            slot_id=slot_id,
            action_extractor=action_extractor,
            close_pool=close_pool,
        )

    def recv(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncResetResult | AsyncStepResult:
        """Wait for one slot result and return it."""
        results = self.recv_ready(max_results=1, timeout_seconds=timeout_seconds)
        return results[0]

    def recv_slot(
        self,
        *,
        slot_id: int,
        command_id: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncResetResult | AsyncStepResult:
        """Wait for the next result produced by one specific unleased slot."""
        return self._recv_slot(
            slot_id=slot_id,
            command_id=command_id,
            timeout_seconds=timeout_seconds,
            allow_leased=False,
            lease_token=None,
        )

    def _recv_slot(
        self,
        *,
        slot_id: int,
        command_id: int | None,
        timeout_seconds: float | None,
        allow_leased: bool,
        lease_token: int | None,
    ) -> AsyncResetResult | AsyncStepResult:
        """Wait for the next result produced by one specific slot.

        Any responses from other slots that arrive while waiting are buffered
        and remain available through subsequent `recv()`/`recv_ready()` or
        `recv_slot()` calls.
        """
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
                    f"Async env slot {slot_id} has no pending command to receive."
                )
            if command_id is not None:
                pending_command_id = self._pending_command_id_by_slot.get(slot_id)
                if pending_command_id != command_id:
                    raise RuntimeError(
                        f"Async env slot {slot_id} has no pending command "
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
                raise TimeoutError("Timed out waiting for async environment results.")

            for connection in ready_connections:
                self._buffer_response(connection=connection)

    def recv_ready(
        self,
        *,
        max_results: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[AsyncResetResult | AsyncStepResult, ...]:
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
            raise RuntimeError("AsyncEnvPool has no unleased commands to receive.")

        ready_connections = cast(
            list[Connection],
            wait(
                [self._connections_by_slot[slot_id] for slot_id in pending_slot_ids],
                timeout=timeout_seconds,
            ),
        )
        if not ready_connections:
            raise TimeoutError("Timed out waiting for async environment results.")

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
        raise RuntimeError("AsyncEnvPool received no buffered results or exceptions.")

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

    def __enter__(self) -> "AsyncEnvPool":
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
                    f"Timed out waiting for async env worker {slot_id} to start."
                )
            try:
                response = connection.recv()
            except EOFError as exc:
                process = self._processes_by_slot[slot_id]
                raise RuntimeError(
                    "Async env worker exited during startup "
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
                f"Async env worker {slot_id} sent an unexpected startup response."
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

    def _enqueue_step(
        self,
        *,
        slot_id: int,
        raw_action: str,
        allow_leased: bool = False,
        lease_token: int | None = None,
    ) -> int:
        """Enqueue a step command and return its per-slot command id."""
        return self._dispatch(
            slot_id=slot_id,
            command=_StepCommand(raw_action=raw_action),
            allow_leased=allow_leased,
            lease_token=lease_token,
        )

    def _dispatch(
        self,
        *,
        slot_id: int,
        command: _ResetCommand | _StepCommand,
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
            raise RuntimeError(f"Async env slot {slot_id} already has a pending task.")
        if self._slot_has_buffered_response(slot_id=slot_id):
            raise RuntimeError(
                f"Async env slot {slot_id} has an unread buffered result."
            )
        command_id = self._next_command_id_by_slot[slot_id]
        self._next_command_id_by_slot[slot_id] = command_id + 1
        try:
            connection.send(command)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"Async env worker for slot {slot_id} is not available."
            ) from exc
        self._busy_slot_ids.add(slot_id)
        self._pending_command_id_by_slot[slot_id] = command_id
        return command_id

    def _buffer_response(
        self,
        *,
        connection: Connection,
    ) -> None:
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
                        "Async env worker exited while a command was in flight "
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
                    AsyncResetResult(
                        slot_id=slot_id,
                        episode_index=response.episode_index,
                        observation=response.observation,
                        reset_info=response.reset_info,
                        turn=response.turn,
                    ),
                )
            )
            return
        if isinstance(response, _WorkerStepResult):
            self._buffered_results.append(
                (
                    slot_id,
                    command_id,
                    AsyncStepResult(
                        slot_id=slot_id,
                        episode_index=response.episode_index,
                        step_result=response.step_result,
                        turn=response.turn,
                    ),
                )
            )
            return
        self._buffered_exceptions.append(
            (
                slot_id,
                command_id,
                RuntimeError(
                    f"Async env worker {slot_id} returned an unknown response."
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
            add_note(f"Raised by async env worker slot {slot_id}.")
            add_note(response.traceback_text.rstrip())
        return exc

    def _pop_buffered_results(
        self,
        *,
        max_results: int | None,
        allow_leased: bool,
    ) -> tuple[AsyncResetResult | AsyncStepResult, ...]:
        """Pop and return up to `max_results` buffered successful results."""
        if not self._buffered_results:
            return ()

        if max_results is None:
            max_results = len(self._buffered_results)

        results: list[AsyncResetResult | AsyncStepResult] = []
        remaining_results: deque[
            tuple[int, int, AsyncResetResult | AsyncStepResult]
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
    ) -> AsyncResetResult | AsyncStepResult | None:
        """Pop and return one buffered successful result for `slot_id`."""
        buffered_result = None
        remaining_results: deque[
            tuple[int, int, AsyncResetResult | AsyncStepResult]
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
        """Pop and return one buffered exception for `slot_id`."""
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
            raise IndexError(f"Async env slot {slot_id} does not exist.")
        return connection

    def _lease_slot(self, *, slot_id: int) -> int:
        """Mark one slot as exclusively owned by a workflow session."""
        self._ensure_open()
        self._connection_for_slot(slot_id=slot_id)
        if slot_id in self._lease_token_by_slot:
            raise RuntimeError(
                f"Async env slot {slot_id} is already leased to a workflow session."
            )
        if slot_id in self._busy_slot_ids:
            raise RuntimeError(
                f"Async env slot {slot_id} cannot be leased while a command is pending."
            )
        if self._slot_has_buffered_response(slot_id=slot_id):
            raise RuntimeError(
                f"Async env slot {slot_id} has an unread buffered result."
            )
        lease_token = self._next_lease_token
        self._next_lease_token += 1
        self._lease_token_by_slot[slot_id] = lease_token
        return lease_token

    def _release_slot(self, *, slot_id: int, lease_token: int) -> None:
        """Release a workflow-session lease on one slot."""
        if self._closed:
            return
        current_lease_token = self._lease_token_by_slot.get(slot_id)
        if current_lease_token != lease_token:
            raise RuntimeError(
                f"Async env slot {slot_id} is not owned by this workflow session."
            )
        if slot_id in self._busy_slot_ids or self._slot_has_buffered_response(
            slot_id=slot_id
        ):
            raise RuntimeError(
                f"Async env slot {slot_id} still has in-flight or unread work."
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
                    f"Async env slot {slot_id} is not owned by this workflow session."
                )
            return
        if current_lease_token is not None:
            raise RuntimeError(
                f"Async env slot {slot_id} is leased to a workflow session."
            )

    def _ensure_open(self) -> None:
        """Raise if the pool has already been closed."""
        if self._closed:
            raise RuntimeError("AsyncEnvPool has already been closed.")


__all__ = [
    "AsyncEnvPool",
    "AsyncResetResult",
    "AsyncStepResult",
]
