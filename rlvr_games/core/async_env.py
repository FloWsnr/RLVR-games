"""Asynchronous process-backed environment pool helpers."""

from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import multiprocessing
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
import traceback
from typing import TYPE_CHECKING, Any, cast

from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import PreparedTurn, prepare_turn
from rlvr_games.core.types import Observation, StepResult

if TYPE_CHECKING:
    from rlvr_games.core.task_spec_base import TaskSpec

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
        Initial observation returned by the environment reset. When `turn` is
        present, any image payloads are omitted because the trainer-facing
        message payload already carries them.
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
        Step transition outcome returned by the underlying environment. When
        `turn` is present, any image payloads are omitted from
        `step_result.observation` because the trainer-facing message payload
        already carries them.
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


def _observation_transport_copy(
    *,
    observation: Observation,
    turn: PreparedTurn | None,
) -> Observation:
    """Return the observation transport payload for one async result."""
    if turn is None or not observation.images:
        return observation
    return Observation(
        text=observation.text,
        images=(),
        metadata=deepcopy(observation.metadata),
    )


def _step_result_transport_copy(
    *,
    step_result: StepResult,
    turn: PreparedTurn | None,
) -> StepResult:
    """Return the step-result transport payload for one async result."""
    if turn is None or not step_result.observation.images:
        return step_result
    return StepResult(
        observation=_observation_transport_copy(
            observation=step_result.observation,
            turn=turn,
        ),
        reward=step_result.reward,
        accepted=step_result.accepted,
        terminated=step_result.terminated,
        truncated=step_result.truncated,
        info=deepcopy(step_result.info),
    )


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
                    observation = _observation_transport_copy(
                        observation=observation,
                        turn=turn,
                    )
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
                    step_result = _step_result_transport_copy(
                        step_result=step_result,
                        turn=turn,
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
        self._buffered_results: deque[AsyncResetResult | AsyncStepResult] = deque()
        self._buffered_exceptions: deque[BaseException] = deque()

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
        """Return slot ids whose most recent command is still in flight."""
        return tuple(sorted(self._busy_slot_ids))

    def reset(self, *, slot_id: int, seed: int) -> None:
        """Enqueue a reset command for one slot and return immediately."""
        self._dispatch(slot_id=slot_id, command=_ResetCommand(seed=seed))

    def reset_all(self, *, seeds: Sequence[int]) -> None:
        """Enqueue one reset command per slot."""
        if len(seeds) != self.slot_count:
            raise ValueError("reset_all() requires exactly one seed per worker slot.")
        for slot_id, seed in enumerate(seeds):
            self.reset(slot_id=slot_id, seed=seed)

    def step(self, *, slot_id: int, raw_action: str) -> None:
        """Enqueue one step command for a slot and return immediately."""
        self._dispatch(
            slot_id=slot_id,
            command=_StepCommand(raw_action=raw_action),
        )

    def recv(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncResetResult | AsyncStepResult:
        """Wait for one slot result and return it."""
        results = self.recv_ready(max_results=1, timeout_seconds=timeout_seconds)
        return results[0]

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

        buffered_results = self._pop_buffered_results(max_results=max_results)
        if buffered_results:
            return buffered_results
        if self._buffered_exceptions:
            raise self._buffered_exceptions.popleft()
        if not self._busy_slot_ids:
            raise RuntimeError("AsyncEnvPool has no pending commands to receive.")

        ready_connections = cast(
            list[Connection],
            wait(
                [self._connections_by_slot[slot_id] for slot_id in self._busy_slot_ids],
                timeout=timeout_seconds,
            ),
        )
        if not ready_connections:
            raise TimeoutError("Timed out waiting for async environment results.")

        if max_results is not None:
            ready_connections = ready_connections[:max_results]

        for connection in ready_connections:
            self._buffer_response(connection=connection)

        buffered_results = self._pop_buffered_results(max_results=max_results)
        if buffered_results:
            return buffered_results
        if self._buffered_exceptions:
            raise self._buffered_exceptions.popleft()
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
        self._buffered_results.clear()
        self._buffered_exceptions.clear()

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

    def _dispatch(self, *, slot_id: int, command: _ResetCommand | _StepCommand) -> None:
        """Send one command to an idle slot."""
        self._ensure_open()
        connection = self._connection_for_slot(slot_id=slot_id)
        if slot_id in self._busy_slot_ids:
            raise RuntimeError(f"Async env slot {slot_id} already has a pending task.")
        try:
            connection.send(command)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"Async env worker for slot {slot_id} is not available."
            ) from exc
        self._busy_slot_ids.add(slot_id)

    def _buffer_response(
        self,
        *,
        connection: Connection,
    ) -> None:
        """Receive one worker response and buffer its translated outcome."""
        fileno = connection.fileno()
        slot_id = self._slot_ids_by_fileno[fileno]
        self._busy_slot_ids.remove(slot_id)

        try:
            response = connection.recv()
        except EOFError:
            process = self._processes_by_slot[slot_id]
            self._buffered_exceptions.append(
                RuntimeError(
                    "Async env worker exited while a command was in flight "
                    f"(slot {slot_id}, exitcode={process.exitcode})."
                )
            )
            return

        if isinstance(response, _WorkerException):
            self._buffered_exceptions.append(
                self._materialize_worker_exception(
                    slot_id=slot_id,
                    response=response,
                )
            )
            return
        if isinstance(response, _WorkerResetResult):
            self._buffered_results.append(
                AsyncResetResult(
                    slot_id=slot_id,
                    episode_index=response.episode_index,
                    observation=response.observation,
                    reset_info=response.reset_info,
                    turn=response.turn,
                )
            )
            return
        if isinstance(response, _WorkerStepResult):
            self._buffered_results.append(
                AsyncStepResult(
                    slot_id=slot_id,
                    episode_index=response.episode_index,
                    step_result=response.step_result,
                    turn=response.turn,
                )
            )
            return
        self._buffered_exceptions.append(
            RuntimeError(f"Async env worker {slot_id} returned an unknown response.")
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
    ) -> tuple[AsyncResetResult | AsyncStepResult, ...]:
        """Pop and return up to `max_results` buffered successful results."""
        if not self._buffered_results:
            return ()

        if max_results is None:
            max_results = len(self._buffered_results)

        results: list[AsyncResetResult | AsyncStepResult] = []
        while self._buffered_results and len(results) < max_results:
            results.append(self._buffered_results.popleft())
        return tuple(results)

    def _connection_for_slot(self, *, slot_id: int) -> Connection:
        """Return the parent connection for a validated slot id."""
        connection = self._connections_by_slot.get(slot_id)
        if connection is None:
            raise IndexError(f"Async env slot {slot_id} does not exist.")
        return connection

    def _ensure_open(self) -> None:
        """Raise if the pool has already been closed."""
        if self._closed:
            raise RuntimeError("AsyncEnvPool has already been closed.")


__all__ = [
    "AsyncEnvPool",
    "AsyncResetResult",
    "AsyncStepResult",
]
