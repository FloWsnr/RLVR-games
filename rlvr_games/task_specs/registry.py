"""Task-spec registry for parsing and task-session construction."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from rlvr_games.core.protocol import Environment
from rlvr_games.core.session import EnvironmentTaskSession, TaskSessionProtocol
from rlvr_games.core.task_spec_base import TaskSpec
from rlvr_games.games.chess.task_spec import (
    build_chess_environment_from_task_spec,
    chess_task_spec_from_mapping,
)
from rlvr_games.games.connect4.task_spec import (
    build_connect4_environment_from_task_spec,
    connect4_task_spec_from_mapping,
)
from rlvr_games.games.game2048.task_spec import (
    build_game2048_environment_from_task_spec,
    game2048_task_spec_from_mapping,
)
from rlvr_games.games.minesweeper.task_spec import (
    build_minesweeper_environment_from_task_spec,
    minesweeper_task_spec_from_mapping,
)
from rlvr_games.games.mastermind.task_spec import (
    build_mastermind_environment_from_task_spec,
    mastermind_task_spec_from_mapping,
)
from rlvr_games.games.yahtzee.task_spec import (
    build_yahtzee_environment_from_task_spec,
    yahtzee_task_spec_from_mapping,
)
from rlvr_games.tasks.arithmetic.task_spec import (
    arithmetic_task_spec_from_mapping,
    build_arithmetic_session_factory_from_task_spec,
)


class TaskSpecMappingParser(Protocol):
    """Callable protocol for parsing one authored task spec."""

    def __call__(self, *, payload: dict[str, object], base_dir: Path) -> TaskSpec:
        """Parse one task spec from a mapping."""
        ...


class TaskSpecEnvironmentBuilder(Protocol):
    """Callable protocol for building one environment from a task spec."""

    def __call__(self, *, task_spec: TaskSpec) -> Environment[Any, Any]:
        """Build one environment from a validated task spec."""
        ...


class TaskSpecSessionFactoryBuilder(Protocol):
    """Callable protocol for building a scalar task-session factory."""

    def __call__(self, *, task_spec: TaskSpec) -> "TaskSessionFactory":
        """Build one task-session factory from a validated task spec."""
        ...


class TaskSessionFactory(Protocol):
    """Callable protocol for constructing a fresh scalar task session."""

    def __call__(self) -> TaskSessionProtocol:
        """Build one mutable task session."""
        ...


@dataclass(slots=True, frozen=True)
class TaskSpecHandler:
    """Parser and builder callbacks for one task-spec kind."""

    parse_mapping: TaskSpecMappingParser
    build_environment: TaskSpecEnvironmentBuilder | None = None
    build_session_factory: TaskSpecSessionFactoryBuilder | None = None
    uses_legacy_game_field: bool = False


_TASK_SPEC_HANDLERS: dict[str, TaskSpecHandler] = {
    "chess": TaskSpecHandler(
        parse_mapping=chess_task_spec_from_mapping,
        build_environment=build_chess_environment_from_task_spec,
        uses_legacy_game_field=True,
    ),
    "connect4": TaskSpecHandler(
        parse_mapping=connect4_task_spec_from_mapping,
        build_environment=build_connect4_environment_from_task_spec,
        uses_legacy_game_field=True,
    ),
    "game2048": TaskSpecHandler(
        parse_mapping=game2048_task_spec_from_mapping,
        build_environment=build_game2048_environment_from_task_spec,
        uses_legacy_game_field=True,
    ),
    "minesweeper": TaskSpecHandler(
        parse_mapping=minesweeper_task_spec_from_mapping,
        build_environment=build_minesweeper_environment_from_task_spec,
        uses_legacy_game_field=True,
    ),
    "mastermind": TaskSpecHandler(
        parse_mapping=mastermind_task_spec_from_mapping,
        build_environment=build_mastermind_environment_from_task_spec,
        uses_legacy_game_field=True,
    ),
    "yahtzee": TaskSpecHandler(
        parse_mapping=yahtzee_task_spec_from_mapping,
        build_environment=build_yahtzee_environment_from_task_spec,
        uses_legacy_game_field=True,
    ),
    "arithmetic": TaskSpecHandler(
        parse_mapping=arithmetic_task_spec_from_mapping,
        build_session_factory=build_arithmetic_session_factory_from_task_spec,
    ),
}


def build_environment_task_session_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TaskSessionProtocol:
    """Build an environment-backed task session from one task spec."""
    environment = build_environment_from_registered_task_spec(task_spec=task_spec)
    return EnvironmentTaskSession(env=environment, task_kind=task_spec.kind)


def build_task_session_factory_from_registered_task_spec(
    *,
    task_spec: TaskSpec,
) -> TaskSessionFactory:
    """Build a fresh-session factory from one validated task spec."""
    handler = get_task_spec_handler(kind=task_spec.kind)
    if handler.build_session_factory is not None:
        return handler.build_session_factory(task_spec=task_spec)
    if handler.build_environment is not None:
        return partial(
            build_environment_task_session_from_task_spec,
            task_spec=task_spec,
        )
    raise ValueError(f"Task spec kind {task_spec.kind!r} cannot build sessions.")


def build_environment_from_registered_task_spec(
    *,
    task_spec: TaskSpec,
) -> Environment[Any, Any]:
    """Construct an environment from one validated task specification."""
    handler = get_task_spec_handler(kind=task_spec.kind)
    if handler.build_environment is None:
        raise ValueError(
            f"Task spec kind {task_spec.kind!r} is not environment-backed."
        )
    return handler.build_environment(task_spec=task_spec)


def get_task_spec_handler(
    *,
    kind: str | None = None,
    game: str | None = None,
) -> TaskSpecHandler:
    """Return the registered parser/builder pair for one task kind."""
    task_kind = kind if kind is not None else game
    if task_kind is None:
        raise ValueError("Task-spec handler lookup requires a kind.")
    handler = _TASK_SPEC_HANDLERS.get(task_kind)
    if handler is None:
        raise ValueError(f"Unsupported task-spec kind: {task_kind!r}.")
    return handler


__all__ = [
    "TaskSessionFactory",
    "TaskSpecHandler",
    "build_environment_from_registered_task_spec",
    "build_environment_task_session_from_task_spec",
    "build_task_session_factory_from_registered_task_spec",
    "get_task_spec_handler",
]
