"""Game registry for TaskSpec parsing and environment construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rlvr_games.core.protocol import Environment
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
from rlvr_games.games.yahtzee.task_spec import (
    build_yahtzee_environment_from_task_spec,
    yahtzee_task_spec_from_mapping,
)


class TaskSpecMappingParser(Protocol):
    """Callable protocol for parsing one game's authored task specs."""

    def __call__(self, *, payload: dict[str, object], base_dir: Path) -> TaskSpec:
        """Parse one game-specific task spec from a mapping."""
        ...


class TaskSpecEnvironmentBuilder(Protocol):
    """Callable protocol for building one game's environment from a task spec."""

    def __call__(self, *, task_spec: TaskSpec) -> Environment[Any, Any]:
        """Build one environment from a validated task spec."""
        ...


@dataclass(slots=True, frozen=True)
class TaskSpecHandler:
    """Parser and builder callbacks for one game's task specifications."""

    parse_mapping: TaskSpecMappingParser
    build_environment: TaskSpecEnvironmentBuilder


_TASK_SPEC_HANDLERS: dict[str, TaskSpecHandler] = {
    "chess": TaskSpecHandler(
        parse_mapping=chess_task_spec_from_mapping,
        build_environment=build_chess_environment_from_task_spec,
    ),
    "connect4": TaskSpecHandler(
        parse_mapping=connect4_task_spec_from_mapping,
        build_environment=build_connect4_environment_from_task_spec,
    ),
    "game2048": TaskSpecHandler(
        parse_mapping=game2048_task_spec_from_mapping,
        build_environment=build_game2048_environment_from_task_spec,
    ),
    "minesweeper": TaskSpecHandler(
        parse_mapping=minesweeper_task_spec_from_mapping,
        build_environment=build_minesweeper_environment_from_task_spec,
    ),
    "yahtzee": TaskSpecHandler(
        parse_mapping=yahtzee_task_spec_from_mapping,
        build_environment=build_yahtzee_environment_from_task_spec,
    ),
}


def get_task_spec_handler(*, game: str) -> TaskSpecHandler:
    """Return the registered parser/builder pair for one game."""
    handler = _TASK_SPEC_HANDLERS.get(game)
    if handler is None:
        raise ValueError(f"Unsupported task-spec game: {game!r}.")
    return handler


__all__ = ["TaskSpecHandler", "get_task_spec_handler"]
