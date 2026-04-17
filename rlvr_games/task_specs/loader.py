"""YAML-backed task specifications for RLVR environments."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator
import yaml

from rlvr_games.core.protocol import Environment
from rlvr_games.core.task_spec_base import TASK_SPEC_SCHEMA_VERSION, TaskSpec
from rlvr_games.task_specs.registry import get_task_spec_handler


class _TaskSpecDispatchModel(BaseModel):
    """Minimal model used to route authored mappings to one game parser."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    game: StrictStr

    @field_validator("game")
    @classmethod
    def validate_game(cls, value: str) -> str:
        """Validate that the authored game name is non-empty."""
        if not value:
            raise ValueError("Task specification field 'game' must be non-empty.")
        return value


def load_task_spec(*, path: Path) -> TaskSpec:
    """Load one task specification from a YAML file.

    Parameters
    ----------
    path : Path
        YAML file path to read.

    Returns
    -------
    TaskSpec
        Parsed game-specific task specification.
    """
    resolved_path = path.expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return task_spec_from_mapping(payload=payload, base_dir=resolved_path.parent)


def task_spec_from_mapping(
    *,
    payload: object,
    base_dir: Path,
) -> TaskSpec:
    """Parse one task specification from an in-memory mapping.

    Parameters
    ----------
    payload : object
        Raw parsed YAML payload.
    base_dir : Path
        Directory used to resolve any relative paths embedded in the payload.

    Returns
    -------
    TaskSpec
        Parsed game-specific task specification.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("task specification must be a mapping.")
    mapping = dict(payload)
    dispatch = _TaskSpecDispatchModel.model_validate(mapping)
    return get_task_spec_handler(game=dispatch.game).parse_mapping(
        payload=mapping,
        base_dir=base_dir,
    )


def build_environment_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> Environment[Any, Any]:
    """Construct an environment from one validated task specification.

    Parameters
    ----------
    task_spec : TaskSpec
        Parsed task specification to materialize.

    Returns
    -------
    Environment[Any, Any]
        Fully wired environment implied by the task specification.
    """
    return get_task_spec_handler(game=task_spec.game).build_environment(
        task_spec=task_spec
    )


def load_environment_from_task_spec_path(
    *,
    path: Path,
) -> Environment[Any, Any]:
    """Load a YAML task spec and immediately build its environment.

    Parameters
    ----------
    path : Path
        YAML task-spec path to load.

    Returns
    -------
    Environment[Any, Any]
        Environment materialized from the YAML task specification.
    """
    task_spec = load_task_spec(path=path)
    return build_environment_from_task_spec(task_spec=task_spec)


__all__ = [
    "TASK_SPEC_SCHEMA_VERSION",
    "TaskSpec",
    "build_environment_from_task_spec",
    "load_environment_from_task_spec_path",
    "load_task_spec",
    "task_spec_from_mapping",
]
