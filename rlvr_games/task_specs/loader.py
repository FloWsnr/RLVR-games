"""YAML-backed task specifications for RLVR environments."""

from pathlib import Path
from typing import Any

import yaml

from rlvr_games.core.protocol import Environment
from rlvr_games.core.task_spec_base import (
    TASK_SPEC_SCHEMA_VERSION,
    TaskSpec,
    TaskSpecHeader,
    optional_bool,
    optional_float,
    optional_int,
    optional_mapping,
    optional_path,
    optional_string,
    parse_episode_config,
    parse_metadata,
    parse_task_spec_header,
    reject_unknown_keys,
    required_float,
    required_int,
    required_string,
    required_string_sequence,
    require_mapping,
    require_nested_sequence,
    require_string_sequence,
    resolve_task_spec_path,
)
from rlvr_games.task_specs.registry import get_task_spec_handler


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
    mapping = require_mapping(payload, context="task specification")
    game = required_string(mapping, "game", context="task specification")
    return get_task_spec_handler(game=game).parse_mapping(
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
    "TaskSpecHeader",
    "build_environment_from_task_spec",
    "load_environment_from_task_spec_path",
    "load_task_spec",
    "optional_bool",
    "optional_float",
    "optional_int",
    "optional_mapping",
    "optional_path",
    "optional_string",
    "parse_episode_config",
    "parse_metadata",
    "parse_task_spec_header",
    "reject_unknown_keys",
    "required_float",
    "required_int",
    "required_string",
    "required_string_sequence",
    "require_mapping",
    "require_nested_sequence",
    "require_string_sequence",
    "resolve_task_spec_path",
    "task_spec_from_mapping",
]
