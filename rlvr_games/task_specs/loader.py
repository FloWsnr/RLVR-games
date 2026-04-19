"""YAML-backed task specifications for executable RLVR tasks."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator
import yaml

from rlvr_games.core.protocol import Environment
from rlvr_games.core.session import TaskSessionProtocol
from rlvr_games.core.task_spec_base import TASK_SPEC_SCHEMA_VERSION, TaskSpec
from rlvr_games.task_specs.registry import (
    TaskSessionFactory,
    build_environment_from_registered_task_spec,
    build_task_session_factory_from_registered_task_spec,
    get_task_spec_handler,
)


class _TaskSpecDispatchModel(BaseModel):
    """Minimal model used to route authored mappings to one task parser."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: StrictStr | None = None
    game: StrictStr | None = None

    @model_validator(mode="after")
    def validate_dispatch_fields(self) -> "_TaskSpecDispatchModel":
        """Validate neutral and legacy dispatch fields."""
        if self.kind is None and self.game is None:
            raise ValueError(
                "Task specification requires a non-empty 'kind' field, or legacy "
                "'game' field."
            )
        if self.kind == "":
            raise ValueError("Task specification field 'kind' must be non-empty.")
        if self.game == "":
            raise ValueError("Task specification field 'game' must be non-empty.")
        if self.kind is not None and self.game is not None and self.kind != self.game:
            raise ValueError(
                "Task specification fields 'kind' and 'game' must match when both "
                "are provided."
            )
        return self

    @property
    def task_kind(self) -> str:
        """Return the neutral task kind used for registry dispatch."""
        if self.kind is not None:
            return self.kind
        if self.game is not None:
            return self.game
        raise RuntimeError("Validated dispatch model has no kind or game.")


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
    handler = get_task_spec_handler(kind=dispatch.task_kind)
    return handler.parse_mapping(
        payload=_payload_for_handler(
            mapping=mapping,
            task_kind=dispatch.task_kind,
            handler_uses_legacy_game_field=handler.uses_legacy_game_field,
        ),
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
    return build_environment_from_registered_task_spec(task_spec=task_spec)


def build_task_session_factory_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TaskSessionFactory:
    """Construct a fresh scalar task-session factory from one task spec.

    Parameters
    ----------
    task_spec : TaskSpec
        Parsed task specification to materialize.

    Returns
    -------
    TaskSessionFactory
        Picklable factory that returns a new mutable task session on each call.
    """
    return build_task_session_factory_from_registered_task_spec(task_spec=task_spec)


def build_task_session_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TaskSessionProtocol:
    """Construct one scalar task session from a validated task specification."""
    session_factory = build_task_session_factory_from_task_spec(task_spec=task_spec)
    return session_factory()


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


def load_task_session_factory_from_task_spec_path(
    *,
    path: Path,
) -> TaskSessionFactory:
    """Load a YAML task spec and build a fresh scalar-session factory."""
    task_spec = load_task_spec(path=path)
    return build_task_session_factory_from_task_spec(task_spec=task_spec)


def load_task_session_from_task_spec_path(
    *,
    path: Path,
) -> TaskSessionProtocol:
    """Load a YAML task spec and immediately build one scalar task session."""
    session_factory = load_task_session_factory_from_task_spec_path(path=path)
    return session_factory()


def _payload_for_handler(
    *,
    mapping: dict[str, object],
    task_kind: str,
    handler_uses_legacy_game_field: bool,
) -> dict[str, object]:
    """Return a parser payload with dispatch fields normalized."""
    payload = dict(mapping)
    if handler_uses_legacy_game_field:
        payload["game"] = task_kind
        payload.pop("kind", None)
    return payload


__all__ = [
    "TASK_SPEC_SCHEMA_VERSION",
    "TaskSpec",
    "build_environment_from_task_spec",
    "build_task_session_factory_from_task_spec",
    "build_task_session_from_task_spec",
    "load_environment_from_task_spec_path",
    "load_task_session_factory_from_task_spec_path",
    "load_task_session_from_task_spec_path",
    "load_task_spec",
    "task_spec_from_mapping",
]
