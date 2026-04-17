"""Shared TaskSpec runtime types and Pydantic parsing helpers."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)

TASK_SPEC_SCHEMA_VERSION = 1
NumericScalar = StrictInt | StrictFloat


@dataclass(slots=True, kw_only=True)
class TaskSpec:
    """Base class shared by all game-specific task specifications.

    Attributes
    ----------
    schema_version : int
        Task-spec schema version used to validate the YAML structure.
    task_id : str
        Stable identifier for the authored task specification.
    episode_config : EpisodeConfig
        Episode execution policy implied by the YAML configuration.
    metadata : dict[str, object]
        Free-form JSON-like bookkeeping metadata carried by the task spec.
    """

    schema_version: int
    task_id: str
    episode_config: EpisodeConfig = field(default_factory=EpisodeConfig)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def game(self) -> str:
        """Return the canonical game name for the task specification."""
        raise NotImplementedError

    def __post_init__(self) -> None:
        """Validate base task-spec fields."""
        if self.schema_version != TASK_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported task-spec schema version: "
                f"{self.schema_version}. Expected {TASK_SPEC_SCHEMA_VERSION}."
            )
        if not self.task_id:
            raise ValueError("Task specifications require a non-empty id.")
        if not isinstance(self.metadata, dict):
            raise TypeError("Task specification metadata must be a dict.")
        self.metadata = _snapshot_json_like(self.metadata, context="metadata")


class InvalidActionPolicyModel(BaseModel):
    """Pydantic representation of one invalid-action policy block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: InvalidActionMode
    penalty: NumericScalar | None = None

    def to_runtime(self) -> InvalidActionPolicy:
        """Convert the authored policy into the runtime dataclass."""
        penalty = None if self.penalty is None else float(self.penalty)
        return InvalidActionPolicy(mode=self.mode, penalty=penalty)


class EpisodeConfigModel(BaseModel):
    """Pydantic representation of the shared episode block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: StrictInt | None = None
    max_transitions: StrictInt | None = None
    invalid_action: InvalidActionPolicyModel | None = None

    def to_runtime(self) -> EpisodeConfig:
        """Convert the authored episode block into the runtime dataclass."""
        invalid_action = self.invalid_action
        if invalid_action is None:
            invalid_action_policy = InvalidActionPolicy(
                mode=InvalidActionMode.RAISE,
                penalty=None,
            )
        else:
            invalid_action_policy = invalid_action.to_runtime()
        return EpisodeConfig(
            max_attempts=self.max_attempts,
            max_transitions=self.max_transitions,
            invalid_action_policy=invalid_action_policy,
        )


class TaskSpecModel(BaseModel):
    """Shared Pydantic task-spec fields loaded directly from authored YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    task_id: StrictStr = Field(alias="id")
    episode: EpisodeConfigModel | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: object) -> dict[str, object]:
        """Validate and detach free-form metadata."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("Task specification metadata must be a dict.")
        snapshot = _snapshot_json_like(value, context="metadata")
        if not isinstance(snapshot, dict):
            raise TypeError("Task specification metadata must be a dict.")
        return snapshot

    @model_validator(mode="after")
    def validate_common_fields(self) -> Self:
        """Validate shared top-level task-spec invariants."""
        if self.schema_version != TASK_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported task-spec schema version: "
                f"{self.schema_version}. Expected {TASK_SPEC_SCHEMA_VERSION}."
            )
        if not self.task_id:
            raise ValueError("Task specifications require a non-empty id.")
        return self

    def episode_config(self) -> EpisodeConfig:
        """Return the runtime episode config, defaulting missing/null blocks."""
        episode = self.episode
        if episode is None:
            return EpisodeConfig()
        return episode.to_runtime()


ModelT = TypeVar("ModelT", bound=TaskSpecModel)


def validate_task_spec_model(
    *,
    model_type: type[ModelT],
    payload: Mapping[str, object],
    base_dir: Path,
) -> ModelT:
    """Validate one authored mapping against a Pydantic task-spec model."""
    try:
        return model_type.model_validate(payload, context={"base_dir": base_dir})
    except ValidationError as exc:
        translated_error = _translate_extra_forbidden_error(exc)
        if translated_error is not None:
            raise ValueError(translated_error) from exc
        raise


def resolve_task_spec_path(
    *,
    raw_path: str,
    base_dir: Path,
) -> Path:
    """Resolve one YAML path field relative to the task-spec directory."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_path_from_context(
    *,
    raw_path: Path | str,
    info: ValidationInfo,
    context: str,
) -> Path:
    """Resolve one potentially-relative path using validation context."""
    if isinstance(raw_path, str) and not raw_path:
        raise ValueError(f"{context} must be non-empty.")
    base_dir = _require_base_dir(info=info, context=context)
    return resolve_task_spec_path(raw_path=str(raw_path), base_dir=base_dir)


def _require_base_dir(*, info: ValidationInfo, context: str) -> Path:
    validation_context = info.context
    if not isinstance(validation_context, dict):
        raise RuntimeError(f"{context} validation requires a base_dir context.")
    base_dir = validation_context.get("base_dir")
    if not isinstance(base_dir, Path):
        raise RuntimeError(f"{context} validation requires a Path base_dir context.")
    return base_dir


def _snapshot_json_like(
    value: object,
    *,
    context: str,
    active_container_ids: set[int] | None = None,
) -> Any:
    """Return a detached JSON-like snapshot of ``value``."""
    if active_container_ids is None:
        active_container_ids = set()

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError(f"{context} must not contain cycles.")
        active_container_ids.add(container_id)
        try:
            return [
                _snapshot_json_like(
                    item,
                    context=context,
                    active_container_ids=active_container_ids,
                )
                for item in value
            ]
        finally:
            active_container_ids.remove(container_id)
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError(f"{context} must not contain cycles.")
        active_container_ids.add(container_id)
        snapshot: dict[str, object] = {}
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{context} keys must be strings.")
                snapshot[key] = _snapshot_json_like(
                    item,
                    context=f"{context}.{key}",
                    active_container_ids=active_container_ids,
                )
            return snapshot
        finally:
            active_container_ids.remove(container_id)
    raise TypeError(
        f"{context} must contain only JSON-like scalar, list, and dict values."
    )


def _translate_extra_forbidden_error(exc: ValidationError) -> str | None:
    errors = exc.errors()
    if not errors:
        return None
    extra_forbidden_errors = [
        error for error in errors if error["type"] == "extra_forbidden"
    ]
    if not extra_forbidden_errors:
        return None
    paths = ", ".join(
        ".".join(str(part) for part in error["loc"]) for error in extra_forbidden_errors
    )
    return f"Task specification contains unsupported fields: {paths}."


__all__ = [
    "EpisodeConfigModel",
    "InvalidActionPolicyModel",
    "NumericScalar",
    "TASK_SPEC_SCHEMA_VERSION",
    "TaskSpec",
    "TaskSpecModel",
    "resolve_path_from_context",
    "resolve_task_spec_path",
    "validate_task_spec_model",
]
