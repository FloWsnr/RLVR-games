"""Shared TaskSpec types and parsing helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)

TASK_SPEC_SCHEMA_VERSION = 1


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


@dataclass(slots=True, frozen=True)
class TaskSpecHeader:
    """Common validated top-level fields extracted before game dispatch.

    Attributes
    ----------
    schema_version : int
        Task-spec schema version.
    task_id : str
        Stable authored task identifier.
    episode_config : EpisodeConfig
        Episode execution policy parsed from the YAML mapping.
    metadata : dict[str, object]
        Free-form JSON-like metadata snapshot.
    """

    schema_version: int
    task_id: str
    episode_config: EpisodeConfig
    metadata: dict[str, object]


def parse_task_spec_header(
    *,
    payload: Mapping[str, object],
    expected_game: str,
    allowed_top_level_keys: Sequence[str],
) -> TaskSpecHeader:
    """Validate shared top-level task-spec fields for one game.

    Parameters
    ----------
    payload : Mapping[str, object]
        Raw top-level task-spec mapping.
    expected_game : str
        Game name that the caller expects.
    allowed_top_level_keys : Sequence[str]
        Top-level fields allowed for the requested game.

    Returns
    -------
    TaskSpecHeader
        Parsed header fields shared across all task specifications.
    """
    reject_unknown_keys(
        payload,
        allowed_keys=allowed_top_level_keys,
        context=f"{expected_game} task specification",
    )
    schema_version = required_int(
        payload,
        "schema_version",
        context=f"{expected_game} task specification",
    )
    if schema_version != TASK_SPEC_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported task-spec schema version: "
            f"{schema_version}. Expected {TASK_SPEC_SCHEMA_VERSION}."
        )
    game = required_string(
        payload,
        "game",
        context=f"{expected_game} task specification",
    )
    if game != expected_game:
        raise ValueError(
            f"Task specification game must be {expected_game!r}, got {game!r}."
        )
    return TaskSpecHeader(
        schema_version=schema_version,
        task_id=required_string(
            payload,
            "id",
            context=f"{expected_game} task specification",
        ),
        episode_config=parse_episode_config(
            payload.get("episode"),
            context=f"{expected_game} task specification",
        ),
        metadata=parse_metadata(
            payload.get("metadata"),
            context=f"{expected_game} task specification",
        ),
    )


def parse_episode_config(
    payload: object,
    *,
    context: str,
) -> EpisodeConfig:
    """Parse the shared episode section of a task specification.

    Parameters
    ----------
    payload : object
        Raw episode mapping or ``None``.
    context : str
        Human-readable context used in error messages.

    Returns
    -------
    EpisodeConfig
        Parsed runtime episode configuration.
    """
    if payload is None:
        return EpisodeConfig()
    mapping = require_mapping(payload, context=f"{context} episode")
    reject_unknown_keys(
        mapping,
        allowed_keys=("max_attempts", "max_transitions", "invalid_action"),
        context=f"{context} episode",
    )
    invalid_action_payload = mapping.get("invalid_action")
    invalid_action_policy = InvalidActionPolicy(
        mode=InvalidActionMode.RAISE,
        penalty=None,
    )
    if invalid_action_payload is not None:
        invalid_action_mapping = require_mapping(
            invalid_action_payload,
            context=f"{context} invalid_action",
        )
        reject_unknown_keys(
            invalid_action_mapping,
            allowed_keys=("mode", "penalty"),
            context=f"{context} invalid_action",
        )
        mode = InvalidActionMode(
            required_string(
                invalid_action_mapping,
                "mode",
                context=f"{context} invalid_action",
            )
        )
        penalty = optional_float(
            invalid_action_mapping,
            "penalty",
            context=f"{context} invalid_action",
        )
        invalid_action_policy = InvalidActionPolicy(mode=mode, penalty=penalty)

    return EpisodeConfig(
        max_attempts=optional_int(
            mapping, "max_attempts", context=f"{context} episode"
        ),
        max_transitions=optional_int(
            mapping,
            "max_transitions",
            context=f"{context} episode",
        ),
        invalid_action_policy=invalid_action_policy,
    )


def parse_metadata(
    payload: object,
    *,
    context: str,
) -> dict[str, object]:
    """Parse the free-form metadata section of a task specification.

    Parameters
    ----------
    payload : object
        Raw metadata mapping or ``None``.
    context : str
        Human-readable context used in error messages.

    Returns
    -------
    dict[str, object]
        JSON-like metadata dictionary.
    """
    if payload is None:
        return {}
    mapping = require_mapping(payload, context=f"{context} metadata")
    return _snapshot_json_like(mapping, context=f"{context} metadata")


def require_mapping(
    payload: object,
    *,
    context: str,
) -> dict[str, object]:
    """Return ``payload`` as a string-keyed mapping or raise.

    Parameters
    ----------
    payload : object
        Raw value to validate.
    context : str
        Human-readable location used in error messages.

    Returns
    -------
    dict[str, object]
        Validated mapping with string keys.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    string_key_mapping: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings.")
        string_key_mapping[key] = value
    return string_key_mapping


def optional_mapping(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> dict[str, object] | None:
    """Return an optional nested mapping from a parent mapping.

    Parameters
    ----------
    payload : Mapping[str, object]
        Parent mapping.
    key : str
        Key whose value should be a mapping when present.
    context : str
        Human-readable location used in error messages.

    Returns
    -------
    dict[str, object] | None
        Nested mapping value or ``None`` when the key is absent.
    """
    raw_value = payload.get(key)
    if raw_value is None:
        return None
    return require_mapping(raw_value, context=f"{context} {key}")


def required_string(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    """Return a required non-empty string value from a mapping."""
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{context} field {key!r} must be a string.")
    if not value:
        raise ValueError(f"{context} field {key!r} must be non-empty.")
    return value


def optional_string(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str | None:
    """Return an optional string value from a mapping."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{context} field {key!r} must be a string.")
    if not value:
        raise ValueError(f"{context} field {key!r} must be non-empty when set.")
    return value


def required_int(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int:
    """Return a required integer value from a mapping."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} field {key!r} must be an int.")
    return value


def optional_int(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int | None:
    """Return an optional integer value from a mapping."""
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} field {key!r} must be an int.")
    return value


def required_float(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> float:
    """Return a required numeric value from a mapping."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context} field {key!r} must be numeric.")
    return float(value)


def optional_float(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> float | None:
    """Return an optional numeric value from a mapping."""
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context} field {key!r} must be numeric.")
    return float(value)


def optional_bool(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> bool | None:
    """Return an optional boolean value from a mapping."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{context} field {key!r} must be a bool.")
    return value


def optional_path(
    payload: Mapping[str, object],
    key: str,
    *,
    base_dir: Path,
    context: str,
) -> Path | None:
    """Return an optional path value resolved relative to ``base_dir``."""
    raw_value = optional_string(payload, key, context=context)
    if raw_value is None:
        return None
    return resolve_task_spec_path(raw_path=raw_value, base_dir=base_dir)


def required_string_sequence(
    payload: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> tuple[str, ...]:
    """Return a required string sequence from a mapping."""
    raw_value = payload.get(key)
    return require_string_sequence(raw_value, context=f"{context} field {key!r}")


def require_string_sequence(
    payload: object,
    *,
    context: str,
) -> tuple[str, ...]:
    """Return one sequence of strings from a raw value."""
    if isinstance(payload, str) or not isinstance(payload, Sequence):
        raise TypeError(f"{context} must be a sequence of strings.")
    values: list[str] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str):
            raise TypeError(f"{context} item {index} must be a string.")
        values.append(value)
    return tuple(values)


def require_nested_sequence(
    payload: object,
    *,
    context: str,
) -> tuple[tuple[object, ...], ...]:
    """Return one nested sequence as immutable tuples.

    Parameters
    ----------
    payload : object
        Raw nested sequence to validate.
    context : str
        Human-readable location used in error messages.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Immutable nested sequence snapshot.
    """
    if isinstance(payload, str) or not isinstance(payload, Sequence):
        raise TypeError(f"{context} must be a nested sequence.")
    rows: list[tuple[object, ...]] = []
    for row_index, row in enumerate(payload):
        if isinstance(row, str):
            rows.append(tuple(row))
            continue
        if isinstance(row, Sequence):
            rows.append(tuple(row))
            continue
        raise TypeError(f"{context} row {row_index} must be a sequence.")
    return tuple(rows)


def reject_unknown_keys(
    payload: Mapping[str, object],
    *,
    allowed_keys: Sequence[str],
    context: str,
) -> None:
    """Raise when ``payload`` contains keys outside ``allowed_keys``."""
    allowed = set(allowed_keys)
    unknown_keys = sorted(key for key in payload if key not in allowed)
    if unknown_keys:
        joined_unknown = ", ".join(repr(key) for key in unknown_keys)
        raise ValueError(f"{context} contains unsupported fields: {joined_unknown}.")


def resolve_task_spec_path(
    *,
    raw_path: str,
    base_dir: Path,
) -> Path:
    """Resolve one YAML path field relative to the task-spec directory.

    Parameters
    ----------
    raw_path : str
        Raw path text extracted from YAML.
    base_dir : Path
        Directory containing the authored YAML file.

    Returns
    -------
    Path
        Resolved absolute filesystem path.
    """
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _snapshot_json_like(
    value: object,
    *,
    context: str,
    active_container_ids: set[int] | None = None,
) -> Any:
    """Return a detached JSON-like snapshot of ``value``.

    Parameters
    ----------
    value : object
        Raw metadata value to validate.
    context : str
        Human-readable location used in error messages.
    active_container_ids : set[int] | None
        Internal recursion guard for cycle detection.

    Returns
    -------
    Any
        Detached JSON-like snapshot.
    """
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


__all__ = [
    "TASK_SPEC_SCHEMA_VERSION",
    "TaskSpec",
    "TaskSpecHeader",
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
]
