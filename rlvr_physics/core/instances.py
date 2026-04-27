"""Immutable task instance types."""

from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class TaskLimits:
    """Public task limits that constrain one rollout.

    Parameters
    ----------
    max_turns:
        Maximum number of model submissions accepted before truncation.
    timeout_seconds:
        Wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Token budget hint for prompt/completion trainers.
    action_budget:
        Action or tool-call budget hint for stateful tasks.
    """

    max_turns: int
    timeout_seconds: float | None = None
    token_budget: int | None = None
    action_budget: int | None = None

    def as_public_dict(self) -> Mapping[str, object]:
        """Return trainer-safe limit metadata."""

        values: dict[str, object] = {"max_turns": self.max_turns}
        if self.timeout_seconds is not None:
            values["timeout_seconds"] = self.timeout_seconds
        if self.token_budget is not None:
            values["token_budget"] = self.token_budget
        if self.action_budget is not None:
            values["action_budget"] = self.action_budget
        return freeze_mapping(values)


@dataclass(frozen=True)
class TaskInstance:
    """Immutable payload sampled for one task.

    Parameters
    ----------
    task_id:
        Stable task identity for replay and joins with trainer records.
    kind:
        Versioned task kind, such as ``physics.numeric.v1``.
    domain:
        Broad domain or ability label.
    seed:
        Seed or source-specific deterministic identity.
    public_payload:
        Data that may be rendered to the model.
    privileged_payload:
        Verifier-only data that must not leak through public metadata.
    limits:
        Rollout limits for sessions created from this instance.
    metadata:
        Public export and curriculum metadata.
    """

    task_id: str
    kind: str
    domain: str
    seed: int
    public_payload: Mapping[str, object]
    privileged_payload: Mapping[str, object]
    limits: TaskLimits
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze mutable mapping and sequence payloads after construction."""

        object.__setattr__(self, "public_payload", freeze_mapping(self.public_payload))
        object.__setattr__(
            self, "privileged_payload", freeze_mapping(self.privileged_payload)
        )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def public_view(self) -> Mapping[str, object]:
        """Return metadata that can be safely exposed to a trainer."""

        return freeze_mapping(
            {
                "task_id": self.task_id,
                "kind": self.kind,
                "domain": self.domain,
                "seed": self.seed,
                "limits": self.limits.as_public_dict(),
                "metadata": self.metadata,
                "payload": self.public_payload,
            }
        )

    def content_hash(self) -> str:
        """Return a deterministic hash of the complete task instance."""

        return stable_hash(
            {
                "task_id": self.task_id,
                "kind": self.kind,
                "domain": self.domain,
                "seed": self.seed,
                "public_payload": self.public_payload,
                "privileged_payload": self.privileged_payload,
                "limits": self.limits.as_public_dict(),
                "metadata": self.metadata,
            }
        )


def freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Recursively freeze a mapping for dataclass payload storage."""

    frozen: dict[str, object] = {}
    for key, value in values.items():
        frozen[key] = freeze_value(value)
    return MappingProxyType(frozen)


def freeze_value(value: object) -> object:
    """Recursively freeze containers while preserving scalar values."""

    if isinstance(value, Mapping):
        string_keyed: dict[str, object] = {}
        for key, item in value.items():
            string_keyed[str(key)] = freeze_value(item)
        return MappingProxyType(string_keyed)
    if isinstance(value, list | tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(freeze_value(item) for item in sorted(value, key=repr))
    return value


def to_plain_data(value: object) -> object:
    """Convert frozen payload data into JSON-serializable containers."""

    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def stable_hash(value: object) -> str:
    """Return a SHA-256 hash for JSON-compatible task data."""

    encoded = json.dumps(
        to_plain_data(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def require_mapping(value: object, name: str) -> Mapping[str, object]:
    """Return ``value`` as a mapping or raise a task-shape error."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def require_tuple_of_ints(value: object, name: str) -> tuple[int, ...]:
    """Return ``value`` as a tuple of integers or raise a task-shape error."""

    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of integers")
    ints: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise TypeError(f"{name} must contain only integers")
        ints.append(item)
    return tuple(ints)


def require_int(value: object, name: str) -> int:
    """Return ``value`` as an integer or raise a task-shape error."""

    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def require_str(value: object, name: str) -> str:
    """Return ``value`` as a string or raise a task-shape error."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def require_bool(value: object, name: str) -> bool:
    """Return ``value`` as a boolean or raise a task-shape error."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def require_optional_str(value: object, name: str) -> str | None:
    """Return ``value`` as an optional string or raise a task-shape error."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def mapping_to_dict(values: Mapping[str, object]) -> dict[str, Any]:
    """Return a mutable plain dictionary from frozen payload data."""

    plain = to_plain_data(values)
    if not isinstance(plain, dict):
        raise TypeError("mapping conversion did not produce a dictionary")
    return plain
