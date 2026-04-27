"""Immutable task instance types."""

from dataclasses import dataclass, field
from typing import Mapping

from rlvr_physics.core.payloads import freeze_mapping, stable_hash


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

    Attributes
    ----------
    max_turns:
        Maximum number of model submissions accepted before truncation.
    timeout_seconds:
        Optional wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    action_budget:
        Optional action or tool-call budget hint for stateful tasks.
    """

    max_turns: int
    timeout_seconds: float | None = None
    token_budget: int | None = None
    action_budget: int | None = None

    def as_public_dict(self) -> Mapping[str, object]:
        """Return trainer-safe limit metadata.

        Returns
        -------
        Mapping[str, object]
            Frozen mapping containing ``max_turns`` and any optional limit
            fields that are not ``None``.
        """

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

    Attributes
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
        Frozen data that may be rendered to the model.
    privileged_payload:
        Frozen verifier-only data that must not leak through public metadata.
    limits:
        Rollout limits for sessions created from this instance.
    metadata:
        Frozen public export and curriculum metadata.
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
        """Return metadata that can be safely exposed to a trainer.

        Returns
        -------
        Mapping[str, object]
            Frozen mapping containing public identity, limits, metadata, and
            public payload fields. Privileged payload data is excluded.
        """

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
        """Return a deterministic hash of the complete task instance.

        Returns
        -------
        str
            SHA-256 hex digest computed from the task identity, public payload,
            privileged payload, public limits, and metadata.

        Raises
        ------
        TypeError
            Raised if any instance data cannot be converted or JSON-encoded by
            the stable hashing helper.
        """

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
