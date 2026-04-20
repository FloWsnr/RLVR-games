"""Generic prompt-row and reward helpers for trainer adapters."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rlvr_physics.core.instances import (
    TaskInstance,
    freeze_mapping,
    mapping_to_dict,
)
from rlvr_physics.core.session import TaskSession, TaskSubmission


SessionFactory = Callable[[TaskInstance], TaskSession]


@dataclass(frozen=True)
class PromptDatasetRow:
    """Trainer-safe prompt row for one immutable task instance.

    Parameters
    ----------
    task_id:
        Stable task instance identifier.
    task_kind:
        Versioned task kind.
    domain:
        Broad task domain or ability.
    prompt:
        Model-facing prompt text.
    renderer:
        Renderer that produced the prompt.
    metadata:
        Public task and rendering metadata.
    reward_model:
        Public reward model metadata.
    extra_info:
        Trainer-specific public metadata used for filtering or reward lookup.
    """

    task_id: str
    task_kind: str
    domain: str
    prompt: str
    renderer: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    reward_model: Mapping[str, object] = field(default_factory=dict)
    extra_info: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze nested metadata after construction."""

        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "reward_model", freeze_mapping(self.reward_model))
        object.__setattr__(self, "extra_info", freeze_mapping(self.extra_info))

    def as_dict(self) -> dict[str, Any]:
        """Return this row as plain trainer-safe containers."""

        return {
            "id": self.task_id,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "domain": self.domain,
            "prompt": self.prompt,
            "renderer": self.renderer,
            "metadata": mapping_to_dict(self.metadata),
            "reward_model": mapping_to_dict(self.reward_model),
            "extra_info": mapping_to_dict(self.extra_info),
        }


@dataclass(frozen=True)
class ScalarAdapterScore:
    """Public reward result returned by adapter scoring helpers.

    Parameters
    ----------
    task_id:
        Stable task instance identifier.
    accepted:
        Whether the completion was evaluable by the task.
    reward:
        Trainer-facing scalar reward.
    score:
        Optional domain score.
    done:
        Whether the scalar session ended.
    public_info:
        Trainer-safe result metadata.
    debug_info:
        Local debug metadata emitted by the task.
    """

    task_id: str
    accepted: bool
    reward: float
    score: float | None
    done: bool
    public_info: Mapping[str, object]
    debug_info: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze nested score metadata after construction."""

        object.__setattr__(self, "public_info", freeze_mapping(self.public_info))
        object.__setattr__(self, "debug_info", freeze_mapping(self.debug_info))

    def as_dict(self) -> dict[str, Any]:
        """Return this score as plain containers."""

        return {
            "task_id": self.task_id,
            "accepted": self.accepted,
            "reward": self.reward,
            "score": self.score,
            "done": self.done,
            "public_info": mapping_to_dict(self.public_info),
            "debug_info": mapping_to_dict(self.debug_info),
        }


def make_instance_registry(
    instances: Sequence[TaskInstance],
) -> dict[str, TaskInstance]:
    """Return an immutable-task lookup keyed by task id.

    Parameters
    ----------
    instances:
        Task instances to make available to reward adapters.
    """

    registry: dict[str, TaskInstance] = {}
    for instance in instances:
        if instance.task_id in registry:
            raise ValueError(f"duplicate task id: {instance.task_id}")
        registry[instance.task_id] = instance
    return registry


def make_prompt_dataset_row(
    instance: TaskInstance,
    session_factory: SessionFactory,
    seed: int,
    extra_info: Mapping[str, object],
) -> PromptDatasetRow:
    """Render one task instance into a trainer-safe prompt row.

    Parameters
    ----------
    instance:
        Immutable task instance to render.
    session_factory:
        Factory that creates one scalar task session for ``instance``.
    seed:
        Deterministic session seed used for rendering.
    extra_info:
        Public metadata to attach to the row.
    """

    session = session_factory(instance)
    reset = session.reset(seed=seed)
    prompt = reset.turn.observation.text()
    if not prompt:
        raise ValueError("prompt row requires a text observation")

    merged_extra = mapping_to_dict(instance.metadata)
    merged_extra.update(mapping_to_dict(extra_info))
    merged_extra["task_id"] = instance.task_id
    merged_extra["task_kind"] = instance.kind
    merged_extra["domain"] = instance.domain
    merged_extra["seed"] = instance.seed

    metadata = {
        "instance": instance.public_view(),
        "turn": {
            "index": reset.turn.turn_index,
            "submission_modes": reset.turn.submission_modes,
            "public_info": reset.turn.public_info,
            "public_limits": reset.turn.public_limits,
        },
        "observation": {
            "renderer": reset.turn.observation.renderer_name,
            "content_digests": reset.turn.observation.content_digests(),
        },
    }
    reward_model = {
        "style": "rlvr_executable",
        "task_id": instance.task_id,
    }
    return PromptDatasetRow(
        task_id=instance.task_id,
        task_kind=instance.kind,
        domain=instance.domain,
        prompt=prompt,
        renderer=reset.turn.observation.renderer_name,
        metadata=metadata,
        reward_model=reward_model,
        extra_info=merged_extra,
    )


def score_final_text(
    instance: TaskInstance,
    completion: object,
    session_factory: SessionFactory,
    seed: int,
) -> ScalarAdapterScore:
    """Score one completion by running a fresh scalar task session.

    Parameters
    ----------
    instance:
        Immutable task instance to score against.
    completion:
        Trainer completion payload.
    session_factory:
        Factory that creates one scalar task session for ``instance``.
    seed:
        Deterministic session seed used for scoring.
    """

    session = session_factory(instance)
    session.reset(seed=seed)
    text = completion_to_text(completion)
    result = session.submit(TaskSubmission.final_text(text))
    return ScalarAdapterScore(
        task_id=instance.task_id,
        accepted=result.accepted,
        reward=result.reward,
        score=result.score,
        done=result.done,
        public_info=result.public_info,
        debug_info=result.debug_info,
    )


def completion_to_text(completion: object) -> str:
    """Convert common trainer completion payloads to final-answer text.

    Parameters
    ----------
    completion:
        Completion payload from a trainer. Strings, chat-message mappings, and
        sequences of chat-message mappings are supported.
    """

    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return _mapping_content_to_text(completion)
    if isinstance(completion, Sequence) and not isinstance(completion, bytes):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, Mapping):
                parts.append(_mapping_content_to_text(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(completion)


def _mapping_content_to_text(value: Mapping[object, object]) -> str:
    content = value.get("content")
    if isinstance(content, str):
        return content
    text = value.get("text")
    if isinstance(text, str):
        return text
    return str(value)
