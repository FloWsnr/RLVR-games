"""Task specification data structures."""

from dataclasses import dataclass, field
from typing import Mapping

from rlvr_physics.core.payloads import freeze_mapping


@dataclass(frozen=True)
class SourceSpec:
    """Reproducible source configuration for task instances.

    Attributes
    ----------
    source_type:
        Identifier for the instance source, such as a generator or dataset.
    seed:
        Deterministic seed passed to the source.
    parameters:
        Source-specific configuration payload. Values are recursively frozen
        after construction.
    """

    source_type: str
    seed: int
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class RendererSpec:
    """Renderer configuration advertised by a task.

    Attributes
    ----------
    renderer_type:
        Identifier for the renderer implementation.
    parameters:
        Renderer-specific configuration payload. Values are recursively frozen
        after construction.
    """

    renderer_type: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class VerifierSpec:
    """Verifier configuration advertised by a task.

    Attributes
    ----------
    verifier_type:
        Identifier for the verifier implementation or strategy.
    parameters:
        Verifier-specific configuration payload. Values are recursively frozen
        after construction.
    """

    verifier_type: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class RewardSpec:
    """Reward configuration advertised by a task.

    Attributes
    ----------
    reward_type:
        Identifier for the reward implementation or scoring policy.
    parameters:
        Reward-specific configuration payload. Values are recursively frozen
        after construction.
    """

    reward_type: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class TaskSpec:
    """Trainer-friendly task setup description.

    Attributes
    ----------
    kind:
        Versioned task kind.
    domain:
        Broad domain or ability label.
    source:
        Reproducible source configuration.
    renderers:
        Supported renderer configurations.
    verifier:
        Verifier behavior summary.
    reward:
        Reward behavior summary.
    max_turns:
        Default maximum number of model submissions accepted before
        truncation.
    timeout_seconds:
        Optional wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    action_budget:
        Optional action or tool-call budget hint for stateful tasks.
    metadata:
        Public export and curriculum hints.
    """

    kind: str
    domain: str
    source: SourceSpec
    renderers: tuple[RendererSpec, ...]
    verifier: VerifierSpec
    reward: RewardSpec
    max_turns: int
    timeout_seconds: float | None = None
    token_budget: int | None = None
    action_budget: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata after construction."""

        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
