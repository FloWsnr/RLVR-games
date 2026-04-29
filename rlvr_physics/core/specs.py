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
    budget_limits:
        Named public budget limits for task-specific interactions.
    timeout_seconds:
        Optional wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    metadata:
        Public export and curriculum hints.
    """

    kind: str
    domain: str
    source: SourceSpec
    renderers: tuple[RendererSpec, ...]
    verifier: VerifierSpec
    reward: RewardSpec
    budget_limits: Mapping[str, int]
    timeout_seconds: float | None = None
    token_budget: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata and budget limits after construction."""

        _validate_budget_limits(self.budget_limits)
        object.__setattr__(self, "budget_limits", freeze_mapping(self.budget_limits))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


def _validate_budget_limits(budget_limits: Mapping[str, int]) -> None:
    """Validate named public task budget limits."""

    for name, amount in budget_limits.items():
        if not isinstance(name, str) or name == "":
            raise ValueError("budget limit name must be a non-empty string")
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError(f"budget limit must be an integer: {name}")
        if amount < 0:
            raise ValueError(f"budget limit must be non-negative: {name}")
