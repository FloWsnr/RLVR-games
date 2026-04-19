"""Task specification data structures."""

from dataclasses import dataclass, field
from typing import Mapping

from rlvr_physics.core.instances import TaskLimits, freeze_mapping


@dataclass(frozen=True)
class SourceSpec:
    """Reproducible source configuration for task instances."""

    source_type: str
    seed: int
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class RendererSpec:
    """Renderer configuration advertised by a task."""

    renderer_type: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class VerifierSpec:
    """Verifier configuration advertised by a task."""

    verifier_type: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class RewardSpec:
    """Reward configuration advertised by a task."""

    reward_type: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze parameter payloads after construction."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class TaskSpec:
    """Adapter-friendly task setup description.

    Parameters
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
    limits:
        Default rollout limits.
    metadata:
        Public export and curriculum hints.
    """

    kind: str
    domain: str
    source: SourceSpec
    renderers: tuple[RendererSpec, ...]
    verifier: VerifierSpec
    reward: RewardSpec
    limits: TaskLimits
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata after construction."""

        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
