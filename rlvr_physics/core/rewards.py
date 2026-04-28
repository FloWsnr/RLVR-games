"""Shared reward result types."""

from dataclasses import dataclass, field
from typing import Mapping

from rlvr_physics.core.payloads import freeze_mapping


@dataclass(frozen=True)
class RewardResult:
    """Trainer-facing reward result produced by a task reward policy.

    Parameters
    ----------
    reward:
        Trainer-facing scalar reward.
    score:
        Optional domain score used for filtering or reporting.
    public_info:
        Trainer-safe reward metadata.
    debug_info:
        Privileged local reward metadata for evaluation and debugging.

    Attributes
    ----------
    reward:
        Trainer-facing scalar reward.
    score:
        Optional domain score used for filtering or reporting.
    public_info:
        Frozen trainer-safe reward metadata.
    debug_info:
        Frozen privileged local reward metadata.
    """

    reward: float
    score: float | None
    public_info: Mapping[str, object] = field(default_factory=dict)
    debug_info: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze reward metadata after construction."""

        object.__setattr__(self, "public_info", freeze_mapping(self.public_info))
        object.__setattr__(self, "debug_info", freeze_mapping(self.debug_info))
