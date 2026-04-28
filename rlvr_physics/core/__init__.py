"""Core task abstractions for RLVR physics."""

from rlvr_physics.core.instances import (
    TaskInstance,
)
from rlvr_physics.core.factory import (
    ConfiguredTask,
)
from rlvr_physics.core.rendering import (
    ImageContent,
    ObservationContent,
    RenderedObservation,
    TextContent,
    image_observation,
    text_observation,
)
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskSession,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
)
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)

__all__ = [
    "ImageContent",
    "ObservationContent",
    "ConfiguredTask",
    "RenderedObservation",
    "RewardResult",
    "RendererSpec",
    "RewardSpec",
    "SourceSpec",
    "TaskInstance",
    "TaskResetResult",
    "TaskSession",
    "TaskSpec",
    "TaskStepResult",
    "TaskSubmission",
    "TaskTurn",
    "TextContent",
    "VerifierSpec",
    "image_observation",
    "text_observation",
]
