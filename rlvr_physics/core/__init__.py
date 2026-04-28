"""Core task abstractions for RLVR physics."""

from rlvr_physics.core.instances import (
    TaskInstance,
)
from rlvr_physics.core.payloads import (
    freeze_mapping,
    mapping_to_dict,
    stable_hash,
    to_plain_data,
)
from rlvr_physics.core.factory import (
    ConfiguredTask,
    TaskInstanceBuilder,
    TaskSessionBuilder,
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
    new_session_id,
)
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.core.trajectory import TaskTrajectory, TrajectoryEvent

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
    "TaskInstanceBuilder",
    "TaskResetResult",
    "TaskSession",
    "TaskSessionBuilder",
    "TaskSpec",
    "TaskStepResult",
    "TaskSubmission",
    "TaskTrajectory",
    "TaskTurn",
    "TextContent",
    "TrajectoryEvent",
    "VerifierSpec",
    "freeze_mapping",
    "image_observation",
    "mapping_to_dict",
    "new_session_id",
    "stable_hash",
    "text_observation",
    "to_plain_data",
]
