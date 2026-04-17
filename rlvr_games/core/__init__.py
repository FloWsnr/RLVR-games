"""Core environment abstractions for RLVR games."""

from rlvr_games.core.action_context import (
    ActionContext,
    AgentContextProjector,
    AgentVisibleEvent,
    PublicResetEvent,
    ProjectedActionContext,
)
from rlvr_games.core.async_env import AsyncEnvPool, AsyncResetResult, AsyncStepResult
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import (
    EpisodeFinishedError,
    EnvironmentNotResetError,
    InvalidActionError,
    RLVRGamesError,
)
from rlvr_games.core.messages import (
    ChatMessage,
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
    ImageMessagePart,
    MessageRole,
    ObservationMessageAdapter,
    ObservationMessagePolicy,
    TextMessagePart,
)
from rlvr_games.core.protocol import (
    AutoAdvancePolicy,
    Environment,
    ImageRenderer,
    ResetEventPolicy,
    TextRenderer,
)
from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.rollout import PreparedTurn, build_action_context, prepare_turn
from rlvr_games.core.trajectory import (
    AppliedResetEvent,
    EpisodeTrajectory,
    RecordedResetEvent,
    RecordedTransition,
    ScenarioReset,
    TrajectoryStep,
)
from rlvr_games.core.types import (
    AutoAction,
    EpisodeConfig,
    EpisodeBoundary,
    InvalidActionMode,
    InvalidActionPolicy,
    Observation,
    ParseResult,
    RenderedImage,
    StepResult,
)
from rlvr_games.core.workflow import (
    AsyncWorkflowSession,
    LocalWorkflowSession,
    WorkflowResetResult,
    WorkflowSession,
    WorkflowSessionProtocol,
    WorkflowSubmission,
    WorkflowTurn,
)

__all__ = [
    "ActionContext",
    "AgentContextProjector",
    "AgentVisibleEvent",
    "PublicResetEvent",
    "PreparedTurn",
    "ProjectedActionContext",
    "AppliedResetEvent",
    "AsyncEnvPool",
    "AsyncResetResult",
    "AsyncStepResult",
    "AsyncWorkflowSession",
    "AutoAction",
    "AutoAdvancePolicy",
    "ChatMessage",
    "DefaultObservationMessageAdapter",
    "DefaultObservationMessagePolicy",
    "Environment",
    "EpisodeConfig",
    "EpisodeBoundary",
    "EpisodeFinishedError",
    "EpisodeTrajectory",
    "EnvironmentNotResetError",
    "ImageRenderer",
    "ImageMessagePart",
    "InvalidActionError",
    "InvalidActionMode",
    "InvalidActionPolicy",
    "LocalWorkflowSession",
    "MessageRole",
    "Observation",
    "ObservationMessageAdapter",
    "ObservationMessagePolicy",
    "ParseResult",
    "RecordedResetEvent",
    "RecordedTransition",
    "RenderedImage",
    "ResetEventPolicy",
    "RLVRGamesError",
    "ScenarioReset",
    "StepResult",
    "TextRenderer",
    "TextMessagePart",
    "TrajectoryStep",
    "TurnBasedEnv",
    "ZeroReward",
    "WorkflowResetResult",
    "WorkflowSession",
    "WorkflowSessionProtocol",
    "WorkflowSubmission",
    "WorkflowTurn",
    "build_action_context",
    "prepare_turn",
]
