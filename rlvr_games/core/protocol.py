"""Protocol definitions for reusable environment components."""

from typing import Any, Protocol, TypeVar

from rlvr_games.core.trajectory import EpisodeTrajectory
from rlvr_games.core.types import (
    AutoAction,
    EpisodeBoundary,
    Observation,
    ParseResult,
    RenderedImage,
    StepResult,
)

BackendStateT = TypeVar("BackendStateT")
BackendActionT = TypeVar("BackendActionT")
EnvStateT = TypeVar("EnvStateT", covariant=True)
EnvActionT = TypeVar("EnvActionT")
ScenarioStateT = TypeVar("ScenarioStateT", covariant=True)
RendererStateT = TypeVar("RendererStateT", contravariant=True)
RewardStateT = TypeVar("RewardStateT", contravariant=True)
RewardActionT = TypeVar("RewardActionT", contravariant=True)
RenderInputT = TypeVar("RenderInputT", contravariant=True)
AutoStateT = TypeVar("AutoStateT")
AutoActionT = TypeVar("AutoActionT")


class GameBackend(Protocol[BackendStateT, BackendActionT]):
    """Protocol for rule-backed game logic.

    A backend is the authoritative verifier for how model actions map onto the
    canonical game state. Implementations should reject illegal actions,
    produce transition metadata, and determine terminal conditions directly
    from executable rules.
    """

    def parse_action(
        self, state: BackendStateT, raw_action: str
    ) -> ParseResult[BackendActionT]:
        """Parse model output into a backend action result.

        Parameters
        ----------
        state : BackendStateT
            Current canonical state used to interpret the raw action.
        raw_action : str
            Model-produced action string.

        Returns
        -------
        ParseResult[BackendActionT]
            Structured parse result containing either a canonical action or an
            explicit rejection message for the current state.
        """
        ...

    def legal_actions(self, state: BackendStateT) -> list[str]:
        """Return serialized legal actions for the current state.

        Parameters
        ----------
        state : BackendStateT
            Canonical state for which legal actions should be enumerated.

        Returns
        -------
        list[str]
            Legal actions serialized in the format accepted by the backend.
        """
        ...

    def apply_action(
        self, state: BackendStateT, action: BackendActionT
    ) -> tuple[BackendStateT, dict[str, Any]]:
        """Apply an action and return the next state and transition metadata.

        Parameters
        ----------
        state : BackendStateT
            Canonical state before the transition.
        action : BackendActionT
            Parsed action to apply.

        Returns
        -------
        tuple[BackendStateT, dict[str, Any]]
            The next canonical state and verifier-produced metadata describing
            the transition.
        """
        ...

    def is_terminal(self, state: BackendStateT) -> bool:
        """Return whether the state is terminal.

        Parameters
        ----------
        state : BackendStateT
            Canonical state to inspect.

        Returns
        -------
        bool
            `True` when the episode should terminate under the game rules.
        """
        ...


class Environment(Protocol[EnvStateT, EnvActionT]):
    """Protocol for stateful reset/step environments.

    Environments coordinate game-specific backend logic, observation rendering,
    reward evaluation, and trajectory recording behind the minimal episode
    lifecycle needed by rollout runners and debugging tools.
    """

    @property
    def episode_finished(self) -> bool:
        """Return whether the active episode can accept more steps.

        Returns
        -------
        bool
            `True` when the episode has already terminated or been truncated.
        """
        ...

    @property
    def state(self) -> EnvStateT:
        """Return the current canonical state for the active episode.

        Returns
        -------
        EnvStateT
            The current canonical game state.
        """
        ...

    def legal_actions(self) -> tuple[str, ...]:
        """Return the legal serialized actions for the current state.

        Returns
        -------
        tuple[str, ...]
            Legal serialized actions accepted by the environment. This is a
            tooling-facing surface rather than part of the default
            observation.
        """
        ...

    def inspect_canonical_state(self) -> dict[str, object]:
        """Return a debug-oriented snapshot of the current canonical state.

        Returns
        -------
        dict[str, object]
            Structured state summary intended for debugging and inspection
            tools rather than as the primary model observation.
        """
        ...

    @property
    def trajectory(self) -> EpisodeTrajectory[EnvActionT]:
        """Return the trajectory recorded for the active episode.

        Returns
        -------
        EpisodeTrajectory[EnvActionT]
            The episode trajectory accumulated so far.
        """
        ...

    def reset(self, *, seed: int) -> tuple[Observation, dict[str, object]]:
        """Start a fresh episode.

        Parameters
        ----------
        seed : int
            Explicit seed used to initialize the scenario.

        Returns
        -------
        tuple[Observation, dict[str, object]]
            Initial observation and public-safe reset metadata.
        """
        ...

    def step(self, raw_action: str) -> StepResult:
        """Advance the episode by one raw model action.

        Parameters
        ----------
        raw_action : str
            Serialized action emitted by an agent or human operator.

        Returns
        -------
        StepResult
            Observation, reward, terminal flags, and transition metadata for
            the attempted step. The returned observation remains the
            agent-facing view, while canonical inspection belongs in
            `inspect_canonical_state()`.
        """
        ...

    def close(self) -> None:
        """Release any environment-owned external resources.

        Returns
        -------
        None
            This method is a no-op for environments without external resources.
        """
        ...


class AutoAdvancePolicy(Protocol[AutoStateT, AutoActionT]):
    """Protocol for automatic internal transitions between agent turns.

    Auto-advance policies let one environment step include zero or more
    verifier-backed internal actions, such as opponent replies or chance
    events, before control returns to the agent.
    """

    def reset(self, *, initial_state: AutoStateT) -> None:
        """Initialize policy state for a fresh episode.

        Parameters
        ----------
        initial_state : AutoStateT
            Canonical state returned by the scenario at reset time.
        """
        ...

    def is_agent_turn(self, *, state: AutoStateT) -> bool:
        """Return whether control should be returned to the agent.

        Parameters
        ----------
        state : AutoStateT
            Current canonical state after any already-applied transitions.

        Returns
        -------
        bool
            `True` when the agent should act next.
        """
        ...

    def select_internal_action(
        self,
        *,
        state: AutoStateT,
        backend: GameBackend[AutoStateT, AutoActionT],
    ) -> AutoAction[AutoActionT] | None:
        """Return the next internal action to auto-apply, if any.

        Parameters
        ----------
        state : AutoStateT
            Current canonical state that is not currently controlled by the
            agent.
        backend : GameBackend[AutoStateT, AutoActionT]
            Backend that will verify and apply the selected action.

        Returns
        -------
        AutoAction[AutoActionT] | None
            Selected internal action, or `None` when the policy cannot
            continue from the supplied state.
        """
        ...

    def episode_boundary(self, *, state: AutoStateT) -> EpisodeBoundary | None:
        """Return an explicit episode boundary for the supplied state.

        Parameters
        ----------
        state : AutoStateT
            Current canonical state after any already-applied transitions.

        Returns
        -------
        EpisodeBoundary | None
            Explicit task boundary to apply, or `None` when the episode should
            continue.
        """
        ...


class Scenario(Protocol[ScenarioStateT]):
    """Protocol for episode initialization logic."""

    def reset(self, *, seed: int) -> tuple[ScenarioStateT, dict[str, Any]]:
        """Create a fresh initial state for a new episode.

        Parameters
        ----------
        seed : int
            Explicit seed used to make scenario generation reproducible.

        Returns
        -------
        tuple[ScenarioStateT, dict[str, Any]]
            The initial canonical state and reset metadata describing the
            generated episode.
        """
        ...


class Renderer(Protocol[RendererStateT]):
    """Protocol for converting canonical state into observations."""

    def render(self, state: RendererStateT) -> Observation:
        """Render a model-facing observation from canonical state.

        Parameters
        ----------
        state : RendererStateT
            Canonical state to expose to the model.

        Returns
        -------
        Observation
            Model-facing observation derived from the canonical state.
        """
        ...


class TextRenderer(Protocol[RenderInputT]):
    """Protocol for rendering text from a canonical or derived value."""

    def render_text(self, value: RenderInputT, /) -> str:
        """Render a text view of the given value.

        Parameters
        ----------
        value : RenderInputT
            Canonical or derived value to render.

        Returns
        -------
        str
            Text representation derived from the input value.
        """
        ...


class ImageRenderer(Protocol[RenderInputT]):
    """Protocol for rendering in-memory images from a canonical value."""

    def render_images(self, value: RenderInputT, /) -> tuple[RenderedImage, ...]:
        """Render in-memory images for the given value.

        Parameters
        ----------
        value : RenderInputT
            Canonical or derived value to render.

        Returns
        -------
        tuple[RenderedImage, ...]
            Zero or more raster image payloads derived from the input value.
        """
        ...


class RewardFn(Protocol[RewardStateT, RewardActionT]):
    """Protocol for computing rewards from verified environment steps."""

    def evaluate(
        self,
        *,
        previous_state: RewardStateT,
        action: RewardActionT,
        next_state: RewardStateT,
        transition_info: dict[str, Any],
    ) -> float:
        """Return the reward for one accepted environment step.

        Parameters
        ----------
        previous_state : RewardStateT
            Canonical state before the agent action was applied.
        action : RewardActionT
            Parsed agent action that triggered the step.
        next_state : RewardStateT
            Canonical state after the agent action and any internal
            auto-advanced transitions have been applied.
        transition_info : dict[str, Any]
            Step metadata associated with the accepted action, including any
            internal auto-advanced transitions.

        Returns
        -------
        float
            Reward assigned to the accepted step.
        """
        ...
