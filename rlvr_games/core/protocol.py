"""Protocol definitions for reusable environment components."""

from pathlib import Path
from typing import Any, Protocol, TypeVar

from rlvr_games.core.trajectory import EpisodeTrajectory
from rlvr_games.core.types import Observation
from rlvr_games.core.types import StepResult

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ScenarioStateT = TypeVar("ScenarioStateT", covariant=True)
RendererStateT = TypeVar("RendererStateT", contravariant=True)
RewardStateT = TypeVar("RewardStateT", contravariant=True)
RewardActionT = TypeVar("RewardActionT", contravariant=True)
RenderInputT = TypeVar("RenderInputT", contravariant=True)


class GameBackend(Protocol[StateT, ActionT]):
    """Protocol for rule-backed game logic.

    A backend is the authoritative verifier for how model actions map onto the
    canonical game state. Implementations should reject illegal actions,
    produce transition metadata, and determine terminal conditions directly
    from executable rules.
    """

    def parse_action(self, state: StateT, raw_action: str) -> ActionT:
        """Parse model output into a backend action object.

        Parameters
        ----------
        state : StateT
            Current canonical state used to interpret the raw action.
        raw_action : str
            Model-produced action string.

        Returns
        -------
        ActionT
            Parsed action in the backend's canonical action type.
        """
        ...

    def legal_actions(self, state: StateT) -> list[str]:
        """Return model-facing legal actions for the current state.

        Parameters
        ----------
        state : StateT
            Canonical state for which legal actions should be enumerated.

        Returns
        -------
        list[str]
            Legal actions serialized in the format expected from the model.
        """
        ...

    def apply_action(
        self, state: StateT, action: ActionT
    ) -> tuple[StateT, dict[str, Any]]:
        """Apply an action and return the next state and transition metadata.

        Parameters
        ----------
        state : StateT
            Canonical state before the transition.
        action : ActionT
            Parsed action to apply.

        Returns
        -------
        tuple[StateT, dict[str, Any]]
            The next canonical state and verifier-produced metadata describing
            the transition.
        """
        ...

    def is_terminal(self, state: StateT) -> bool:
        """Return whether the state is terminal.

        Parameters
        ----------
        state : StateT
            Canonical state to inspect.

        Returns
        -------
        bool
            `True` when the episode should terminate under the game rules.
        """
        ...


class Environment(Protocol[StateT, ActionT]):
    """Protocol for stateful reset/step environments.

    Environments coordinate game-specific backend logic, observation rendering,
    reward evaluation, and trajectory recording behind the minimal episode
    lifecycle needed by rollout runners and debugging tools.
    """

    backend: GameBackend[StateT, ActionT]

    @property
    def state(self) -> StateT:
        """Return the current canonical state for the active episode.

        Returns
        -------
        StateT
            The current canonical game state.
        """
        ...

    @property
    def trajectory(self) -> EpisodeTrajectory[ActionT]:
        """Return the trajectory recorded for the active episode.

        Returns
        -------
        EpisodeTrajectory[ActionT]
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
            Initial observation and reset metadata.
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
            the attempted step.
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
    """Protocol for rendering image paths from a canonical or derived value."""

    def render_images(self, value: RenderInputT, /) -> tuple[Path, ...]:
        """Render image paths for the given value.

        Parameters
        ----------
        value : RenderInputT
            Canonical or derived value to render.

        Returns
        -------
        tuple[Path, ...]
            Zero or more filesystem paths to images derived from the input
            value.
        """
        ...


class RewardFn(Protocol[RewardStateT, RewardActionT]):
    """Protocol for computing rewards from verified transitions."""

    def evaluate(
        self,
        *,
        previous_state: RewardStateT,
        action: RewardActionT,
        next_state: RewardStateT,
        transition_info: dict[str, Any],
    ) -> float:
        """Return the reward for a transition.

        Parameters
        ----------
        previous_state : RewardStateT
            Canonical state before the action was applied.
        action : RewardActionT
            Parsed action that triggered the transition.
        next_state : RewardStateT
            Canonical state after the transition.
        transition_info : dict[str, Any]
            Verifier-produced metadata associated with the transition.

        Returns
        -------
        float
            Reward assigned to the transition.
        """
        ...
