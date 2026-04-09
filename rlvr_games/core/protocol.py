"""Protocol definitions for reusable environment components."""

from typing import Any, Protocol, TypeVar

from rlvr_games.core.types import Observation

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ScenarioStateT = TypeVar("ScenarioStateT", covariant=True)
RendererStateT = TypeVar("RendererStateT", contravariant=True)
RewardStateT = TypeVar("RewardStateT", contravariant=True)
RewardActionT = TypeVar("RewardActionT", contravariant=True)


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


class Scenario(Protocol[ScenarioStateT]):
    """Protocol for episode initialization logic."""

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[ScenarioStateT, dict[str, Any]]:
        """Create a fresh initial state for a new episode.

        Parameters
        ----------
        seed : int | None
            Optional seed used to make scenario generation reproducible.

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
