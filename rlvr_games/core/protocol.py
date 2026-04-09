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
    """Game rules, parsing, transitions, and terminal checks."""

    def parse_action(self, state: StateT, raw_action: str) -> ActionT:
        """Parse model output into a backend action."""
        ...

    def legal_actions(self, state: StateT) -> list[str]:
        """Return model-facing legal actions for the current state."""
        ...

    def apply_action(
        self, state: StateT, action: ActionT
    ) -> tuple[StateT, dict[str, Any]]:
        """Apply an action and return the next state and transition metadata."""
        ...

    def is_terminal(self, state: StateT) -> bool:
        """Return whether the state is terminal."""
        ...


class Scenario(Protocol[ScenarioStateT]):
    """Episode initializer."""

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[ScenarioStateT, dict[str, Any]]:
        """Create a fresh initial state for a new episode."""
        ...


class Renderer(Protocol[RendererStateT]):
    """State-to-observation adapter."""

    def render(self, state: RendererStateT) -> Observation:
        """Render a model-facing observation from canonical state."""
        ...


class RewardFn(Protocol[RewardStateT, RewardActionT]):
    """Computes verifiable rewards from state transitions."""

    def evaluate(
        self,
        *,
        previous_state: RewardStateT,
        action: RewardActionT,
        next_state: RewardStateT,
        transition_info: dict[str, Any],
    ) -> float:
        """Return the reward for a transition."""
        ...
