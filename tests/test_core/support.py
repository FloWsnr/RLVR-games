"""Shared helpers for core tests."""

from dataclasses import dataclass, field
from typing import Any

from rlvr_games.core import ZeroReward
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.rollout import ActionContext
from rlvr_games.core.types import EpisodeConfig, Observation, ParseResult
from rlvr_games.games.chess import (
    ChessAction,
    ChessBoardOrientation,
    ChessState,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import (
    STANDARD_START_FEN,
    StartingPositionScenario,
)


@dataclass(slots=True)
class CounterState:
    """Simple integer-valued state used by core env tests."""

    value: int


@dataclass(slots=True)
class CounterAction:
    """Simple additive action used by core env tests."""

    delta: int


class CounterScenario:
    """Return a fixed initial counter state."""

    def reset(self, *, seed: int) -> tuple[CounterState, dict[str, Any]]:
        """Return the standard initial counter state."""
        return CounterState(value=0), {"scenario": "counter", "seed": seed}


class CounterRenderer:
    """Render the counter state as a simple text observation."""

    def render(self, state: CounterState) -> Observation:
        """Return the rendered counter observation."""
        return Observation(text=f"value={state.value}", metadata={"value": state.value})


class CounterStateInspector:
    """Expose the counter state for debug inspection."""

    def inspect_state(self, state: CounterState) -> dict[str, object]:
        """Return a structured view of the counter state."""
        return {"value": state.value}


class CounterBackend:
    """Minimal backend used to test the generic turn-based env."""

    def parse_action(
        self, state: CounterState, raw_action: str
    ) -> ParseResult[CounterAction]:
        """Parse an integer delta or reject the sentinel invalid action."""
        del state
        if raw_action == "bad":
            return ParseResult(action=None, error="bad action")
        return ParseResult(action=CounterAction(delta=int(raw_action)), error=None)

    def legal_actions(self, state: CounterState) -> list[str]:
        """Return the minimal legal action set."""
        del state
        return ["1"]

    def apply_action(
        self, state: CounterState, action: CounterAction
    ) -> tuple[CounterState, dict[str, Any]]:
        """Apply the additive counter transition."""
        next_state = CounterState(value=state.value + action.delta)
        return next_state, {"value": next_state.value}

    def is_terminal(self, state: CounterState) -> bool:
        """Terminate once the counter reaches three."""
        return state.value >= 3


class CounterReward:
    """Return the applied delta as reward."""

    def evaluate(
        self,
        *,
        previous_state: CounterState,
        action: CounterAction,
        next_state: CounterState,
        transition_info: dict[str, Any],
    ) -> float:
        """Return the applied delta as a float reward."""
        del previous_state
        del next_state
        del transition_info
        return float(action.delta)


class ApplyRejectingCounterBackend(CounterBackend):
    """Counter backend that rejects a parsed action during apply time."""

    def parse_action(
        self, state: CounterState, raw_action: str
    ) -> ParseResult[CounterAction]:
        """Parse the sentinel zero action before apply-time rejection."""
        if raw_action == "zero":
            return ParseResult(action=CounterAction(delta=0), error=None)
        return super().parse_action(state, raw_action)

    def apply_action(
        self, state: CounterState, action: CounterAction
    ) -> tuple[CounterState, dict[str, Any]]:
        """Reject zero deltas after parsing succeeds."""
        if action.delta == 0:
            raise InvalidActionError("zero delta is rejected during apply")
        return super().apply_action(state, action)


@dataclass(slots=True)
class ScriptedAgent:
    """Deterministic rollout agent with optional context capture."""

    actions: list[str]
    contexts: list[ActionContext] = field(default_factory=list)

    def act(self, observation: Observation, context: ActionContext) -> str:
        """Return the next scripted action and record the context."""
        del observation
        self.contexts.append(context)
        if not self.actions:
            raise AssertionError("ScriptedAgent ran out of actions.")
        return self.actions.pop(0)


def make_counter_env(
    *,
    backend: CounterBackend,
    config: EpisodeConfig,
) -> TurnBasedEnv[CounterState, CounterAction]:
    """Construct the standard counter env used in core tests."""
    return TurnBasedEnv(
        backend=backend,
        scenario=CounterScenario(),
        renderer=CounterRenderer(),
        state_inspector=CounterStateInspector(),
        reward_fn=CounterReward(),
        config=config,
    )


def make_chess_env_for_core_tests(
    *,
    config: EpisodeConfig,
    initial_fen: str,
) -> TurnBasedEnv[ChessState, ChessAction]:
    """Construct the standard chess env used by core integration tests."""
    return make_chess_env(
        scenario=StartingPositionScenario(initial_fen=initial_fen),
        reward_fn=ZeroReward(),
        config=config,
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )


__all__ = [
    "ApplyRejectingCounterBackend",
    "CounterBackend",
    "CounterState",
    "ScriptedAgent",
    "make_chess_env_for_core_tests",
    "make_counter_env",
    "STANDARD_START_FEN",
]
