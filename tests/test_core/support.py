"""Shared helpers for core tests."""

from dataclasses import dataclass
from typing import Any

from rlvr_games.core import ZeroReward
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.protocol import AutoAdvancePolicy, ResetEventPolicy, RewardFn
from rlvr_games.core.trajectory import ScenarioReset
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

    def reset(self, *, seed: int) -> ScenarioReset[CounterState]:
        """Return the standard initial counter state."""
        return ScenarioReset(
            initial_state=CounterState(value=0),
            reset_info={"scenario": "counter", "seed": seed},
        )


class CounterRenderer:
    """Render the counter state as a simple text observation."""

    def render(self, state: CounterState) -> Observation:
        """Return the rendered counter observation."""
        return Observation(text=f"value={state.value}", metadata={"value": state.value})


def inspect_counter_state(state: CounterState) -> dict[str, object]:
    """Return a structured debug view of the counter state.

    Parameters
    ----------
    state : CounterState
        Canonical counter state to inspect.

    Returns
    -------
    dict[str, object]
        Debug-oriented state summary for inspection tooling.
    """
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


def make_counter_env(
    *,
    backend: CounterBackend,
    config: EpisodeConfig,
    reward_fn: RewardFn[CounterState, CounterAction] | None = None,
    reset_event_policy: ResetEventPolicy[CounterState] | None = None,
    auto_advance_policy: AutoAdvancePolicy[CounterState, CounterAction] | None = None,
) -> TurnBasedEnv[CounterState, CounterAction]:
    """Construct the standard counter env used in core tests."""
    if reward_fn is None:
        reward_fn = CounterReward()
    return TurnBasedEnv(
        backend=backend,
        scenario=CounterScenario(),
        renderer=CounterRenderer(),
        inspect_canonical_state_fn=inspect_counter_state,
        reward_fn=reward_fn,
        config=config,
        reset_event_policy=reset_event_policy,
        auto_advance_policy=auto_advance_policy,
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
    "CounterAction",
    "CounterBackend",
    "CounterReward",
    "CounterScenario",
    "CounterState",
    "inspect_counter_state",
    "make_chess_env_for_core_tests",
    "make_counter_env",
    "STANDARD_START_FEN",
]
