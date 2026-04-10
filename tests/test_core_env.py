"""Standard-library tests for the minimal scaffold."""

import unittest
from dataclasses import dataclass
from typing import Any

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EnvironmentNotResetError
from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.types import EpisodeConfig, Observation
from rlvr_games.games.chess import (
    AsciiBoardFormatter,
    ChessBackend,
    ChessEnv,
    ChessObservationRenderer,
    STANDARD_START_FEN,
    StartingPositionScenario,
)


@dataclass(slots=True)
class CounterState:
    value: int


@dataclass(slots=True)
class CounterAction:
    delta: int


class CounterScenario:
    def reset(self, *, seed: int) -> tuple[CounterState, dict[str, Any]]:
        return CounterState(value=0), {"scenario": "counter", "seed": seed}


class CounterRenderer:
    def render(self, state: CounterState) -> Observation:
        return Observation(text=f"value={state.value}", metadata={"value": state.value})


class CounterBackend:
    def parse_action(self, state: CounterState, raw_action: str) -> CounterAction:
        return CounterAction(delta=int(raw_action))

    def legal_actions(self, state: CounterState) -> list[str]:
        return ["1"]

    def apply_action(
        self, state: CounterState, action: CounterAction
    ) -> tuple[CounterState, dict[str, Any]]:
        next_state = CounterState(value=state.value + action.delta)
        return next_state, {"value": next_state.value}

    def is_terminal(self, state: CounterState) -> bool:
        return state.value >= 3


class CounterReward:
    def evaluate(
        self,
        *,
        previous_state: CounterState,
        action: CounterAction,
        next_state: CounterState,
        transition_info: dict[str, Any],
    ) -> float:
        return float(action.delta)


class TurnBasedEnvTests(unittest.TestCase):
    def test_step_requires_reset(self) -> None:
        env = TurnBasedEnv(
            backend=CounterBackend(),
            scenario=CounterScenario(),
            renderer=CounterRenderer(),
            reward_fn=CounterReward(),
            config=EpisodeConfig(),
        )
        with self.assertRaises(EnvironmentNotResetError):
            env.step("1")

    def test_records_trajectory_until_terminal(self) -> None:
        env = TurnBasedEnv(
            backend=CounterBackend(),
            scenario=CounterScenario(),
            renderer=CounterRenderer(),
            reward_fn=CounterReward(),
            config=EpisodeConfig(max_turns=10),
        )

        observation, info = env.reset(seed=7)
        self.assertEqual(observation.text, "value=0")
        self.assertEqual(info["seed"], 7)

        first = env.step("1")
        second = env.step("1")
        third = env.step("1")

        self.assertFalse(first.terminated)
        self.assertFalse(second.terminated)
        self.assertTrue(third.terminated)
        self.assertEqual(env.trajectory.total_reward, 3.0)
        self.assertEqual(len(env.trajectory.steps), 3)
        self.assertEqual(env.trajectory.steps[-1].observation.metadata["value"], 3)


class ChessEnvTests(unittest.TestCase):
    def test_reset_returns_starting_position(self) -> None:
        env = ChessEnv(
            backend=ChessBackend(),
            scenario=StartingPositionScenario(),
            renderer=ChessObservationRenderer(
                board_formatter=AsciiBoardFormatter(),
                image_renderer=None,
            ),
            reward_fn=ZeroReward(),
            config=EpisodeConfig(),
        )
        observation, info = env.reset(seed=123)

        self.assertIn(STANDARD_START_FEN, observation.text or "")
        self.assertEqual(info["scenario"], "starting_position")
        self.assertEqual(info["seed"], 123)


if __name__ == "__main__":
    unittest.main()
