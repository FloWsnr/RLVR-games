"""Tests for the async environment pool."""

from dataclasses import dataclass, field
from pathlib import Path
import time

from PIL import Image
import pytest

from rlvr_games.core.async_env import (
    AsyncEnvPool,
    AsyncResetResult,
    AsyncStepResult,
)
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.messages import (
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
    ImageMessagePart,
    TextMessagePart,
)
from rlvr_games.core.trajectory import ScenarioReset
from rlvr_games.core.types import EpisodeConfig, Observation, RenderedImage
from rlvr_games.task_specs import TaskSpec, task_spec_from_mapping

from tests.test_core.support import (
    CounterAction,
    CounterBackend,
    CounterRenderer,
    CounterReward,
    CounterScenario,
    CounterState,
    inspect_counter_state,
    make_counter_env,
)


def _make_connect4_task_spec(*, include_images: bool = False) -> TaskSpec:
    """Return a deterministic Connect 4 task spec for async pool tests."""
    return task_spec_from_mapping(
        payload={
            "schema_version": 1,
            "id": "connect4_async_pool_test",
            "game": "connect4",
            "scenario": {
                "kind": "random_position",
                "rows": 6,
                "columns": 7,
                "connect_length": 4,
                "min_start_moves": 0,
                "max_start_moves": 0,
            },
            "reward": {
                "kind": "terminal_outcome",
                "perspective": "mover",
                "win_reward": 1.0,
                "draw_reward": 0.0,
                "loss_reward": -1.0,
            },
            "episode": {
                "max_attempts": 8,
                "max_transitions": 8,
            },
            "observation": {
                "include_images": include_images,
                "image_size": 180,
            },
        },
        base_dir=Path(__file__).resolve().parents[2],
    )


def _build_counter_env() -> TurnBasedEnv[CounterState, CounterAction]:
    """Return the default counter env for spawn-safe async tests."""
    return make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )


@dataclass(slots=True)
class FlakyCounterScenario:
    """Fail the first reset, then behave like the standard counter scenario."""

    _failed: bool = field(init=False, default=False, repr=False)

    def reset(self, *, seed: int) -> ScenarioReset[CounterState]:
        """Raise on the first reset and succeed afterwards."""
        if not self._failed:
            self._failed = True
            raise RuntimeError("transient reset failure")
        return ScenarioReset(
            initial_state=CounterState(value=0),
            reset_info={"scenario": "flaky_counter", "seed": seed},
        )


def _build_flaky_reset_env() -> TurnBasedEnv[CounterState, CounterAction]:
    """Return one counter env whose first reset fails inside the worker."""
    return TurnBasedEnv(
        backend=CounterBackend(),
        scenario=FlakyCounterScenario(),
        renderer=CounterRenderer(),
        inspect_canonical_state_fn=inspect_counter_state,
        reward_fn=CounterReward(),
        config=EpisodeConfig(),
        observation_message_adapter=DefaultObservationMessageAdapter(
            policy=DefaultObservationMessagePolicy()
        ),
    )


class CounterImageRenderer:
    """Render the counter state with a tiny image payload."""

    def render(self, state: CounterState) -> Observation:
        """Return the rendered counter observation with an image."""
        image = Image.new("RGB", (2, 2), color=(state.value, 0, 0))
        return Observation(
            text=f"value={state.value}",
            images=(
                RenderedImage(
                    key=f"counter-{state.value}",
                    image=image,
                ),
            ),
            metadata={"value": state.value},
        )


def _build_counter_image_env() -> TurnBasedEnv[CounterState, CounterAction]:
    """Return one counter env with multimodal observations."""
    return TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterImageRenderer(),
        inspect_canonical_state_fn=inspect_counter_state,
        reward_fn=CounterReward(),
        config=EpisodeConfig(),
        observation_message_adapter=DefaultObservationMessageAdapter(
            policy=DefaultObservationMessagePolicy()
        ),
    )


def test_async_env_pool_reset_and_step_across_multiple_slots() -> None:
    task_spec = _make_connect4_task_spec()

    with AsyncEnvPool.from_task_specs(task_specs=(task_spec, task_spec)) as pool:
        pool.reset_all(seeds=(3, 7))

        reset_results: dict[int, AsyncResetResult] = {}
        while len(reset_results) < 2:
            ready_results = pool.recv_ready(max_results=2, timeout_seconds=5.0)
            for result in ready_results:
                assert isinstance(result, AsyncResetResult)
                reset_results[result.slot_id] = result

        assert pool.pending_slot_ids == ()
        for slot_id, result in reset_results.items():
            assert result.episode_index == 0
            assert result.episode_finished is False
            assert result.turn is not None
            assert result.turn.action_context.turn_index == 0
            assert result.observation.text is not None
            text_part = result.turn.messages[0].content[0]
            assert isinstance(text_part, TextMessagePart)
            assert "Respond with one legal column number" in text_part.text
            assert slot_id in {0, 1}

        pool.step(slot_id=0, raw_action="1")
        pool.step(slot_id=1, raw_action="2")

        step_results: dict[int, AsyncStepResult] = {}
        while len(step_results) < 2:
            ready_results = pool.recv_ready(max_results=2, timeout_seconds=5.0)
            for result in ready_results:
                assert isinstance(result, AsyncStepResult)
                step_results[result.slot_id] = result

        for result in step_results.values():
            assert result.episode_index == 0
            assert result.step_result.accepted is True
            assert result.step_result.terminated is False
            assert result.step_result.truncated is False
            assert result.turn is not None
            assert result.turn.action_context.turn_index == 1


def test_async_env_pool_propagates_invalid_action_errors_and_slot_stays_usable() -> (
    None
):
    task_spec = _make_connect4_task_spec()

    with AsyncEnvPool.from_task_specs(task_specs=(task_spec,)) as pool:
        pool.reset(slot_id=0, seed=11)
        reset_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(reset_result, AsyncResetResult)
        assert reset_result.turn is not None

        pool.step(slot_id=0, raw_action="9")
        with pytest.raises(InvalidActionError):
            pool.recv(timeout_seconds=5.0)

        assert pool.pending_slot_ids == ()

        pool.step(slot_id=0, raw_action="1")
        step_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(step_result, AsyncStepResult)
        assert step_result.step_result.accepted is True
        assert step_result.turn is not None


def test_async_env_pool_rejects_dispatch_to_busy_slots() -> None:
    task_spec = _make_connect4_task_spec()

    with AsyncEnvPool.from_task_specs(task_specs=(task_spec,)) as pool:
        pool.reset(slot_id=0, seed=13)

        with pytest.raises(RuntimeError, match="pending task"):
            pool.step(slot_id=0, raw_action="1")

        result = pool.recv(timeout_seconds=5.0)
        assert isinstance(result, AsyncResetResult)


def test_recv_ready_preserves_successful_results_when_another_slot_fails() -> None:
    with AsyncEnvPool(env_factories=(_build_counter_env, _build_counter_env)) as pool:
        pool.reset_all(seeds=(1, 2))
        reset_results: dict[int, AsyncResetResult] = {}
        while len(reset_results) < 2:
            for result in pool.recv_ready(max_results=2, timeout_seconds=5.0):
                assert isinstance(result, AsyncResetResult)
                reset_results[result.slot_id] = result

        pool.step(slot_id=0, raw_action="1")
        pool.step(slot_id=1, raw_action="bad")
        time.sleep(0.1)

        ready_results = pool.recv_ready(max_results=2, timeout_seconds=5.0)

        assert len(ready_results) == 1
        assert isinstance(ready_results[0], AsyncStepResult)
        assert ready_results[0].slot_id in {0, 1}
        assert ready_results[0].step_result.accepted is True
        assert pool.pending_slot_ids == ()

        with pytest.raises(InvalidActionError):
            pool.recv(timeout_seconds=5.0)


def test_reset_failures_do_not_advance_episode_index() -> None:
    with AsyncEnvPool(env_factories=(_build_flaky_reset_env,)) as pool:
        pool.reset(slot_id=0, seed=5)

        with pytest.raises(RuntimeError, match="transient reset failure"):
            pool.recv(timeout_seconds=5.0)

        pool.reset(slot_id=0, seed=6)
        reset_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(reset_result, AsyncResetResult)
        assert reset_result.episode_index == 0
        assert reset_result.reset_info["seed"] == 6


def test_async_env_pool_strips_duplicate_images_from_actionable_observations() -> None:
    with AsyncEnvPool(env_factories=(_build_counter_image_env,)) as pool:
        pool.reset(slot_id=0, seed=3)
        reset_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(reset_result, AsyncResetResult)
        assert reset_result.turn is not None
        assert reset_result.observation.images == ()
        assert any(
            isinstance(part, ImageMessagePart)
            for part in reset_result.turn.messages[0].content
        )

        pool.step(slot_id=0, raw_action="1")
        step_result = pool.recv(timeout_seconds=5.0)

        assert isinstance(step_result, AsyncStepResult)
        assert step_result.turn is not None
        assert step_result.step_result.observation.images == ()
        assert any(
            isinstance(part, ImageMessagePart)
            for part in step_result.turn.messages[0].content
        )
