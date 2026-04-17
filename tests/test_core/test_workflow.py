"""Workflow session tests."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
import time
from typing import cast

from PIL import Image
import pytest

from rlvr_games.core import (
    ActionContext,
    AsyncEnvPool,
    AsyncResetResult,
    AsyncWorkflowSession,
    ChatMessage,
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
    EpisodeConfig,
    EnvironmentNotResetError,
    ImageMessagePart,
    Observation,
    RenderedImage,
    TextMessagePart,
    TurnBasedEnv,
    WorkflowSession,
)

from tests.test_core.support import (
    CounterAction,
    CounterBackend,
    CounterReward,
    CounterScenario,
    CounterState,
    inspect_counter_state,
    make_counter_env,
)
from rlvr_games.core.trajectory import ScenarioReset


class CounterImageRenderer:
    """Render the counter state as text plus an in-memory image."""

    def render(self, state: CounterState) -> Observation:
        """Return the rendered counter observation with one image payload."""
        return Observation(
            text=f"value={state.value}",
            images=(
                RenderedImage(
                    key=f"counter-{state.value}",
                    image=Image.new("RGB", (2, 2), color=(state.value, 0, 0)),
                ),
            ),
            metadata={"value": state.value},
        )


@dataclass(slots=True)
class SlowCounterScenario:
    """Sleep briefly during reset to control async slot ordering."""

    delay_seconds: float

    def reset(self, *, seed: int) -> ScenarioReset[CounterState]:
        """Return the standard counter reset after a short delay."""
        time.sleep(self.delay_seconds)
        return ScenarioReset(
            initial_state=CounterState(value=0),
            reset_info={"scenario": "slow_counter", "seed": seed},
        )


class DecoratedImageObservationAdapter:
    """Append an extra non-canonical image to each observation message."""

    def __init__(self) -> None:
        self._base = DefaultObservationMessageAdapter(
            policy=DefaultObservationMessagePolicy()
        )

    def to_messages(
        self,
        *,
        observation: Observation,
        action_context: ActionContext,
    ) -> tuple[ChatMessage, ...]:
        """Return the base messages plus one decorative image part."""
        base_messages = self._base.to_messages(
            observation=observation,
            action_context=action_context,
        )
        message = base_messages[-1]
        decorative_image = RenderedImage(
            key="decorative",
            image=Image.new("RGB", (1, 1), color=(0, 255, 0)),
        )
        return (
            *base_messages[:-1],
            ChatMessage(
                role=message.role,
                content=message.content
                + (
                    ImageMessagePart(
                        image=decorative_image,
                        alt_text="decorative",
                    ),
                ),
            ),
        )


def _build_async_counter_env() -> TurnBasedEnv[CounterState, CounterAction]:
    """Return one spawn-safe counter env for async workflow tests."""
    return make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )


def _build_async_counter_image_env() -> TurnBasedEnv[CounterState, CounterAction]:
    """Return one spawn-safe multimodal counter env for async workflow tests."""
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


def _build_async_slow_counter_env() -> TurnBasedEnv[CounterState, CounterAction]:
    """Return one spawn-safe counter env whose reset is intentionally slow."""
    return TurnBasedEnv(
        backend=CounterBackend(),
        scenario=SlowCounterScenario(delay_seconds=0.2),
        renderer=CounterImageRenderer(),
        inspect_canonical_state_fn=inspect_counter_state,
        reward_fn=CounterReward(),
        config=EpisodeConfig(),
        observation_message_adapter=DefaultObservationMessageAdapter(
            policy=DefaultObservationMessagePolicy()
        ),
    )


def _build_async_counter_decorated_image_env() -> TurnBasedEnv[
    CounterState, CounterAction
]:
    """Return one spawn-safe counter env with extra message-only images."""
    return TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterImageRenderer(),
        inspect_canonical_state_fn=inspect_counter_state,
        reward_fn=CounterReward(),
        config=EpisodeConfig(),
        observation_message_adapter=DecoratedImageObservationAdapter(),
    )


def test_workflow_session_reset_prepares_initial_turn() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = WorkflowSession(env=env)

    reset_result = session.reset(seed=7)

    assert reset_result.reset_info == {"scenario": "counter", "seed": 7}
    assert reset_result.turn is not None
    assert reset_result.turn.observation.metadata["value"] == 0
    assert reset_result.turn.action_context.turn_index == 0
    text_part = reset_result.turn.messages[0].content[0]
    assert isinstance(text_part, TextMessagePart)
    assert text_part.text == "Observation:\nvalue=0"
    assert session.current_observation.metadata["value"] == 0
    assert session.done is False


def test_workflow_session_submit_can_use_action_extractor() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = WorkflowSession(
        env=env,
        action_extractor=lambda assistant_output: assistant_output.removeprefix(
            "move: "
        ),
    )
    session.reset(seed=3)

    submission = session.submit("move: 1")

    assert submission.assistant_output == "move: 1"
    assert submission.raw_action == "1"
    assert submission.step_result.accepted is True
    assert submission.step_result.reward == 1.0
    assert submission.turn is not None
    assert submission.turn.action_context.turn_index == 1
    assert submission.turn.observation.metadata["value"] == 1
    assert session.current_observation.metadata["value"] == 1


def test_workflow_session_submit_returns_no_turn_after_terminal_step() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = WorkflowSession(env=env)
    session.reset(seed=5)

    session.submit("1")
    session.submit("1")
    submission = session.submit("1")

    assert submission.done is True
    assert submission.step_result.terminated is True
    assert submission.turn is None
    assert session.done is True
    assert session.episode_return == 3.0


def test_workflow_session_can_wrap_async_pool_slot() -> None:
    with AsyncEnvPool(env_factories=(_build_async_counter_env,)) as pool:
        session = pool.session(slot_id=0)

        assert isinstance(session, AsyncWorkflowSession)

        reset_result = session.reset(seed=21)

        assert reset_result.reset_info == {"scenario": "counter", "seed": 21}
        assert reset_result.turn is not None
        assert reset_result.turn.action_context.turn_index == 0
        assert reset_result.turn.observation.metadata["value"] == 0

        submission = session.submit("1")

        assert submission.step_result.accepted is True
        assert submission.turn is not None
        assert submission.turn.action_context.turn_index == 1
        assert submission.turn.observation.metadata["value"] == 1
        assert session.current_observation.metadata["value"] == 1
        assert session.episode_return == 1.0
        assert session.done is False


def test_async_backed_workflow_session_preserves_multimodal_observations() -> None:
    with AsyncEnvPool(env_factories=(_build_async_counter_image_env,)) as pool:
        session = pool.session(slot_id=0)

        reset_result = session.reset(seed=17)

        assert len(reset_result.observation.images) == 1
        assert reset_result.turn is not None
        assert len(reset_result.turn.observation.images) == 1
        initial_image_part = reset_result.turn.messages[0].content[1]
        assert isinstance(initial_image_part, ImageMessagePart)
        assert initial_image_part.image.key == "counter-0"

        submission = session.submit("1")

        assert len(submission.step_result.observation.images) == 1
        assert submission.turn is not None
        assert len(submission.turn.observation.images) == 1
        next_image_part = submission.turn.messages[0].content[1]
        assert isinstance(next_image_part, ImageMessagePart)
        assert next_image_part.image.key == "counter-1"


def test_workflow_sessions_keep_renderer_images_canonical_with_custom_adapter() -> None:
    env = TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterImageRenderer(),
        inspect_canonical_state_fn=inspect_counter_state,
        reward_fn=CounterReward(),
        config=EpisodeConfig(),
        observation_message_adapter=DecoratedImageObservationAdapter(),
    )
    local_session = WorkflowSession(env=env)

    local_reset = local_session.reset(seed=29)

    assert tuple(image.key for image in local_reset.observation.images) == (
        "counter-0",
    )
    assert local_reset.turn is not None
    local_image_keys = tuple(
        part.image.key
        for part in local_reset.turn.messages[0].content
        if isinstance(part, ImageMessagePart)
    )
    assert local_image_keys == ("counter-0", "decorative")

    local_submission = local_session.submit("1")

    assert tuple(
        image.key for image in local_submission.step_result.observation.images
    ) == ("counter-1",)
    assert local_submission.turn is not None
    local_step_image_keys = tuple(
        part.image.key
        for part in local_submission.turn.messages[0].content
        if isinstance(part, ImageMessagePart)
    )
    assert local_step_image_keys == ("counter-1", "decorative")

    with AsyncEnvPool(
        env_factories=(_build_async_counter_decorated_image_env,)
    ) as pool:
        async_session = pool.session(slot_id=0)

        async_reset = async_session.reset(seed=29)

        assert tuple(image.key for image in async_reset.observation.images) == (
            "counter-0",
        )
        assert async_reset.turn is not None
        async_image_keys = tuple(
            part.image.key
            for part in async_reset.turn.messages[0].content
            if isinstance(part, ImageMessagePart)
        )
        assert async_image_keys == ("counter-0", "decorative")

        async_submission = async_session.submit("1")

        assert tuple(
            image.key for image in async_submission.step_result.observation.images
        ) == ("counter-1",)
        assert async_submission.turn is not None
        async_step_image_keys = tuple(
            part.image.key
            for part in async_submission.turn.messages[0].content
            if isinstance(part, ImageMessagePart)
        )
        assert async_step_image_keys == ("counter-1", "decorative")


def test_workflow_session_preserves_multimodal_turns() -> None:
    env = TurnBasedEnv(
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
    session = WorkflowSession(env=env)

    reset_result = session.reset(seed=13)

    assert len(reset_result.observation.images) == 1
    assert reset_result.turn is not None
    assert len(reset_result.turn.observation.images) == 1
    initial_text_part = reset_result.turn.messages[0].content[0]
    initial_image_part = reset_result.turn.messages[0].content[1]
    assert isinstance(initial_text_part, TextMessagePart)
    assert isinstance(initial_image_part, ImageMessagePart)
    assert initial_text_part.text == "Observation:\nvalue=0"
    assert initial_image_part.image.key == "counter-0"

    submission = session.submit("1")

    assert submission.turn is not None
    assert len(submission.step_result.observation.images) == 1
    next_text_part = submission.turn.messages[0].content[0]
    next_image_part = submission.turn.messages[0].content[1]
    assert isinstance(next_text_part, TextMessagePart)
    assert isinstance(next_image_part, ImageMessagePart)
    assert next_text_part.text == "Observation:\nvalue=1"
    assert next_image_part.image.key == "counter-1"


def test_workflow_session_requires_reset_before_use() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = WorkflowSession(env=env)

    with pytest.raises(EnvironmentNotResetError):
        _ = session.current_observation
    with pytest.raises(EnvironmentNotResetError):
        _ = session.reset_info
    with pytest.raises(EnvironmentNotResetError):
        _ = session.turn
    with pytest.raises(EnvironmentNotResetError):
        session.submit("1")


def test_workflow_session_rejects_non_string_extracted_action() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = WorkflowSession(
        env=env,
        action_extractor=cast(
            Callable[[str], str],
            lambda assistant_output: 1,
        ),
    )
    session.reset(seed=9)

    with pytest.raises(TypeError, match="must return a string action"):
        session.submit("1")


def test_async_backed_workflow_session_does_not_expose_trajectory() -> None:
    with AsyncEnvPool(env_factories=(_build_async_counter_env,)) as pool:
        session = pool.session(slot_id=0)
        session.reset(seed=11)

        with pytest.raises(AttributeError):
            getattr(session, "trajectory")


def test_async_workflow_session_exclusively_leases_one_slot() -> None:
    with AsyncEnvPool(env_factories=(_build_async_counter_env,)) as pool:
        session = pool.session(slot_id=0)

        with pytest.raises(RuntimeError, match="leased"):
            pool.session(slot_id=0)
        with pytest.raises(RuntimeError, match="leased"):
            pool.reset(slot_id=0, seed=3)
        with pytest.raises(RuntimeError, match="leased"):
            pool.recv_slot(slot_id=0, timeout_seconds=0.0)

        session.close()
        replacement_session = pool.session(slot_id=0)

        with pytest.raises(RuntimeError, match="not owned"):
            session.reset(seed=3)

        reset_result = replacement_session.reset(seed=3)

        assert reset_result.reset_info["seed"] == 3


def test_async_workflow_session_close_rejects_in_flight_work() -> None:
    with AsyncEnvPool(env_factories=(_build_async_slow_counter_env,)) as pool:
        session = pool.session(slot_id=0)
        reset_results = []
        thread_errors: list[BaseException] = []

        def _run_reset() -> None:
            try:
                reset_results.append(session.reset(seed=31))
            except BaseException as exc:  # pragma: no cover - defensive capture
                thread_errors.append(exc)

        reset_thread = Thread(target=_run_reset)
        reset_thread.start()

        deadline = time.monotonic() + 5.0
        while 0 not in pool._busy_slot_ids:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "Timed out waiting for the leased slot to go busy."
                )
            time.sleep(0.01)

        assert pool.pending_slot_ids == ()
        with pytest.raises(RuntimeError, match="in-flight or unread work"):
            session.close()

        reset_thread.join()

        assert thread_errors == []
        assert reset_results[0].reset_info["seed"] == 31

        session.close()


def test_async_workflow_session_rejects_slots_with_unread_buffered_results() -> None:
    with AsyncEnvPool(
        env_factories=(_build_async_counter_env, _build_async_slow_counter_env)
    ) as pool:
        pool.reset(slot_id=0, seed=4)
        pool.reset(slot_id=1, seed=5)

        slot_one_result = pool.recv_slot(slot_id=1, timeout_seconds=5.0)

        assert isinstance(slot_one_result, AsyncResetResult)
        assert slot_one_result.reset_info["seed"] == 5

        with pytest.raises(RuntimeError, match="unread buffered result"):
            pool.session(slot_id=0)
