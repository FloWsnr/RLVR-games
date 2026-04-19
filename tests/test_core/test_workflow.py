"""Workflow session tests."""

from collections.abc import Callable
from typing import cast

from PIL import Image
import pytest

from rlvr_games.core import (
    ActionContext,
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
    CounterBackend,
    CounterReward,
    CounterScenario,
    CounterState,
    inspect_counter_state,
    make_counter_env,
)


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
