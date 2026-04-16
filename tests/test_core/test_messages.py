"""Tests for observation-to-message adaptation."""

from dataclasses import dataclass

from PIL import Image

from rlvr_games.core import (
    ActionContext,
    AgentVisibleEvent,
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
    EpisodeConfig,
    ImageMessagePart,
    MessageRole,
    Observation,
    ProjectedActionContext,
    RenderedImage,
    TextMessagePart,
    build_action_context,
)

from tests.test_core.support import CounterBackend, CounterState, make_counter_env


@dataclass(slots=True)
class CounterOpeningProjector:
    """Project one fixed opening event into the action context."""

    def project_action_context(
        self,
        *,
        state: CounterState,
        reset_events: tuple[object, ...],
    ) -> ProjectedActionContext:
        """Return one human-readable opening event."""
        del state
        del reset_events
        return ProjectedActionContext(
            opening_events=(
                AgentVisibleEvent(
                    kind="opening_hint",
                    source="chance",
                    text="The counter begins at zero.",
                ),
            )
        )


def test_default_observation_message_adapter_emits_system_text_and_images() -> None:
    image = Image.new("RGB", (1, 1), color=(0, 0, 0))
    observation = Observation(
        text="Current board state",
        images=(RenderedImage(key="board", image=image),),
        metadata={"value": 7},
    )
    adapter = DefaultObservationMessageAdapter(
        policy=DefaultObservationMessagePolicy(
            system_prompt_text="You are playing a board game.",
            action_reminder_text="Respond with one legal action.",
            include_turn_index=True,
            metadata_formatter=lambda metadata: f"Metadata value: {metadata['value']}",
        )
    )
    action_context = ActionContext(
        turn_index=2,
        opening_events=(
            AgentVisibleEvent(
                kind="opening_move",
                source="opponent",
                text="Opponent opened in the center.",
            ),
        ),
    )

    messages = adapter.to_messages(
        observation=observation,
        action_context=action_context,
    )

    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER

    system_text_part = messages[0].content[0]
    user_text_part = messages[1].content[0]
    image_part = messages[1].content[1]
    assert isinstance(system_text_part, TextMessagePart)
    assert isinstance(user_text_part, TextMessagePart)
    assert isinstance(image_part, ImageMessagePart)
    assert system_text_part.text == "You are playing a board game."
    assert user_text_part.text == (
        "Turn: 2\n\n"
        "Opening events:\n"
        "- Opponent opened in the center.\n\n"
        "Observation:\n"
        "Current board state\n\n"
        "Metadata value: 7\n\n"
        "Respond with one legal action."
    )
    assert image_part.image.key == "board"

    image.putpixel((0, 0), (255, 0, 0))
    assert image_part.image.image.getpixel((0, 0)) == (0, 0, 0)


def test_messages_for_observation_uses_supplied_action_context() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
        agent_context_projector=CounterOpeningProjector(),
        observation_message_adapter=DefaultObservationMessageAdapter(
            policy=DefaultObservationMessagePolicy(
                system_prompt_text="Play the counter game.",
                action_reminder_text="Reply with one integer delta.",
                include_turn_index=True,
            )
        ),
    )

    initial_observation, _ = env.reset(seed=3)
    initial_action_context = build_action_context(env=env)
    initial_messages = env.messages_for_observation(
        initial_observation,
        action_context=initial_action_context,
    )
    next_result = env.step("1")
    next_action_context = build_action_context(env=env)
    next_messages = env.messages_for_observation(
        next_result.observation,
        action_context=next_action_context,
    )

    initial_text_part = initial_messages[1].content[0]
    next_text_part = next_messages[1].content[0]
    assert isinstance(initial_text_part, TextMessagePart)
    assert isinstance(next_text_part, TextMessagePart)
    assert "Turn: 0" in initial_text_part.text
    assert "The counter begins at zero." in initial_text_part.text
    assert "Turn: 1" in next_text_part.text


def test_messages_for_observation_preserves_past_turn_context() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
        agent_context_projector=CounterOpeningProjector(),
        observation_message_adapter=DefaultObservationMessageAdapter(
            policy=DefaultObservationMessagePolicy(
                action_reminder_text="Reply with one integer delta.",
                include_turn_index=True,
            )
        ),
    )

    initial_observation, _ = env.reset(seed=3)
    initial_action_context = build_action_context(env=env)
    env.step("1")

    initial_messages = env.messages_for_observation(
        initial_observation,
        action_context=initial_action_context,
    )

    initial_text_part = initial_messages[0].content[0]
    assert isinstance(initial_text_part, TextMessagePart)
    assert "Turn: 0" in initial_text_part.text
    assert "The counter begins at zero." in initial_text_part.text


def test_counter_helper_installs_default_message_adapter() -> None:
    env = make_counter_env(backend=CounterBackend(), config=EpisodeConfig())

    observation, _ = env.reset(seed=5)
    messages = env.messages_for_observation(
        observation,
        action_context=build_action_context(env=env),
    )

    text_part = messages[0].content[0]
    assert env.observation_message_adapter is not None
    assert isinstance(text_part, TextMessagePart)
    assert text_part.text == "Observation:\nvalue=0"
