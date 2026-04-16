"""Message adapters for turning observations into trainer-facing chat turns."""

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Protocol

from rlvr_games.core.action_context import ActionContext, AgentVisibleEvent
from rlvr_games.core.types import Observation, RenderedImage


class MessageRole(StrEnum):
    """Supported roles for adapter-produced chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class TextMessagePart:
    """One text content part inside a chat message.

    Attributes
    ----------
    text : str
        Text payload to expose in the message.
    """

    text: str

    def __post_init__(self) -> None:
        """Validate the text payload."""
        if not isinstance(self.text, str):
            raise TypeError("TextMessagePart text must be a string.")
        if not self.text:
            raise ValueError("TextMessagePart text must be non-empty.")


@dataclass(slots=True)
class ImageMessagePart:
    """One image content part inside a chat message.

    Attributes
    ----------
    image : RenderedImage
        Image payload to attach to the message.
    alt_text : str | None
        Optional textual description carried alongside the image payload.
    """

    image: RenderedImage
    alt_text: str | None = None

    def __post_init__(self) -> None:
        """Validate the image payload and detach its raster data."""
        if not isinstance(self.image, RenderedImage):
            raise TypeError("ImageMessagePart image must be a RenderedImage.")
        if self.alt_text is not None and not isinstance(self.alt_text, str):
            raise TypeError("ImageMessagePart alt_text must be a string or None.")
        self.image = self.image.copy()


MessagePart = TextMessagePart | ImageMessagePart


@dataclass(slots=True)
class ChatMessage:
    """One structured chat message derived from an observation.

    Attributes
    ----------
    role : MessageRole
        Role associated with the message.
    content : tuple[MessagePart, ...]
        Ordered content parts contained in the message.
    """

    role: MessageRole
    content: tuple[MessagePart, ...]

    def __post_init__(self) -> None:
        """Validate message role and content parts."""
        if not isinstance(self.role, MessageRole):
            raise TypeError("ChatMessage role must be a MessageRole.")
        if not self.content:
            raise ValueError("ChatMessage content must contain at least one part.")
        for part in self.content:
            if not isinstance(part, TextMessagePart | ImageMessagePart):
                raise TypeError(
                    "ChatMessage content must contain only text or image parts."
                )


class ObservationMessagePolicy(Protocol):
    """Policy for rendering trainer-facing text around an observation."""

    def system_prompt(self) -> str | None:
        """Return the optional system prompt for the adapted message stream.

        Returns
        -------
        str | None
            System prompt text to prepend before the observation turn, or
            `None` when no system message should be emitted.
        """
        ...

    def format_observation_text(
        self,
        *,
        observation: Observation,
        action_context: ActionContext,
    ) -> str | None:
        """Return the text content for the current observation turn.

        Parameters
        ----------
        observation : Observation
            Observation to adapt into a trainer-facing turn.
        action_context : ActionContext
            Structured agent-visible context for the next action choice.

        Returns
        -------
        str | None
            Text content to attach to the observation message, or `None` when
            the message should be image-only.
        """
        ...


def _format_default_opening_event(event: AgentVisibleEvent) -> str:
    """Return a compact default rendering for one opening event.

    Parameters
    ----------
    event : AgentVisibleEvent
        Opening event projected into the action context.

    Returns
    -------
    str
        Human-readable opening event summary.
    """
    if event.text is not None:
        return event.text
    return f"{event.source}: {event.kind}"


@dataclass(slots=True)
class DefaultObservationMessagePolicy:
    """General-purpose text policy for observation message adaptation.

    Attributes
    ----------
    system_prompt_text : str | None
        Optional system prompt to prepend before the observation turn.
    action_reminder_text : str | None
        Optional action-format reminder appended after the observation text.
    observation_label : str | None
        Optional section label placed ahead of `Observation.text`.
    opening_events_label : str | None
        Optional header used when projected opening events are present.
    include_turn_index : bool
        Whether to prepend the zero-based next-turn index.
    metadata_formatter : Callable[[dict[str, Any]], str | None] | None
        Optional callback that renders observation metadata into extra text.
    opening_event_formatter : Callable[[AgentVisibleEvent], str] | None
        Optional callback that renders one projected opening event.
    """

    system_prompt_text: str | None = None
    action_reminder_text: str | None = None
    observation_label: str | None = "Observation"
    opening_events_label: str | None = "Opening events"
    include_turn_index: bool = False
    metadata_formatter: Callable[[dict[str, Any]], str | None] | None = None
    opening_event_formatter: Callable[[AgentVisibleEvent], str] | None = None

    def system_prompt(self) -> str | None:
        """Return the optional system prompt.

        Returns
        -------
        str | None
            Configured system prompt text, or `None` when disabled.
        """
        return self.system_prompt_text

    def format_observation_text(
        self,
        *,
        observation: Observation,
        action_context: ActionContext,
    ) -> str | None:
        """Compose the default text payload for an observation turn.

        Parameters
        ----------
        observation : Observation
            Observation to adapt into user-facing text.
        action_context : ActionContext
            Structured context for the next action turn.

        Returns
        -------
        str | None
            Joined text sections for the message, or `None` when no textual
            section should be emitted.
        """
        sections: list[str] = []
        if self.include_turn_index:
            sections.append(f"Turn: {action_context.turn_index}")

        if action_context.opening_events:
            format_event = self.opening_event_formatter
            if format_event is None:
                format_event = _format_default_opening_event
            opening_lines = [
                f"- {format_event(event)}" for event in action_context.opening_events
            ]
            opening_block = "\n".join(opening_lines)
            if self.opening_events_label is not None:
                opening_block = f"{self.opening_events_label}:\n{opening_block}"
            sections.append(opening_block)

        if observation.text is not None:
            observation_text = observation.text
            if self.observation_label is not None:
                observation_text = f"{self.observation_label}:\n{observation_text}"
            sections.append(observation_text)

        if self.metadata_formatter is not None:
            metadata_text = self.metadata_formatter(deepcopy(observation.metadata))
            if metadata_text is not None:
                sections.append(metadata_text)

        if self.action_reminder_text is not None:
            sections.append(self.action_reminder_text)

        if not sections:
            return None
        return "\n\n".join(sections)


class ObservationMessageAdapter(Protocol):
    """Protocol for adapting observations into structured chat messages."""

    def to_messages(
        self,
        *,
        observation: Observation,
        action_context: ActionContext,
    ) -> tuple[ChatMessage, ...]:
        """Convert an observation and action context into chat messages.

        Parameters
        ----------
        observation : Observation
            Observation to adapt into messages.
        action_context : ActionContext
            Structured context for the next action choice.

        Returns
        -------
        tuple[ChatMessage, ...]
            Ordered chat messages ready for downstream trainer serialization.
        """
        ...


@dataclass(slots=True)
class DefaultObservationMessageAdapter:
    """Default adapter that wraps one observation into system and user turns.

    Attributes
    ----------
    policy : ObservationMessagePolicy
        Policy used to render the textual parts of the message stream.
    observation_role : MessageRole
        Role used for the observation-carrying message.
    image_alt_text_factory : Callable[[RenderedImage], str | None] | None
        Optional callback that generates alt text for each image part.
    """

    policy: ObservationMessagePolicy
    observation_role: MessageRole = MessageRole.USER
    image_alt_text_factory: Callable[[RenderedImage], str | None] | None = None

    def to_messages(
        self,
        *,
        observation: Observation,
        action_context: ActionContext,
    ) -> tuple[ChatMessage, ...]:
        """Convert one observation into structured chat messages.

        Parameters
        ----------
        observation : Observation
            Observation to adapt into messages.
        action_context : ActionContext
            Structured context for the next action choice.

        Returns
        -------
        tuple[ChatMessage, ...]
            Ordered system and observation messages derived from the supplied
            observation.

        Raises
        ------
        ValueError
            If the adapted observation would contain neither text nor images.
        """
        messages: list[ChatMessage] = []
        system_prompt = self.policy.system_prompt()
        if system_prompt is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(TextMessagePart(text=system_prompt),),
                )
            )

        content: list[MessagePart] = []
        observation_text = self.policy.format_observation_text(
            observation=observation,
            action_context=action_context,
        )
        if observation_text is not None:
            content.append(TextMessagePart(text=observation_text))

        for image in observation.images:
            alt_text = None
            if self.image_alt_text_factory is not None:
                alt_text = self.image_alt_text_factory(image)
            content.append(ImageMessagePart(image=image, alt_text=alt_text))

        if not content:
            raise ValueError(
                "Observation message adaptation requires text or images to emit."
            )

        messages.append(
            ChatMessage(
                role=self.observation_role,
                content=tuple(content),
            )
        )
        return tuple(messages)


__all__ = [
    "ChatMessage",
    "DefaultObservationMessageAdapter",
    "DefaultObservationMessagePolicy",
    "ImageMessagePart",
    "MessagePart",
    "MessageRole",
    "ObservationMessageAdapter",
    "ObservationMessagePolicy",
    "TextMessagePart",
]
