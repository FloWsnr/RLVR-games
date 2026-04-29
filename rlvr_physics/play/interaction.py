"""Public JSONL interaction protocol for scalar task sessions."""

from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import TextIO

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.payloads import mapping_to_dict, to_plain_data
from rlvr_physics.core.rendering import ImageContent, TextContent
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskTurn,
)
from rlvr_physics.core.submissions import (
    TaskSubmission,
    invalid_submission_policies_payload,
    parse_json_object,
)

INTERACTION_PROTOCOL_VERSION = "rlvr.physics.interaction.v1"
DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS = frozenset(
    {
        "instance_hash",
        "instance_seed",
        "rollout_seed",
        "seed",
        "task_id",
    }
)


@dataclass(frozen=True)
class TaskInteractionConfig:
    """Configuration for one task interaction process.

    Parameters
    ----------
    task:
        Configured task used to build the private instance and scalar session.
    instance_seed:
        Private seed used to build the immutable task instance.
    session_seed:
        Seed used to reset the scalar session.
    public_info_excluded_keys:
        ``public_info`` keys omitted from public interaction events.

    Attributes
    ----------
    task:
        Configured task used to build the private instance and scalar session.
    instance_seed:
        Private seed used to build the immutable task instance.
    session_seed:
        Seed used to reset the scalar session.
    public_info_excluded_keys:
        ``public_info`` keys omitted from public interaction events.
    """

    task: ConfiguredTask
    instance_seed: int
    session_seed: int
    public_info_excluded_keys: frozenset[str]


def run_task_interaction(
    config: TaskInteractionConfig,
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
) -> int:
    """Run one multi-turn task over stdin/stdout style streams.

    Parameters
    ----------
    config:
        Interaction configuration.
    input_stream:
        Stream containing one model submission per line.
    output_stream:
        Stream receiving one public JSON event per line.
    error_stream:
        Stream receiving process-level errors that are not task observations.

    Returns
    -------
    int
        Process-style status code. ``0`` means the rollout completed; ``1``
        means the submission stream ended before the rollout completed.
    """

    instance = config.task.build_instance(seed=config.instance_seed)
    session = config.task.create_session(instance)
    reset = session.reset(seed=config.session_seed)
    write_jsonl_event(
        output_stream,
        reset_interaction_event(
            reset=reset,
            public_info_excluded_keys=config.public_info_excluded_keys,
        ),
    )

    for line in input_stream:
        raw_line = line.strip()
        if raw_line == "":
            continue

        result = session.submit(decode_jsonl_submission(raw_line))
        write_jsonl_event(
            output_stream,
            step_interaction_event(
                result=result,
                public_info_excluded_keys=config.public_info_excluded_keys,
            ),
        )
        if result.done:
            return 0

    error_stream.write("input closed before rollout completed\n")
    return 1


def decode_jsonl_submission(raw_line: str) -> TaskSubmission:
    """Decode one JSONL input line into a task submission.

    Parameters
    ----------
    raw_line:
        Raw non-empty line read from an interaction input stream.

    Returns
    -------
    TaskSubmission
        Submission decoded from either a JSON object or a raw action string.
    """

    decoded = parse_json_object(raw_line)
    if decoded is None:
        return TaskSubmission.action(raw_line)

    kind = _submission_kind(decoded)
    return TaskSubmission(
        kind=kind,
        raw=_submission_raw(decoded, raw_line, kind),
        parsed=decoded,
    )


def reset_interaction_event(
    reset: TaskResetResult, public_info_excluded_keys: frozenset[str]
) -> dict[str, object]:
    """Return the public JSON event for a reset result.

    Parameters
    ----------
    reset:
        Scalar session reset result.
    public_info_excluded_keys:
        ``public_info`` keys omitted from the event.

    Returns
    -------
    dict[str, object]
        Public reset event with the initial turn.
    """

    return {
        "protocol": INTERACTION_PROTOCOL_VERSION,
        "event": "reset",
        "session_id": reset.session_id,
        "done": False,
        "turn": turn_payload(reset.turn, public_info_excluded_keys),
        "public_info": filtered_public_info(
            reset.public_info, public_info_excluded_keys
        ),
    }


def step_interaction_event(
    result: TaskStepResult, public_info_excluded_keys: frozenset[str]
) -> dict[str, object]:
    """Return the public JSON event for a step result.

    Parameters
    ----------
    result:
        Scalar session step result.
    public_info_excluded_keys:
        ``public_info`` keys omitted from the event.

    Returns
    -------
    dict[str, object]
        Public step event with reward, done flags, and optional next turn.
    """

    event: dict[str, object] = {
        "protocol": INTERACTION_PROTOCOL_VERSION,
        "event": "step",
        "accepted": result.accepted,
        "reward": result.reward,
        "score": result.score,
        "terminal": result.terminal,
        "truncated": result.truncated,
        "done": result.done,
        "public_info": filtered_public_info(
            result.public_info, public_info_excluded_keys
        ),
    }
    if result.observation is not None:
        event["turn"] = turn_payload(result.observation, public_info_excluded_keys)
    return event


def turn_payload(
    turn: TaskTurn, public_info_excluded_keys: frozenset[str]
) -> dict[str, object]:
    """Return the public JSON payload for one model-facing turn.

    Parameters
    ----------
    turn:
        Model-facing task turn.
    public_info_excluded_keys:
        ``public_info`` keys omitted from the turn payload.

    Returns
    -------
    dict[str, object]
        Plain JSON-compatible turn payload.
    """

    return {
        "turn_index": turn.turn_index,
        "observation": observation_payload(turn),
        "submission_modes": list(turn.submission_modes),
        "submission_format": to_plain_data(turn.submission_format),
        "action_schema": to_plain_data(turn.action_schema),
        "invalid_submission_policies": invalid_submission_policies_payload(
            turn.invalid_submission_policies
        ),
        "public_limits": to_plain_data(turn.public_limits),
        "public_info": filtered_public_info(
            turn.public_info, public_info_excluded_keys
        ),
    }


def observation_payload(turn: TaskTurn) -> dict[str, object]:
    """Return the public observation payload for a turn.

    Parameters
    ----------
    turn:
        Model-facing task turn.

    Returns
    -------
    dict[str, object]
        Plain JSON-compatible observation payload.
    """

    contents: list[dict[str, object]] = []
    for content in turn.observation.contents:
        if isinstance(content, TextContent):
            contents.append({"kind": "text", "text": content.text})
        elif isinstance(content, ImageContent):
            contents.append(
                {
                    "kind": "image",
                    "mime_type": content.mime_type,
                    "alt_text": content.alt_text,
                    "sha256": content.digest(),
                    "data_base64": b64encode(content.data).decode("ascii"),
                }
            )

    return {
        "renderer": turn.observation.renderer_name,
        "text": turn.observation.text(),
        "contents": contents,
    }


def filtered_public_info(
    public_info: Mapping[str, object], excluded_keys: frozenset[str]
) -> dict[str, object]:
    """Return public info with selected top-level keys omitted.

    Parameters
    ----------
    public_info:
        Trainer-safe metadata emitted by a session.
    excluded_keys:
        Top-level keys omitted from the interaction event.

    Returns
    -------
    dict[str, object]
        Plain JSON-compatible public metadata.
    """

    filtered = mapping_to_dict(public_info)
    for key in excluded_keys:
        filtered.pop(key, None)
    return filtered


def write_jsonl_event(output_stream: TextIO, event: Mapping[str, object]) -> None:
    """Write one JSONL protocol event and flush the stream.

    Parameters
    ----------
    output_stream:
        Stream receiving JSONL events.
    event:
        Public event payload.
    """

    output_stream.write(json.dumps(event, allow_nan=False, sort_keys=True))
    output_stream.write("\n")
    output_stream.flush()


def _submission_kind(values: Mapping[str, object]) -> str:
    """Return the submission kind requested by an input mapping."""

    kind_value = values.get("kind")
    if not isinstance(kind_value, str):
        kind_value = values.get("submission_kind")
    if isinstance(kind_value, str):
        return kind_value
    return "action"


def _submission_raw(
    values: Mapping[str, object], raw_line: str, submission_kind: str
) -> str:
    """Return the raw submission text for an input mapping."""

    text_value = values.get("text")
    if submission_kind == "final_text" and isinstance(text_value, str):
        return text_value
    return raw_line
