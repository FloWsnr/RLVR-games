"""Tests for the cart inference JSONL interaction runner."""

from base64 import b64decode
from collections.abc import Mapping
from io import StringIO
import json
from typing import Any

import pytest

from rlvr_physics.core.rendering import PNG_MIME_TYPE, PNG_SIGNATURE
from rlvr_physics.play.interaction import (
    INTERACTION_PROTOCOL_VERSION,
    run_task_interaction,
)
from rlvr_physics.play.task import build_playable_interaction_config
from rlvr_physics.tasks.physics.cart_inference.play import (
    CART_PLAYABLE,
    cart_inference_config_from_parameters,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import CART_IMAGE_RENDERER


def test_cart_interaction_runs_multiple_turns_without_debug_leaks() -> None:
    """The runner emits reset and step events without privileged metadata."""

    input_stream = StringIO(
        "\n".join(
            [
                '{"action": "measure_position", "arguments": {"time": 5}}',
                '{"action": "final_answer", "arguments": {"x": 0}}',
            ]
        )
    )
    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_cart_interaction(
        instance_seed=123,
        session_seed=456,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 0
    assert error_stream.getvalue() == ""
    assert [event["event"] for event in events] == ["reset", "step", "step"]
    assert events[0]["protocol"] == INTERACTION_PROTOCOL_VERSION
    assert events[0]["turn"]["submission_format"]["required_fields"] == [
        "action",
        "arguments",
    ]
    assert events[0]["turn"]["submission_format"]["examples"][0] == {
        "action": "measure_position",
        "arguments": {"time": 10.0},
    }
    assert "invalid_submission_policy" not in events[0]["turn"]["submission_format"]
    assert sorted(events[0]["turn"]["invalid_submission_policies"]) == [
        "budget_exceeded",
        "invalid_final_answer",
        "retryable_invalid_submission",
    ]
    assert events[0]["turn"]["invalid_submission_policies"][
        "retryable_invalid_submission"
    ]["consumes_budget"] == {"turns": 1}
    assert events[0]["turn"]["action_schema"]["actions"]["measure_position"][
        "consumes_budget"
    ] == {"turns": 1, "actions": 1}
    assert events[0]["turn"]["action_schema"]["actions"]["final_answer"][
        "consumes_budget"
    ] == {"turns": 1, "final_answers": 1}
    assert events[0]["turn"]["public_limits"]["budget_limits"]["actions"] == 3
    assert events[0]["turn"]["public_limits"]["budget_limits"]["final_answers"] == 1
    assert "measurement_budget" not in events[0]["turn"]["public_limits"]
    assert events[1]["accepted"]
    assert events[1]["public_info"]["budget_usage"] == {
        "turns": 1,
        "actions": 1,
        "final_answers": 0,
    }
    assert events[1]["public_info"]["budget_remaining"] == {
        "turns": 3,
        "actions": 2,
        "final_answers": 1,
    }
    assert events[1]["reward_info"] == {
        "accepted_action": "measure_position",
        "measurement_index": 0,
        "reward_event": "accepted_action",
    }
    assert not events[1]["done"]
    assert events[2]["done"]
    assert events[2]["terminal"]
    assert events[2]["reward_info"] == {
        "correct": False,
        "reward_event": "final_answer",
    }
    assert "turn" in events[1]
    assert "turn" not in events[2]

    serialized = output_stream.getvalue()
    assert "debug_info" not in serialized
    assert "acceleration_mps2" not in serialized
    assert "exact_target_position_m" not in serialized
    assert "measurement_noise_seed" not in serialized
    assert "true_position_m" not in serialized
    assert "noise_m" not in serialized
    assert "instance_seed" not in serialized
    assert "rollout_seed" not in serialized


def test_cart_interaction_reports_incomplete_input() -> None:
    """Closing stdin before a terminal step returns a nonzero status."""

    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_cart_interaction(
        instance_seed=123,
        session_seed=456,
        input_stream=StringIO(""),
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 1
    assert error_stream.getvalue() == "input closed before rollout completed\n"
    assert [event["event"] for event in events] == ["reset"]


def test_cart_interaction_rejects_string_measurement_time() -> None:
    """String numeric arguments do not consume measurement budget."""

    input_stream = StringIO(
        "\n".join(
            [
                '{"action": "measure_position", "arguments": {"time": "NaN"}}',
                '{"action": "final_answer", "arguments": {"x": 0}}',
            ]
        )
    )
    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_cart_interaction(
        instance_seed=123,
        session_seed=456,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 0
    assert not events[1]["accepted"]
    assert events[1]["public_info"]["actions_used"] == 0
    assert events[1]["public_info"]["actions_remaining"] == 3
    assert events[1]["public_info"]["submissions_used"] == 1
    assert "time must be numeric" in events[1]["turn"]["observation"]["text"]


def test_cart_interaction_parse_errors_consume_turn_but_not_action_budget() -> None:
    """Malformed action envelopes are task-visible invalid submissions."""

    input_stream = StringIO(
        "\n".join(
            [
                '{"measure_position": {"time": 10}}',
                '{"action": "measure_position", "arguments": {"time": 10}}',
                '{"action": "final_answer", "arguments": {"x": 0}}',
            ]
        )
    )
    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_cart_interaction(
        instance_seed=123,
        session_seed=456,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 0
    assert not events[1]["accepted"]
    assert "expected one JSON line" in events[1]["public_info"]["reason"]
    assert events[1]["public_info"]["invalid_submission_category"] == (
        "malformed_transport"
    )
    assert (
        events[1]["public_info"]["invalid_submission_policy"]
        == "retryable_invalid_submission"
    )
    assert events[1]["public_info"]["submissions_used"] == 1
    assert events[1]["public_info"]["invalid_submissions"] == 1
    assert events[1]["public_info"]["actions_remaining"] == 3
    assert events[3]["done"]
    assert events[3]["terminal"]


@pytest.mark.parametrize(
    "payload",
    [
        {"parsed": {"action": "measure_position", "arguments": {"time": 5}}},
        {"raw": ('{"action": "measure_position", "arguments": {"time": 5}}')},
    ],
)
def test_cart_interaction_rejects_public_wrapper_action_bypass(
    payload: dict[str, object],
) -> None:
    """JSONL wrapper fields cannot bypass the canonical action envelope."""

    input_stream = StringIO(
        "\n".join(
            [
                json.dumps(payload, sort_keys=True),
                '{"action": "final_answer", "arguments": {"x": 0}}',
            ]
        )
    )
    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_cart_interaction(
        instance_seed=123,
        session_seed=456,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 0
    assert not events[1]["accepted"]
    assert events[1]["public_info"]["invalid_submission_category"] == (
        "malformed_transport"
    )
    assert events[1]["public_info"]["actions_used"] == 0
    assert events[1]["public_info"]["actions_remaining"] == 3
    assert events[1]["public_info"]["budget_usage"]["actions"] == 0
    assert events[2]["done"]
    assert events[2]["terminal"]


def test_cart_interaction_rejects_legacy_function_call_actions() -> None:
    """Cart accepts the canonical JSON action envelope only."""

    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_cart_interaction(
        instance_seed=123,
        session_seed=456,
        input_stream=StringIO("measure_position(10)\n"),
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 1
    assert not events[1]["accepted"]
    assert events[1]["public_info"]["invalid_submission_category"] == (
        "unparseable_action"
    )
    assert (
        events[1]["public_info"]["invalid_submission_policy"]
        == "retryable_invalid_submission"
    )
    assert events[1]["public_info"]["submissions_used"] == 1


def test_cart_interaction_final_answer_format_error_uses_final_attempt() -> None:
    """Invalid final-answer arguments follow task policy, not CLI policy."""

    input_stream = StringIO('{"action": "final_answer", "arguments": {"x": "NaN"}}')
    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_cart_interaction(
        instance_seed=123,
        session_seed=456,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 0
    assert not events[1]["accepted"]
    assert events[1]["done"]
    assert events[1]["terminal"]
    assert events[1]["reward"] == 0.0
    assert events[1]["public_info"]["invalid_submission_category"] == (
        "invalid_final_answer"
    )
    assert events[1]["public_info"]["invalid_submission_policy"] == (
        "invalid_final_answer"
    )
    assert events[1]["public_info"]["final_answers_used"] == 1
    assert events[1]["public_info"]["final_answers_remaining"] == 0


def test_cart_image_interaction_serializes_image_content() -> None:
    """Image renderer observations include JSONL image payload fields."""

    output_stream = StringIO()
    error_stream = StringIO()
    config = build_playable_interaction_config(
        playable=CART_PLAYABLE,
        parameters=CART_PLAYABLE.default_parameters,
        renderer_type=CART_IMAGE_RENDERER,
        instance_seed=123,
        session_seed=456,
    )

    status_code = run_task_interaction(
        config=config,
        input_stream=StringIO(""),
        output_stream=output_stream,
        error_stream=error_stream,
    )
    reset_event = _jsonl_events(output_stream)[0]
    image_content = reset_event["turn"]["observation"]["contents"][0]

    assert status_code == 1
    assert image_content["kind"] == "image"
    assert image_content["mime_type"] == PNG_MIME_TYPE
    assert len(image_content["sha256"]) == 64
    assert b64decode(image_content["data_base64"]).startswith(PNG_SIGNATURE)
    assert reset_event["turn"]["observation"]["text"] != ""


def test_cart_play_parameters_reject_unknown_keys() -> None:
    """Typoed public parameter overrides fail instead of being ignored."""

    parameters = dict(CART_PLAYABLE.default_parameters)
    parameters["max_tunrs"] = 1

    with pytest.raises(ValueError, match="max_tunrs"):
        cart_inference_config_from_parameters(parameters)


def test_cart_play_parameters_accept_reward_config() -> None:
    """Public play parameters preserve task reward policy settings."""

    parameters = dict(CART_PLAYABLE.default_parameters)
    reward_parameter = parameters["reward"]
    if not isinstance(reward_parameter, Mapping):
        raise TypeError("reward parameter must be a mapping")
    reward = dict(reward_parameter)
    reward["accepted_measurement_reward"] = 0.125
    reward["invalid_submission_reward"] = -0.25
    parameters["reward"] = reward

    config = cart_inference_config_from_parameters(parameters)

    assert config.reward.accepted_measurement_reward == 0.125
    assert config.reward.invalid_submission_reward == -0.25


def run_cart_interaction(
    instance_seed: int,
    session_seed: int,
    input_stream: StringIO,
    output_stream: StringIO,
    error_stream: StringIO,
) -> int:
    """Run the cart playable through the generic interaction runner.

    Parameters
    ----------
    instance_seed:
        Private seed used to build the immutable task instance.
    session_seed:
        Seed used to reset the scalar session.
    input_stream:
        Stream containing one model submission per line.
    output_stream:
        Stream receiving one public JSON event per line.
    error_stream:
        Stream receiving process-level errors that are not task observations.

    Returns
    -------
    int
        Process-style status code.
    """

    config = build_playable_interaction_config(
        playable=CART_PLAYABLE,
        parameters=CART_PLAYABLE.default_parameters,
        renderer_type=CART_PLAYABLE.default_renderer,
        instance_seed=instance_seed,
        session_seed=session_seed,
    )
    return run_task_interaction(
        config=config,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )


def _jsonl_events(output_stream: StringIO) -> list[dict[str, Any]]:
    """Parse JSONL events from an in-memory output stream.

    Parameters
    ----------
    output_stream:
        Stream containing one JSON object per line.

    Returns
    -------
    list of dict[str, Any]
        Parsed JSONL event payloads.
    """

    return [
        json.loads(line)
        for line in output_stream.getvalue().splitlines()
        if line.strip() != ""
    ]
