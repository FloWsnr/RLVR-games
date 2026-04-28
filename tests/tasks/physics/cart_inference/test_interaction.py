"""Tests for the cart inference JSONL interaction runner."""

from base64 import b64decode
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
                '{"action": "measure_position", "time": 5}',
                '{"action": "final_answer", "x": 0}',
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
    assert events[1]["accepted"]
    assert not events[1]["done"]
    assert events[2]["done"]
    assert events[2]["terminal"]
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


def test_cart_interaction_rejects_non_finite_measurement_time() -> None:
    """Non-finite numeric arguments do not consume measurement budget."""

    input_stream = StringIO(
        "\n".join(
            [
                '{"action": "measure_position", "time": "NaN"}',
                '{"action": "final_answer", "x": 0}',
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
    assert events[1]["public_info"]["measurements_used"] == 0
    assert events[1]["public_info"]["measurements_remaining"] == 3
    assert "time must be finite" in events[1]["turn"]["observation"]["text"]


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
