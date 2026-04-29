"""Tests for shared submission parsing helpers."""

import pytest

from rlvr_physics.core.submissions import (
    ACTION_ARGUMENTS_FIELD,
    ACTION_NAME_FIELD,
    TaskSubmission,
    parse_action_envelope,
    parse_action_submission,
    parse_json_object,
)


def test_parse_action_envelope_accepts_canonical_shape() -> None:
    parsed = parse_action_envelope(
        {
            ACTION_NAME_FIELD: "measure_position",
            ACTION_ARGUMENTS_FIELD: {"time": 5},
        }
    )

    assert parsed is not None
    assert parsed.name == "measure_position"
    assert parsed.arguments["time"] == 5
    with pytest.raises(TypeError):
        parsed.arguments["time"] = 6  # type: ignore[index]


def test_parse_action_envelope_rejects_missing_arguments() -> None:
    parsed = parse_action_envelope({ACTION_NAME_FIELD: "measure_position"})

    assert parsed is None


def test_parse_action_submission_falls_back_to_raw_json() -> None:
    submission = TaskSubmission(
        kind="action",
        raw='{"action": "final_answer", "arguments": {"x": 1.5}}',
        parsed={},
    )

    parsed = parse_action_submission(submission)

    assert parsed is not None
    assert parsed.name == "final_answer"
    assert parsed.arguments["x"] == 1.5


def test_parse_json_object_rejects_non_standard_constants() -> None:
    assert parse_json_object('{"x": NaN}') is None
    assert parse_json_object("[1, 2, 3]") is None
