"""Tests for shared JSONL interaction helpers."""

from io import StringIO
import json
from typing import Any

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.play.interaction import (
    DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS,
    INTERACTION_PROTOCOL_VERSION,
    TaskInteractionConfig,
    decode_jsonl_submission,
    run_task_interaction,
    write_jsonl_event,
)
from rlvr_physics.core.rendering import text_observation
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskTurn,
)
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.core.submissions import TaskSubmission


class CompletingSession:
    """Minimal terminal session used by interaction tests."""

    def __init__(self, instance: TaskInstance) -> None:
        """Initialize the test session.

        Parameters
        ----------
        instance:
            Immutable task instance.
        """

        self._instance = instance
        self._turn: TaskTurn | None = None

    def reset(self, seed: int) -> TaskResetResult:
        """Start the test rollout.

        Parameters
        ----------
        seed:
            Deterministic session seed.

        Returns
        -------
        TaskResetResult
            Initial test turn and metadata.
        """

        self._turn = TaskTurn(
            turn_index=0,
            observation=text_observation("test.text", "test prompt"),
            submission_modes=("final_text",),
            submission_format={},
            action_schema={},
            invalid_submission_policies={},
            public_limits=self._instance.public_limits(),
            public_info={"task_id": self._instance.task_id},
        )
        return TaskResetResult(
            session_id="test-session",
            turn=self._turn,
            public_info={
                "kind": self._instance.kind,
                "task_id": self._instance.task_id,
                "rollout_seed": seed,
                "visible": "public",
            },
            debug_info={"secret": 42},
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current test turn."""

        return self._turn

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Accept one submission and terminate.

        Parameters
        ----------
        submission:
            Model submission decoded by the interaction layer.

        Returns
        -------
        TaskStepResult
            Terminal result containing public echo metadata.
        """

        self._turn = None
        return TaskStepResult(
            accepted=True,
            reward_result=RewardResult(reward=1.0, score=1.0),
            terminal=True,
            truncated=False,
            observation=None,
            public_info={
                "received_kind": submission.kind,
                "received_raw": submission.raw,
            },
            debug_info={"secret": "hidden"},
        )


def test_task_interaction_filters_public_info_and_omits_debug() -> None:
    """The generic runner emits public reset and terminal step events."""

    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_task_interaction(
        config=TaskInteractionConfig(
            task=_configured_task(),
            instance_seed=8,
            session_seed=9,
            public_info_excluded_keys=DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS,
        ),
        input_stream=StringIO('{"kind": "final_text", "text": "answer"}\n'),
        output_stream=output_stream,
        error_stream=error_stream,
    )

    events = _jsonl_events(output_stream)

    assert status_code == 0
    assert error_stream.getvalue() == ""
    assert events[0]["protocol"] == INTERACTION_PROTOCOL_VERSION
    assert events[0]["public_info"] == {
        "kind": "tests.interaction.v1",
        "visible": "public",
    }
    assert events[1]["public_info"] == {
        "received_kind": "final_text",
        "received_raw": "answer",
    }
    assert "debug_info" not in output_stream.getvalue()
    assert "secret" not in output_stream.getvalue()
    assert "rollout_seed" not in output_stream.getvalue()
    assert "task_id" not in output_stream.getvalue()


def test_decode_jsonl_submission_supports_action_mapping_and_raw_action() -> None:
    """JSON action objects and raw action strings both decode as actions."""

    mapped = decode_jsonl_submission('{"action": "measure_position", "time": 5}')
    raw = decode_jsonl_submission("measure_position(5)")

    assert mapped.kind == "action"
    assert mapped.parsed["action"] == "measure_position"
    assert mapped.parsed["time"] == 5
    assert raw.kind == "action"
    assert raw.raw == "measure_position(5)"


def test_decode_jsonl_submission_does_not_unwrap_public_wrapper_fields() -> None:
    """Public JSONL input cannot override the preserved raw or parsed payload."""

    wrapped_payload = {
        "parsed": {
            "action": "measure_position",
            "arguments": {"time": 5},
        }
    }
    wrapped_line = json.dumps(wrapped_payload, sort_keys=True)
    wrapped = decode_jsonl_submission(wrapped_line)
    embedded_raw = json.dumps(
        {"action": "measure_position", "arguments": {"time": 5}},
        sort_keys=True,
    )
    raw_payload = {"raw": embedded_raw}
    raw_line = json.dumps(raw_payload, sort_keys=True)
    raw_wrapped = decode_jsonl_submission(raw_line)

    assert wrapped.kind == "action"
    assert wrapped.raw == wrapped_line
    assert wrapped.parsed["parsed"] == wrapped_payload["parsed"]
    assert "action" not in wrapped.parsed
    assert raw_wrapped.kind == "action"
    assert raw_wrapped.raw == raw_line
    assert raw_wrapped.parsed["raw"] == embedded_raw


def test_decode_jsonl_submission_treats_non_standard_json_constants_as_raw() -> None:
    """JSON constants such as NaN are not parsed into structured payloads."""

    raw_line = '{"action": "measure_position", "time": NaN}'

    submission = decode_jsonl_submission(raw_line)

    assert submission.kind == "action"
    assert submission.raw == raw_line
    assert submission.parsed["action"] == raw_line


def test_write_jsonl_event_rejects_non_finite_numbers() -> None:
    """The JSONL writer refuses non-standard JSON numeric values."""

    with pytest.raises(ValueError, match="Out of range float values"):
        write_jsonl_event(StringIO(), {"value": float("nan")})


def _configured_task() -> ConfiguredTask:
    """Return a configured task backed by the completing test session."""

    return ConfiguredTask(
        spec=TaskSpec(
            kind="tests.interaction.v1",
            domain="tests",
            source=SourceSpec(source_type="tests.interaction", seed=0),
            renderers=(RendererSpec(renderer_type="test.text"),),
            verifier=VerifierSpec(verifier_type="tests.interaction"),
            reward=RewardSpec(reward_type="tests.interaction", parameters={}),
            budget_limits={"turns": 1},
        ),
        instance_builder=_build_instance,
        session_builder=CompletingSession,
    )


def _build_instance(seed: int) -> TaskInstance:
    """Build an immutable test instance.

    Parameters
    ----------
    seed:
        Deterministic instance seed.

    Returns
    -------
    TaskInstance
        Immutable test instance.
    """

    return TaskInstance(
        task_id=f"interaction-test-{seed}",
        kind="tests.interaction.v1",
        domain="tests",
        seed=seed,
        public_payload={},
        privileged_payload={},
        budget_limits={"turns": 1},
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
