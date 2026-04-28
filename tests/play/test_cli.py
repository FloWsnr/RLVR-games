"""Tests for the generic play command line interface."""

from io import StringIO
import json
from typing import Any

import pytest

from rlvr_physics.play.cli import run_play_cli


def test_play_cli_lists_registered_tasks() -> None:
    """The generic CLI lists playable task names and aliases."""

    output_stream = StringIO()

    status_code = run_play_cli(
        argv=("--list",),
        input_stream=StringIO(""),
        output_stream=output_stream,
        error_stream=StringIO(),
    )

    assert status_code == 0
    assert output_stream.getvalue().splitlines() == [
        "cart_inference",
        "physics.cart_inference",
    ]


def test_play_cli_requires_task_or_list() -> None:
    """Running play without a task exits as an invocation error."""

    with pytest.raises(SystemExit) as error:
        run_play_cli(
            argv=(),
            input_stream=StringIO(""),
            output_stream=StringIO(),
            error_stream=StringIO(),
        )

    assert error.value.code == 2


def test_play_cli_runs_cart_alias() -> None:
    """The generic CLI can run cart inference by short alias."""

    output_stream = StringIO()
    error_stream = StringIO()

    status_code = run_play_cli(
        argv=(
            "cart_inference",
            "--instance-seed",
            "123",
            "--session-seed",
            "456",
        ),
        input_stream=StringIO('{"action": "final_answer", "x": 0}\n'),
        output_stream=output_stream,
        error_stream=error_stream,
    )
    events = _jsonl_events(output_stream)

    assert status_code == 0
    assert error_stream.getvalue() == ""
    assert [event["event"] for event in events] == ["reset", "step"]
    assert events[1]["done"]


def test_play_cli_passes_help_to_selected_task(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Task-specific help shows task options rather than generic play help."""

    try:
        run_play_cli(
            argv=("cart_inference", "--help"),
            input_stream=StringIO(""),
            output_stream=StringIO(),
            error_stream=StringIO(),
        )
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()

    assert "usage: play cart_inference" in captured.out
    assert "--instance-seed" in captured.out
    assert "--renderer" in captured.out


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
