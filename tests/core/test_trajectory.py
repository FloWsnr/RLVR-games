"""Tests for trajectory records."""

from typing import Any

import pytest

from rlvr_physics.core.trajectory import TaskTrajectory


def test_trajectory_appends_and_snapshots_events() -> None:
    trajectory = TaskTrajectory(task_id="task-1", session_id="session-1")

    event = trajectory.append("reset", 0, {"renderer": "text"}, {"seed": 3})

    assert trajectory.snapshot() == (event,)
    assert trajectory.events == (event,)
    assert event.public["renderer"] == "text"
    assert event.debug["seed"] == 3

    events: Any = trajectory.events
    with pytest.raises(AttributeError):
        events.clear()
    assert trajectory.snapshot() == (event,)
