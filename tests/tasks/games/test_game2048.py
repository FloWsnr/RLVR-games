"""Tests for seeded 2048 behavior."""

from rlvr_physics.core.rendering import ImageContent
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.games.game2048 import (
    Game2048Session,
    game2048_task_spec,
    initial_2048_state,
    legal_2048_actions,
    make_2048_instance,
    move_board,
    render_2048_image,
    render_2048_text,
)


def test_2048_instance_spec_and_initial_state_are_deterministic() -> None:
    first = make_2048_instance(seed=5, max_turns=4, target_tile=64)
    second = make_2048_instance(seed=5, max_turns=4, target_tile=64)
    spec = game2048_task_spec(seed=5, max_turns=4, target_tile=64)
    state = initial_2048_state(first)

    assert first.task_id == second.task_id
    assert (
        first.public_payload["initial_board"] == second.public_payload["initial_board"]
    )
    assert spec.kind == first.kind
    assert state.board == ((2, 0, 0, 0), (0, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0))
    assert state.spawn_index == 2


def test_2048_move_merge_rules() -> None:
    board = ((2, 2, 2, 2), (4, 0, 4, 0), (0, 0, 0, 0), (2, 4, 8, 16))

    left = move_board(board, "left")
    right = move_board(board, "right")

    assert left.board[0] == (4, 4, 0, 0)
    assert left.board[1] == (8, 0, 0, 0)
    assert left.score_delta == 16
    assert right.board[0] == (0, 0, 4, 4)
    assert right.board[1] == (0, 0, 0, 8)
    assert right.score_delta == 16


def test_2048_text_and_image_renderers() -> None:
    instance = make_2048_instance(seed=5, max_turns=4, target_tile=64)
    state = initial_2048_state(instance)

    text = render_2048_text(state, target_tile=64)
    image = render_2048_image(state, target_tile=64)

    assert "Submit one action" in text.text()
    assert isinstance(image.contents[0], ImageContent)
    assert image.contents[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    assert image.text() == text.text()


def test_2048_invalid_action_does_not_advance_state() -> None:
    instance = make_2048_instance(seed=5, max_turns=4, target_tile=64)
    session = Game2048Session(instance, "text")
    reset = session.reset(seed=1)

    result = session.submit(TaskSubmission.action("left"))

    assert result.accepted is False
    assert result.reward == -1.0
    assert result.observation is reset.turn
    assert session.turn is reset.turn
    assert result.public_info["reason"] == "move_did_not_change_board"
    assert result.public_info["submissions"] == 1


def test_2048_invalid_actions_consume_submission_budget() -> None:
    instance = make_2048_instance(seed=5, max_turns=1, target_tile=64)
    session = Game2048Session(instance, "text")
    session.reset(seed=1)

    result = session.submit(TaskSubmission.action("left"))

    assert result.accepted is False
    assert result.truncated
    assert result.observation is None
    assert session.turn is None


def test_2048_valid_action_updates_score_and_can_truncate() -> None:
    instance = make_2048_instance(seed=5, max_turns=1, target_tile=2048)
    session = Game2048Session(instance, "text")
    session.reset(seed=1)

    result = session.submit(TaskSubmission.action("up"))

    assert result.accepted
    assert result.reward == 4.0
    assert result.truncated
    assert not result.terminal
    assert result.observation is None
    assert result.public_info["score"] == 4
    assert "up" in legal_2048_actions(initial_2048_state(instance).board)


def test_2048_continuing_step_returns_all_emitted_events() -> None:
    instance = make_2048_instance(seed=5, max_turns=4, target_tile=2048)
    session = Game2048Session(instance, "text")
    session.reset(seed=1)
    before_event_count = len(session.trajectory.events)

    result = session.submit(TaskSubmission.action("up"))
    emitted_events = session.trajectory.events[before_event_count:]

    assert not result.done
    assert [event.event_type for event in result.events] == [
        "submission",
        "transition",
        "observation",
    ]
    assert result.events == emitted_events
