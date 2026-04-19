"""Tests for Countdown task behavior."""

from typing import Mapping

from rlvr_physics.core.rendering import ImageContent
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.games.countdown import (
    CountdownSession,
    countdown_task_spec,
    make_countdown_instance,
    render_countdown_image,
    render_countdown_text,
    verify_countdown_submission,
)


def test_countdown_instance_is_deterministic_and_spec_is_public() -> None:
    first = make_countdown_instance(seed=17, source_index=0)
    second = make_countdown_instance(seed=17, source_index=0)
    spec = countdown_task_spec(seed=17, size=10)

    assert first.task_id == second.task_id
    assert first.public_payload == second.public_payload
    assert spec.kind == first.kind
    assert [renderer.renderer_type for renderer in spec.renderers] == ["text", "image"]
    payload = first.public_view()["payload"]
    assert isinstance(payload, Mapping)
    assert "reference_expression" not in payload


def test_countdown_text_and_image_renderers() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)

    text = render_countdown_text(instance)
    image = render_countdown_image(instance)

    assert "Target:" in text.text()
    assert str(instance.public_payload["target"]) in text.text()
    assert isinstance(image.contents[0], ImageContent)
    assert image.contents[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    assert image.text() == text.text()


def test_countdown_verifier_accepts_reference_expression() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    expression = str(instance.privileged_payload["reference_expression"])

    verification = verify_countdown_submission(
        instance, TaskSubmission.final_text(expression)
    )

    assert verification.accepted
    assert verification.correct
    assert verification.reward == 1.0
    assert verification.reason == "correct"


def test_countdown_verifier_distinguishes_invalid_numbers_and_value() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    numbers_payload = instance.public_payload["numbers"]
    assert isinstance(numbers_payload, tuple)
    numbers = tuple(int(value) for value in numbers_payload)
    wrong_value_expression = " + ".join(str(number) for number in numbers)

    wrong_numbers = verify_countdown_submission(
        instance, TaskSubmission.final_text("1 + 2")
    )
    wrong_value = verify_countdown_submission(
        instance, TaskSubmission.final_text(wrong_value_expression)
    )
    invalid = verify_countdown_submission(
        instance, TaskSubmission.final_text("__import__('os').system('echo nope')")
    )

    assert wrong_numbers.accepted
    assert wrong_numbers.reason == "wrong_numbers"
    assert wrong_value.accepted
    assert wrong_value.reason == "wrong_value"
    assert not invalid.accepted
    assert invalid.reason == "invalid_expression"


def test_countdown_sessions_are_independent_for_same_instance() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    expression = str(instance.privileged_payload["reference_expression"])
    first = CountdownSession(instance, "text")
    second = CountdownSession(instance, "text")

    first_reset = first.reset(seed=1)
    second_reset = second.reset(seed=1)
    first_result = first.submit(TaskSubmission.final_text(expression))
    second_result = second.submit(TaskSubmission.final_text("not an expression"))

    assert first_reset.session_id != second_reset.session_id
    assert first_result.reward == 1.0
    assert second_result.reward == 0.01
    assert first.turn is None
    assert second.turn is None
    assert [event.event_type for event in first.trajectory.snapshot()] == [
        "reset",
        "observation",
        "submission",
        "reward",
    ]
