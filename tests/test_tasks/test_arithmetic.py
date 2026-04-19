"""Arithmetic task tests."""

import pytest

from rlvr_games.core import EpisodeFinishedError, TextMessagePart
from rlvr_games.tasks.arithmetic import (
    ArithmeticOperation,
    ArithmeticTaskSource,
    ArithmeticVerifier,
    make_arithmetic_session,
)


def test_arithmetic_task_source_samples_deterministically() -> None:
    source = ArithmeticTaskSource(
        min_value=1,
        max_value=3,
        operations=(ArithmeticOperation.ADD,),
    )

    first_task = source.sample(seed=123)
    second_task = source.sample(seed=123)

    assert first_task.instance.task_instance_id == second_task.instance.task_instance_id
    assert first_task.payload == second_task.payload
    assert first_task.payload.operation == ArithmeticOperation.ADD
    assert first_task.instance.task_kind == "arithmetic"


def test_arithmetic_session_reset_renders_prompt() -> None:
    source = ArithmeticTaskSource(
        min_value=2,
        max_value=2,
        operations=(ArithmeticOperation.MULTIPLY,),
    )
    session = make_arithmetic_session(task_source=source)

    reset_result = session.reset(seed=0)

    assert reset_result.turn is not None
    assert reset_result.observation is not None
    assert reset_result.observation.text is not None
    assert "2 * 2" in reset_result.observation.text
    text_part = reset_result.turn.messages[0].content[0]
    assert isinstance(text_part, TextMessagePart)
    assert "2 * 2" in text_part.text
    assert reset_result.info["task_kind"] == "arithmetic"
    assert session.trajectory.initial_turn is reset_result.turn


def test_arithmetic_session_rewards_correct_answer() -> None:
    source = ArithmeticTaskSource(
        min_value=2,
        max_value=2,
        operations=(ArithmeticOperation.MULTIPLY,),
    )
    session = make_arithmetic_session(task_source=source)
    session.reset(seed=0)

    result = session.submit("4")

    assert result.valid_submission is True
    assert result.parsed_output == 4
    assert result.reward == 1.0
    assert result.info == {"valid_submission": True, "correct": True}
    assert result.debug_info == {"expected": 4}
    assert result.done is True
    assert session.done is True
    assert session.trajectory.total_reward == 1.0


def test_arithmetic_session_wrong_answer_is_valid_with_zero_reward() -> None:
    source = ArithmeticTaskSource(
        min_value=2,
        max_value=2,
        operations=(ArithmeticOperation.MULTIPLY,),
    )
    session = make_arithmetic_session(task_source=source)
    session.reset(seed=0)

    result = session.submit("5")

    assert result.valid_submission is True
    assert result.parsed_output == 5
    assert result.reward == 0.0
    assert result.info == {"valid_submission": True, "correct": False}
    assert result.debug_info == {"expected": 4}


def test_arithmetic_session_parses_final_integer_from_completion() -> None:
    source = ArithmeticTaskSource(
        min_value=2,
        max_value=2,
        operations=(ArithmeticOperation.ADD,),
    )
    session = make_arithmetic_session(task_source=source)
    session.reset(seed=0)

    result = session.submit("The result is 3, so final answer: 4")

    assert result.valid_submission is True
    assert result.parsed_output == 4
    assert result.reward == 1.0


def test_arithmetic_session_malformed_completion_is_invalid() -> None:
    source = ArithmeticTaskSource(
        min_value=2,
        max_value=2,
        operations=(ArithmeticOperation.ADD,),
    )
    session = make_arithmetic_session(task_source=source)
    session.reset(seed=0)

    result = session.submit("I do not know")

    assert result.valid_submission is False
    assert result.parsed_output is None
    assert result.reward == 0.0
    assert result.info == {"valid_submission": False, "reason": "no_integer_found"}
    assert result.debug_info == {"expected": 4}


def test_arithmetic_session_rejects_second_submission() -> None:
    source = ArithmeticTaskSource(
        min_value=2,
        max_value=2,
        operations=(ArithmeticOperation.ADD,),
    )
    session = make_arithmetic_session(task_source=source)
    session.reset(seed=0)
    session.submit("4")

    with pytest.raises(EpisodeFinishedError):
        session.submit("4")


def test_arithmetic_verifier_exposes_expected_answer_only_in_debug_info() -> None:
    source = ArithmeticTaskSource(
        min_value=2,
        max_value=2,
        operations=(ArithmeticOperation.ADD,),
    )
    task = source.sample(seed=0)

    result = ArithmeticVerifier().verify(task=task, completion="4")

    assert "expected" not in result.info
    assert result.debug_info == {"expected": 4}
