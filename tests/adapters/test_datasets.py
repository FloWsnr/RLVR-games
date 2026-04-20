"""Tests for generic adapter dataset helpers."""

from rlvr_physics.adapters.datasets import (
    completion_to_text,
    make_instance_registry,
    make_prompt_dataset_row,
    score_final_text,
)
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.tasks.games.countdown import CountdownSession, make_countdown_instance


def _countdown_text_session(instance: TaskInstance) -> TaskSession:
    return CountdownSession(instance, "text")


def test_prompt_dataset_row_is_public_and_scoreable() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    row = make_prompt_dataset_row(
        instance=instance,
        session_factory=_countdown_text_session,
        seed=3,
        extra_info={"split": "train", "ability": "countdown"},
    )
    row_data = row.as_dict()

    assert row.task_id == instance.task_id
    assert "Target:" in row.prompt
    assert row_data["extra_info"]["split"] == "train"
    assert row_data["reward_model"]["style"] == "rlvr_executable"
    assert "reference_expression" not in repr(row_data)

    completion = str(instance.privileged_payload["reference_expression"])
    score = score_final_text(
        instance=instance,
        completion=completion,
        session_factory=_countdown_text_session,
        seed=4,
    )

    assert score.reward == 1.0
    assert score.done
    assert score.public_info["reason"] == "correct"


def test_instance_registry_rejects_duplicate_task_ids() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)

    try:
        make_instance_registry((instance, instance))
    except ValueError as error:
        assert instance.task_id in str(error)
    else:
        raise AssertionError("duplicate task ids should fail")


def test_completion_to_text_accepts_chat_payloads() -> None:
    completion = [
        {"role": "assistant", "content": "thinking"},
        {"role": "assistant", "content": "answer: 1 + 2"},
    ]

    assert completion_to_text(completion) == "thinking\nanswer: 1 + 2"
