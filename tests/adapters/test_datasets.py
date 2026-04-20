"""Tests for generic adapter dataset helpers."""

from rlvr_physics.adapters.datasets import (
    completion_to_text,
    make_prompt_row,
    make_task_instance_registry,
    score_text_completion,
    task_id_from_mapping,
)
from rlvr_physics.core.factory import ConfiguredTaskFactory
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.tasks.games.countdown import (
    CountdownSession,
    countdown_task_spec,
    make_countdown_instance,
)


def _countdown_text_session(instance: TaskInstance) -> TaskSession:
    return CountdownSession(instance, "text")


def _countdown_factory() -> ConfiguredTaskFactory:
    return ConfiguredTaskFactory(
        spec=countdown_task_spec(seed=17, size=1),
        session_builder=_countdown_text_session,
    )


def test_prompt_row_is_public_and_scoreable() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    factory = _countdown_factory()
    row = make_prompt_row(
        instance=instance,
        task_factory=factory,
        seed=3,
        extra_info={"split": "train", "ability": "countdown"},
    )
    row_data = row.as_dict()

    assert row.task_id == instance.task_id
    assert "Target:" in row.prompt
    assert row_data["extra_info"]["split"] == "train"
    assert row_data["metadata"]["task_spec"]["reward"] == "graded_countdown"
    assert row_data["reward_model"]["style"] == "rlvr_executable"
    assert row_data["reward_model"]["reward_type"] == "graded_countdown"
    assert "reference_expression" not in repr(row_data)

    completion = str(instance.privileged_payload["reference_expression"])
    score = score_text_completion(
        instance=instance,
        completion=completion,
        task_factory=factory,
        seed=4,
    )

    assert score.reward == 1.0
    assert score.done
    assert score.public_info["reason"] == "correct"


def test_task_id_lookup_accepts_prompt_row_reward_model_shape() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    row = make_prompt_row(
        instance=instance,
        task_factory=_countdown_factory(),
        seed=3,
        extra_info={},
    )

    assert task_id_from_mapping({"reward_model": row.reward_model}) == instance.task_id


def test_instance_registry_rejects_duplicate_task_ids() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)

    try:
        make_task_instance_registry((instance, instance))
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
