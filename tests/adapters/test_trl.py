"""Tests for TRL adapter helpers."""

from datasets import Dataset

from rlvr_physics.adapters.datasets import (
    make_instance_registry,
    make_prompt_dataset_row,
)
from rlvr_physics.adapters.trl import TrlRewardFunction, make_trl_dataset, to_trl_row
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.tasks.games.countdown import CountdownSession, make_countdown_instance


def _countdown_text_session(instance: TaskInstance) -> TaskSession:
    return CountdownSession(instance, "text")


def test_trl_dataset_row_and_reward_function() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    row = make_prompt_dataset_row(
        instance=instance,
        session_factory=_countdown_text_session,
        seed=3,
        extra_info={"split": "train", "ability": "countdown"},
    )

    trl_row = to_trl_row(row)
    dataset = make_trl_dataset((row,))
    completion = [{"role": "assistant", "content": "answer: 1 + 2"}]
    reward_function = TrlRewardFunction(
        instances=make_instance_registry((instance,)),
        session_factory=_countdown_text_session,
        seed=4,
    )
    rewards = reward_function(
        prompts=[row.prompt],
        completions=[completion],
        task_id=[row.task_id],
    )

    assert dataset.num_rows == 1
    assert isinstance(dataset, Dataset)
    assert trl_row["prompt"] == row.prompt
    assert trl_row["task_id"] == instance.task_id
    assert rewards == [0.05]
