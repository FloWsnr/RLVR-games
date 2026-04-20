"""Tests for verl adapter helpers."""

from pathlib import Path

import pyarrow.parquet as pq
import verl

from rlvr_physics.adapters.datasets import (
    make_instance_registry,
    make_prompt_dataset_row,
)
from rlvr_physics.adapters.verl import (
    VerlRewardFunction,
    to_verl_row,
    write_verl_parquet,
)
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.tasks.games.countdown import CountdownSession, make_countdown_instance


def _countdown_text_session(instance: TaskInstance) -> TaskSession:
    return CountdownSession(instance, "text")


def test_verl_row_parquet_and_reward_function(tmp_path: Path) -> None:
    assert verl is not None
    instance = make_countdown_instance(seed=17, source_index=0)
    row = make_prompt_dataset_row(
        instance=instance,
        session_factory=_countdown_text_session,
        seed=3,
        extra_info={"split": "train", "ability": "countdown"},
    )
    verl_row = to_verl_row(row)
    parquet_path = tmp_path / "countdown.parquet"
    write_verl_parquet((row,), parquet_path)
    reward_function = VerlRewardFunction(
        instances=make_instance_registry((instance,)),
        session_factory=_countdown_text_session,
        seed=4,
    )
    reward = reward_function(
        data_source=verl_row["data_source"],
        solution_str=str(instance.privileged_payload["reference_expression"]),
        ground_truth=verl_row["reward_model"]["ground_truth"],
        extra_info=verl_row["extra_info"],
    )

    table = pq.read_table(parquet_path)

    assert verl_row["data_source"] == instance.kind
    assert verl_row["ability"] == "countdown"
    assert verl_row["reward_model"]["ground_truth"] == instance.task_id
    assert "reference_expression" not in repr(verl_row)
    assert table.num_rows == 1
    assert reward == 1.0
