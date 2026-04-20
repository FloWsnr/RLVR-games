"""verl row export and reward-function adapter helpers."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import verl

from rlvr_physics.adapters.datasets import (
    PromptDatasetRow,
    SessionFactory,
    score_final_text,
)
from rlvr_physics.core.instances import TaskInstance, mapping_to_dict


class VerlRewardFunction:
    """Callable verl reward function backed by immutable task instances.

    Parameters
    ----------
    instances:
        Immutable task instances keyed by task id.
    session_factory:
        Factory that creates a scalar session for one task instance.
    seed:
        Deterministic session seed used for reward scoring.
    """

    def __init__(
        self,
        instances: Mapping[str, TaskInstance],
        session_factory: SessionFactory,
        seed: int,
    ) -> None:
        self._instances = dict(instances)
        self._session_factory = session_factory
        self._seed = seed

    def __call__(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Mapping[str, object],
    ) -> float:
        """Return a scalar verl reward for one generated response.

        Parameters
        ----------
        data_source:
            Trainer data-source label. It is accepted for verl compatibility.
        solution_str:
            Model completion text.
        ground_truth:
            Opaque task id pointer stored in the verl reward model field.
        extra_info:
            Public row metadata containing ``task_id`` when available.
        """

        _ = data_source
        task_id = _task_id_from_verl_payload(ground_truth, extra_info)
        instance = self._instances[task_id]
        score = score_final_text(
            instance=instance,
            completion=solution_str,
            session_factory=self._session_factory,
            seed=self._seed,
        )
        return score.reward


def _ensure_verl_dependencies() -> None:
    """Confirm that verl dependencies are importable."""

    _ = verl


def to_verl_row(row: PromptDatasetRow) -> dict[str, Any]:
    """Convert a generic prompt row to a verl parquet-style row.

    Parameters
    ----------
    row:
        Generic prompt row produced from an immutable task instance.
    """

    extra_info = mapping_to_dict(row.extra_info)
    extra_info["id"] = row.task_id
    extra_info["task_id"] = row.task_id
    extra_info["task_kind"] = row.task_kind
    extra_info["renderer"] = row.renderer
    return {
        "data_source": row.task_kind,
        "prompt": [{"role": "user", "content": row.prompt}],
        "ability": str(extra_info.get("ability", row.domain)),
        "reward_model": {
            "style": "rlvr_executable",
            "ground_truth": row.task_id,
        },
        "extra_info": extra_info,
    }


def write_verl_parquet(rows: Sequence[PromptDatasetRow], path: Path) -> None:
    """Write verl rows to a parquet file.

    Parameters
    ----------
    rows:
        Prompt rows to export.
    path:
        Destination parquet path.
    """

    _ensure_verl_dependencies()
    table = pa.Table.from_pylist([to_verl_row(row) for row in rows])
    pq.write_table(table, path)


def _task_id_from_verl_payload(
    ground_truth: str, extra_info: Mapping[str, object]
) -> str:
    task_id = extra_info.get("task_id")
    if isinstance(task_id, str):
        return task_id
    if ground_truth:
        return ground_truth
    raise KeyError("verl reward payload must include task_id or ground_truth")
