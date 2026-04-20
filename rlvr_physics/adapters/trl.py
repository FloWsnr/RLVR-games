"""TRL dataset and reward-function adapter helpers."""

from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset
import trl

from rlvr_physics.adapters.datasets import (
    PromptDatasetRow,
    SessionFactory,
    completion_to_text,
    score_final_text,
)
from rlvr_physics.core.instances import TaskInstance


class TrlRewardFunction:
    """Callable TRL reward function backed by immutable task instances.

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
        prompts: Sequence[object],
        completions: Sequence[object],
        **row_columns: object,
    ) -> list[float]:
        """Return one reward per TRL completion.

        Parameters
        ----------
        prompts:
            Prompt payloads supplied by TRL. They are accepted for signature
            compatibility and are not authoritative.
        completions:
            Completion payloads supplied by TRL.
        row_columns:
            Dataset row columns, including ``task_id`` or ``id``.
        """

        _ = prompts
        task_ids = _task_ids_from_columns(row_columns, len(completions))
        rewards: list[float] = []
        for task_id, completion in zip(task_ids, completions, strict=True):
            instance = self._instances[task_id]
            score = score_final_text(
                instance=instance,
                completion=completion_to_text(completion),
                session_factory=self._session_factory,
                seed=self._seed,
            )
            rewards.append(score.reward)
        return rewards


def _ensure_trl_dependencies() -> None:
    """Confirm that TRL dependencies are importable."""

    _ = trl


def to_trl_row(row: PromptDatasetRow) -> dict[str, Any]:
    """Convert a generic prompt row to a TRL dataset row.

    Parameters
    ----------
    row:
        Generic prompt row produced from an immutable task instance.
    """

    data = row.as_dict()
    data["task_id"] = row.task_id
    return data


def make_trl_dataset(rows: Sequence[PromptDatasetRow]) -> Dataset:
    """Build a Hugging Face ``Dataset`` for TRL.

    Parameters
    ----------
    rows:
        Prompt rows to expose to TRL.
    """

    _ensure_trl_dependencies()
    return Dataset.from_list([to_trl_row(row) for row in rows])


def _task_ids_from_columns(
    row_columns: Mapping[str, object], expected_count: int
) -> list[str]:
    raw_task_ids = row_columns.get("task_id")
    if raw_task_ids is None:
        raw_task_ids = row_columns.get("id")
    if raw_task_ids is None:
        raise KeyError("TRL reward rows must include task_id or id")
    if isinstance(raw_task_ids, Sequence) and not isinstance(raw_task_ids, str | bytes):
        if len(raw_task_ids) != expected_count:
            raise ValueError("task id count must match completion count")
        return [str(task_id) for task_id in raw_task_ids]
    if expected_count != 1:
        raise ValueError("scalar task id can only score one completion")
    return [str(raw_task_ids)]
