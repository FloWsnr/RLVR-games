"""TRL dataset, reward-function, and environment adapter helpers."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from datasets import Dataset
import trl

from rlvr_physics.adapters.datasets import (
    PromptRow,
    completion_to_text,
    score_text_completion,
)
from rlvr_physics.adapters.multiturn import ScalarSessionEnvironment
from rlvr_physics.core.factory import TaskFactory
from rlvr_physics.core.instances import TaskInstance


class TrlRewardFunction:
    """Callable TRL reward function backed by immutable task instances.

    Parameters
    ----------
    instances:
        Immutable task instances keyed by task id.
    task_factory:
        Factory that creates scalar sessions for the configured task family.
    seed:
        Deterministic session seed used for reward scoring.
    """

    __name__ = "rlvr_physics_reward"

    def __init__(
        self,
        instances: Mapping[str, TaskInstance],
        task_factory: TaskFactory,
        seed: int,
    ) -> None:
        self._instances = dict(instances)
        self._task_factory = task_factory
        self._seed = seed

    def __call__(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        **row_columns: object,
    ) -> list[float | None]:
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
        rewards: list[float | None] = []
        for task_id, completion in zip(task_ids, completions, strict=True):
            instance = self._instances[task_id]
            score = score_text_completion(
                instance=instance,
                completion=completion_to_text(completion),
                task_factory=self._task_factory,
                seed=self._seed,
            )
            rewards.append(score.reward)
        return rewards


def _ensure_trl_dependencies() -> None:
    """Confirm that TRL dependencies are importable."""

    _ = trl


def to_trl_row(row: PromptRow) -> dict[str, Any]:
    """Convert a generic prompt row to a TRL dataset row.

    Parameters
    ----------
    row:
        Generic prompt row produced from an immutable task instance.
    """

    data = row.as_dict()
    data["task_id"] = row.task_id
    return data


def make_trl_dataset(rows: Sequence[PromptRow]) -> Dataset:
    """Build a Hugging Face ``Dataset`` for TRL.

    Parameters
    ----------
    rows:
        Prompt rows to expose to TRL.
    """

    _ensure_trl_dependencies()
    return Dataset.from_list([to_trl_row(row) for row in rows])


class TrlTaskEnvironment:
    """TRL tool-calling environment backed by one scalar task session.

    Parameters
    ----------
    instances:
        Immutable task instances keyed by task id.
    task_factory:
        Factory that creates scalar sessions for the configured task family.
    seed:
        Deterministic session seed used for resets.
    """

    def __init__(
        self,
        instances: Mapping[str, TaskInstance],
        task_factory: TaskFactory,
        seed: int,
    ) -> None:
        self._environment = ScalarSessionEnvironment(instances, task_factory, seed)

    def reset(self, **row_columns: object) -> str:
        """Reset this environment from one TRL dataset row.

        Parameters
        ----------
        row_columns:
            Dataset row columns containing a task id.
        """

        observation = self._environment.reset_from_mapping(row_columns)
        if observation:
            return "\n\n" + observation
        return observation

    def submit_action(self, action: str) -> str:
        """Submit one task action and return observation feedback.

        Args:
            action: Raw task action to submit to the current scalar task.

        Returns:
            Text feedback with acceptance, reward, score, done state, reason,
            and the next observation when the task continues.
        """

        return self._environment.submit_action_text(action)

    @property
    def step_rewards(self) -> tuple[float, ...]:
        """Return one scalar reward per submitted environment step."""

        return self._environment.step_rewards

    @property
    def total_reward(self) -> float:
        """Return the sum of environment step rewards."""

        return self._environment.total_reward

    @property
    def done(self) -> bool:
        """Return whether this environment has ended."""

        return self._environment.done


class TrlDenseRewardFunction:
    """TRL reward function that reads step rewards from environments.

    Parameters
    ----------
    fallback_reward:
        Reward returned for completions that do not have an environment.
    """

    __name__ = "rlvr_physics_dense_reward"

    def __init__(self, fallback_reward: float = 0.0) -> None:
        self._fallback_reward = fallback_reward

    def __call__(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        **row_columns: object,
    ) -> list[float | None]:
        """Return one environment reward per TRL completion.

        Parameters
        ----------
        prompts:
            Prompt payloads supplied by TRL.
        completions:
            Completion payloads supplied by TRL.
        row_columns:
            Reward context. When TRL uses ``environment_factory``, this includes
            ``environments``.
        """

        _ = prompts
        environments = row_columns.get("environments")
        if isinstance(environments, Sequence):
            rewards: list[float | None] = []
            for environment, _completion in zip(environments, completions, strict=True):
                if isinstance(environment, TrlTaskEnvironment):
                    rewards.append(environment.total_reward)
                else:
                    rewards.append(self._fallback_reward)
            return rewards
        return [self._fallback_reward for _completion in completions]


def to_trl_multiturn_row(row: PromptRow) -> dict[str, Any]:
    """Convert a generic prompt row to a TRL tool-calling row.

    Parameters
    ----------
    row:
        Generic prompt row produced from an immutable task instance.
    """

    data = to_trl_row(row)
    data["prompt"] = [
        {
            "role": "user",
            "content": (
                "Play this task by calling submit_action with one action per "
                "turn. The environment reset will provide the current state."
            ),
        }
    ]
    return data


def make_trl_multiturn_dataset(rows: Sequence[PromptRow]) -> Dataset:
    """Build a Hugging Face ``Dataset`` for TRL environment rollouts.

    Parameters
    ----------
    rows:
        Prompt rows to expose to TRL.
    """

    _ensure_trl_dependencies()
    return Dataset.from_list([to_trl_multiturn_row(row) for row in rows])


def make_trl_environment_factory(
    instances: Mapping[str, TaskInstance],
    task_factory: TaskFactory,
    seed: int,
) -> Callable[[], TrlTaskEnvironment]:
    """Return a TRL ``environment_factory`` for scalar task sessions.

    Parameters
    ----------
    instances:
        Immutable task instances keyed by task id.
    task_factory:
        Factory that creates scalar sessions for the configured task family.
    seed:
        Deterministic session seed used for resets.
    """

    registry = dict(instances)

    def _factory() -> TrlTaskEnvironment:
        return TrlTaskEnvironment(registry, task_factory, seed)

    return _factory


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
