"""Instance sampling for Countdown."""

from reasoning_gym.games.countdown import CountdownConfig, CountdownDataset

from rlvr_physics.core.instances import TaskInstance, TaskLimits, stable_hash
from rlvr_physics.tasks.games.countdown.constants import (
    COUNTDOWN_DOMAIN,
    COUNTDOWN_KIND,
)


def make_countdown_instance(seed: int, source_index: int) -> TaskInstance:
    """Sample one immutable Countdown task instance from Reasoning Gym.

    Parameters
    ----------
    seed:
        Procedural Reasoning Gym seed.
    source_index:
        Deterministic source index within the virtual dataset.
    """

    config = CountdownConfig(seed=seed, size=max(source_index + 1, 1))
    dataset = CountdownDataset(config)
    entry = dataset[source_index]
    metadata = entry["metadata"]
    numbers = tuple(int(number) for number in metadata["numbers"])
    target = int(metadata["target"])
    reference_expression = str(entry["answer"])
    task_id = (
        "countdown-"
        + stable_hash(
            {
                "seed": seed,
                "source_index": source_index,
                "numbers": numbers,
                "target": target,
                "reference_expression": reference_expression,
            }
        )[:16]
    )
    return TaskInstance(
        task_id=task_id,
        kind=COUNTDOWN_KIND,
        domain=COUNTDOWN_DOMAIN,
        seed=seed,
        public_payload={
            "numbers": numbers,
            "target": target,
            "question": str(entry["question"]),
        },
        privileged_payload={
            "reference_expression": reference_expression,
            "source_dataset": str(metadata["source_dataset"]),
            "source_index": source_index,
        },
        limits=TaskLimits(max_turns=1, token_budget=256),
        metadata={
            "source_dataset": str(metadata["source_dataset"]),
            "source_index": source_index,
        },
    )
