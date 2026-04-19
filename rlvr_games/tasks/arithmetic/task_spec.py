"""Task-spec parsing and session construction for arithmetic verifier tasks."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictInt

from rlvr_games.core.session import TaskSessionProtocol
from rlvr_games.core.task_spec_base import (
    TaskSpec,
    TaskSpecModel,
    validate_task_spec_model,
)
from rlvr_games.tasks.arithmetic.core import (
    ArithmeticOperation,
    ArithmeticTaskSource,
    make_arithmetic_session,
)


@dataclass(slots=True)
class ArithmeticTaskSpec(TaskSpec):
    """Validated authored arithmetic verifier task specification.

    Attributes
    ----------
    min_value : int
        Inclusive lower bound for sampled operands.
    max_value : int
        Inclusive upper bound for sampled operands.
    operations : tuple[ArithmeticOperation, ...]
        Operation set used by the deterministic task source.
    """

    min_value: int = -20
    max_value: int = 20
    operations: tuple[ArithmeticOperation, ...] = field(
        default_factory=lambda: (
            ArithmeticOperation.ADD,
            ArithmeticOperation.SUBTRACT,
            ArithmeticOperation.MULTIPLY,
        )
    )

    @property
    def kind(self) -> str:
        """Return the neutral task kind carried by this task spec."""
        return "arithmetic"

    def __post_init__(self) -> None:
        """Validate arithmetic task-spec fields."""
        TaskSpec.__post_init__(self)
        ArithmeticTaskSource(
            min_value=self.min_value,
            max_value=self.max_value,
            operations=self.operations,
        )


class _ArithmeticYamlModel(BaseModel):
    """Base model for authored arithmetic YAML fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArithmeticSourceModel(_ArithmeticYamlModel):
    """Authored arithmetic source configuration."""

    min_value: StrictInt = -20
    max_value: StrictInt = 20
    operations: tuple[ArithmeticOperation, ...] = (
        ArithmeticOperation.ADD,
        ArithmeticOperation.SUBTRACT,
        ArithmeticOperation.MULTIPLY,
    )


class ArithmeticTaskSpecModel(TaskSpecModel):
    """Authored top-level arithmetic task specification."""

    kind: Literal["arithmetic"] = "arithmetic"
    source: ArithmeticSourceModel | None = None

    def to_runtime(self) -> ArithmeticTaskSpec:
        """Convert the authored model into the runtime task spec."""
        source = self.source
        if source is None:
            source = ArithmeticSourceModel()
        return ArithmeticTaskSpec(
            schema_version=self.schema_version,
            task_id=self.task_id,
            episode_config=self.episode_config(),
            metadata=self.metadata,
            min_value=source.min_value,
            max_value=source.max_value,
            operations=source.operations,
        )


def arithmetic_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> ArithmeticTaskSpec:
    """Parse an arithmetic task specification from a raw mapping."""
    task_spec = validate_task_spec_model(
        model_type=ArithmeticTaskSpecModel,
        payload=payload,
        base_dir=base_dir,
    )
    return task_spec.to_runtime()


def build_arithmetic_session_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TaskSessionProtocol:
    """Build one arithmetic task session from a validated task specification."""
    if not isinstance(task_spec, ArithmeticTaskSpec):
        raise TypeError(
            "build_arithmetic_session_from_task_spec requires ArithmeticTaskSpec."
        )
    return make_arithmetic_session(
        task_source=ArithmeticTaskSource(
            min_value=task_spec.min_value,
            max_value=task_spec.max_value,
            operations=task_spec.operations,
        )
    )


def build_arithmetic_session_factory_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> Callable[[], TaskSessionProtocol]:
    """Build a fresh-session factory from an arithmetic task spec."""
    if not isinstance(task_spec, ArithmeticTaskSpec):
        raise TypeError(
            "build_arithmetic_session_factory_from_task_spec requires "
            "ArithmeticTaskSpec."
        )
    return partial(build_arithmetic_session_from_task_spec, task_spec=task_spec)


__all__ = [
    "ArithmeticTaskSpec",
    "ArithmeticTaskSpecModel",
    "build_arithmetic_session_factory_from_task_spec",
    "build_arithmetic_session_from_task_spec",
    "arithmetic_task_spec_from_mapping",
]
