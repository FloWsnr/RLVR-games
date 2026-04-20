"""Types for physics discovery."""

from dataclasses import dataclass, field
from typing import Mapping

from rlvr_physics.core.instances import freeze_mapping


@dataclass(frozen=True)
class PhysicsDiscoveryRecord:
    """One source law used to build physics discovery instances.

    Parameters
    ----------
    source_id:
        Original PhysGym source identifier.
    tag:
        Coarse physics domain tag.
    context:
        Public problem context used in the richest prior mode.
    equation:
        Ground-truth scalar expression in terms of the input variables.
    input_variables:
        Mapping from variable name to public description.
    output_variable:
        Mapping with one output variable name and public description.
    """

    source_id: int
    tag: str
    context: str
    equation: str
    input_variables: Mapping[str, object]
    output_variable: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze record mappings after construction."""

        object.__setattr__(
            self, "input_variables", freeze_mapping(self.input_variables)
        )
        object.__setattr__(
            self, "output_variable", freeze_mapping(self.output_variable)
        )


@dataclass(frozen=True)
class ExperimentObservation:
    """Public result from one controlled experiment."""

    sample_id: int
    inputs: Mapping[str, object]
    output: float

    def __post_init__(self) -> None:
        """Freeze input mapping after construction."""

        object.__setattr__(self, "inputs", freeze_mapping(self.inputs))

    def as_public_dict(self) -> Mapping[str, object]:
        """Return trainer-safe observation data."""

        return freeze_mapping(
            {
                "sample_id": self.sample_id,
                "inputs": self.inputs,
                "output": self.output,
            }
        )


@dataclass(frozen=True)
class HypothesisAttempt:
    """Public result from one tested hypothesis."""

    hypothesis_id: int
    expression: str
    score: float
    correct: bool

    def as_public_dict(self) -> Mapping[str, object]:
        """Return trainer-safe hypothesis data."""

        return freeze_mapping(
            {
                "hypothesis_id": self.hypothesis_id,
                "expression": self.expression,
                "score": self.score,
                "correct": self.correct,
            }
        )


@dataclass(frozen=True)
class HypothesisEvaluation:
    """Numeric hidden-point evaluation result."""

    accepted: bool
    score: float
    correct: bool
    reason: str
    valid_points: int
    max_relative_error: float
    mean_relative_error: float


@dataclass(frozen=True)
class ParsedDiscoveryAction:
    """Interpreted discovery action."""

    action_type: str
    inputs: Mapping[str, object] = field(default_factory=dict)
    equation: str = ""

    def __post_init__(self) -> None:
        """Freeze parsed action inputs after construction."""

        object.__setattr__(self, "inputs", freeze_mapping(self.inputs))
