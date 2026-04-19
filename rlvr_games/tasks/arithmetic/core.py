"""Procedural arithmetic verifier task implementation."""

from dataclasses import dataclass
from enum import StrEnum
from random import Random
import re

from rlvr_games.core import (
    Observation,
    SingleStepTask,
    SingleStepVerifierSession,
    TaskInstance,
    VerificationResult,
)


class ArithmeticOperation(StrEnum):
    """Supported arithmetic operations."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"


@dataclass(slots=True, frozen=True)
class ArithmeticTaskPayload:
    """Canonical payload for one arithmetic task instance.

    Attributes
    ----------
    left : int
        Left operand.
    right : int
        Right operand.
    operation : ArithmeticOperation
        Operation to apply.
    """

    left: int
    right: int
    operation: ArithmeticOperation

    @property
    def answer(self) -> int:
        """Return the exact integer answer."""
        if self.operation == ArithmeticOperation.ADD:
            return self.left + self.right
        if self.operation == ArithmeticOperation.SUBTRACT:
            return self.left - self.right
        if self.operation == ArithmeticOperation.MULTIPLY:
            return self.left * self.right
        raise ValueError(f"Unsupported arithmetic operation: {self.operation!r}")

    @property
    def expression(self) -> str:
        """Return the human-readable arithmetic expression."""
        symbol_by_operation = {
            ArithmeticOperation.ADD: "+",
            ArithmeticOperation.SUBTRACT: "-",
            ArithmeticOperation.MULTIPLY: "*",
        }
        return f"{self.left} {symbol_by_operation[self.operation]} {self.right}"


@dataclass(slots=True, frozen=True)
class ArithmeticTaskSource:
    """Deterministic procedural arithmetic task source."""

    min_value: int = -20
    max_value: int = 20
    operations: tuple[ArithmeticOperation, ...] = (
        ArithmeticOperation.ADD,
        ArithmeticOperation.SUBTRACT,
        ArithmeticOperation.MULTIPLY,
    )

    def __post_init__(self) -> None:
        """Validate source configuration."""
        if self.min_value > self.max_value:
            raise ValueError("ArithmeticTaskSource min_value cannot exceed max_value.")
        if not self.operations:
            raise ValueError("ArithmeticTaskSource requires at least one operation.")

    def sample(self, *, seed: int) -> SingleStepTask[ArithmeticTaskPayload]:
        """Sample one deterministic arithmetic task for ``seed``."""
        random = Random(seed)
        operation = self.operations[random.randrange(len(self.operations))]
        payload = ArithmeticTaskPayload(
            left=random.randint(self.min_value, self.max_value),
            right=random.randint(self.min_value, self.max_value),
            operation=operation,
        )
        task_instance_id = (
            "arithmetic:"
            f"seed={seed}:"
            f"{payload.operation.value}:"
            f"{payload.left}:{payload.right}"
        )
        return SingleStepTask(
            instance=TaskInstance(
                task_instance_id=task_instance_id,
                task_kind="arithmetic",
                seed=seed,
                prompt_key=f"{payload.operation.value}:{payload.left}:{payload.right}",
                metadata={
                    "left": payload.left,
                    "right": payload.right,
                    "operation": payload.operation.value,
                },
            ),
            payload=payload,
        )


class ArithmeticPromptRenderer:
    """Render arithmetic tasks as prompt observations."""

    def render(self, task: SingleStepTask[ArithmeticTaskPayload]) -> Observation:
        """Render one arithmetic prompt."""
        return Observation(
            text=(
                "Compute the exact integer result of this expression. "
                "Reply with the integer answer only.\n\n"
                f"{task.payload.expression}"
            ),
            metadata={
                "task_instance_id": task.instance.task_instance_id,
                "task_kind": task.instance.task_kind,
                "operation": task.payload.operation.value,
            },
        )


class ArithmeticVerifier:
    """Executable verifier for arithmetic completions."""

    def verify(
        self,
        *,
        task: SingleStepTask[ArithmeticTaskPayload],
        completion: str,
    ) -> VerificationResult:
        """Verify one arithmetic completion."""
        parsed_output = _parse_integer_completion(completion)
        if parsed_output is None:
            return VerificationResult(
                parsed_output=None,
                valid_submission=False,
                reward=0.0,
                info={"valid_submission": False, "reason": "no_integer_found"},
                debug_info={"expected": task.payload.answer},
            )
        correct = parsed_output == task.payload.answer
        return VerificationResult(
            parsed_output=parsed_output,
            valid_submission=True,
            reward=1.0 if correct else 0.0,
            info={
                "valid_submission": True,
                "correct": correct,
            },
            debug_info={"expected": task.payload.answer},
        )


def make_arithmetic_session(
    *,
    task_source: ArithmeticTaskSource | None = None,
) -> SingleStepVerifierSession[ArithmeticTaskPayload]:
    """Create one arithmetic single-step verifier session.

    Parameters
    ----------
    task_source : ArithmeticTaskSource | None
        Optional deterministic task source. Defaults to the standard arithmetic
        source.

    Returns
    -------
    SingleStepVerifierSession[ArithmeticTaskPayload]
        Scalar verifier session for arithmetic tasks.
    """
    return SingleStepVerifierSession(
        task_source=task_source if task_source is not None else ArithmeticTaskSource(),
        prompt_renderer=ArithmeticPromptRenderer(),
        verifier=ArithmeticVerifier(),
    )


def _parse_integer_completion(completion: str) -> int | None:
    """Parse the final integer mentioned in a completion."""
    matches = re.findall(r"[-+]?\d+", completion)
    if not matches:
        return None
    return int(matches[-1])


__all__ = [
    "ArithmeticOperation",
    "ArithmeticPromptRenderer",
    "ArithmeticTaskPayload",
    "ArithmeticTaskSource",
    "ArithmeticVerifier",
    "make_arithmetic_session",
]
