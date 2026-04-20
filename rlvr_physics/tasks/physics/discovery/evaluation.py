"""Hypothesis evaluation for physics discovery."""

from typing import Mapping

from rlvr_physics.core.instances import TaskInstance, require_mapping, require_str
from rlvr_physics.tasks.physics.discovery.expressions import (
    evaluate_expression,
    extract_hypothesis_expression,
)
from rlvr_physics.tasks.physics.discovery.types import HypothesisEvaluation
from rlvr_physics.tasks.physics.discovery.utils import float_mapping


def evaluate_physics_hypothesis(
    instance: TaskInstance, expression: str
) -> HypothesisEvaluation:
    """Evaluate a candidate equation on hidden deterministic points.

    Parameters
    ----------
    instance:
        Immutable discovery instance.
    expression:
        Candidate expression using the instance's public variable names.
    """

    candidate = extract_hypothesis_expression(expression)
    if not candidate:
        return HypothesisEvaluation(
            accepted=False,
            score=0.0,
            correct=False,
            reason="empty_hypothesis",
            valid_points=0,
            max_relative_error=float("inf"),
            mean_relative_error=float("inf"),
        )
    true_equation = require_str(instance.privileged_payload["equation"], "equation")
    hidden_points = _hidden_points(instance)
    relative_errors: list[float] = []
    for point in hidden_points:
        variables = float_mapping(point)
        try:
            true_value = evaluate_expression(true_equation, variables)
            candidate_value = evaluate_expression(candidate, variables)
        except (ArithmeticError, ValueError, TypeError, OverflowError):
            return HypothesisEvaluation(
                accepted=False,
                score=0.0,
                correct=False,
                reason="invalid_hypothesis",
                valid_points=len(relative_errors),
                max_relative_error=float("inf"),
                mean_relative_error=float("inf"),
            )
        scale = max(1.0, abs(true_value))
        relative_errors.append(abs(candidate_value - true_value) / scale)
    if not relative_errors:
        return HypothesisEvaluation(
            accepted=False,
            score=0.0,
            correct=False,
            reason="no_hidden_points",
            valid_points=0,
            max_relative_error=float("inf"),
            mean_relative_error=float("inf"),
        )
    mean_error = sum(relative_errors) / len(relative_errors)
    max_error = max(relative_errors)
    score = max(0.0, 1.0 - min(1.0, mean_error))
    correct = max_error <= 1e-5
    if correct:
        score = 1.0
    return HypothesisEvaluation(
        accepted=True,
        score=score,
        correct=correct,
        reason="correct_hypothesis" if correct else "hypothesis_tested",
        valid_points=len(relative_errors),
        max_relative_error=max_error,
        mean_relative_error=mean_error,
    )


def _hidden_points(instance: TaskInstance) -> tuple[Mapping[str, object], ...]:
    value = instance.privileged_payload["hidden_points"]
    if not isinstance(value, tuple):
        raise TypeError("hidden_points must be a tuple")
    points: list[Mapping[str, object]] = []
    for point in value:
        points.append(require_mapping(point, "hidden point"))
    return tuple(points)
