"""Countdown expression verification."""

import ast
from fractions import Fraction
import re

from rlvr_physics.core.instances import TaskInstance, require_int, require_tuple_of_ints
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks._shared.submissions import nonempty_stripped_lines
from rlvr_physics.tasks.games.countdown.types import CountdownVerification


def verify_countdown_submission(
    instance: TaskInstance, submission: TaskSubmission
) -> CountdownVerification:
    """Verify one Countdown model submission."""

    numbers = require_tuple_of_ints(instance.public_payload["numbers"], "numbers")
    target = require_int(instance.public_payload["target"], "target")
    expression = _extract_expression(submission.raw)
    if not expression:
        return CountdownVerification(
            accepted=False,
            correct=False,
            reward=0.01,
            reason="empty_submission",
            expression="",
            value=None,
            used_numbers=(),
        )
    try:
        parsed = ast.parse(expression, mode="eval")
        used_numbers: list[int] = []
        value = _evaluate_countdown_ast(parsed.body, used_numbers)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError):
        return CountdownVerification(
            accepted=False,
            correct=False,
            reward=0.01,
            reason="invalid_expression",
            expression=expression,
            value=None,
            used_numbers=(),
        )

    used_tuple = tuple(used_numbers)
    if sorted(used_tuple) != sorted(numbers):
        return CountdownVerification(
            accepted=True,
            correct=False,
            reward=0.05,
            reason="wrong_numbers",
            expression=expression,
            value=value,
            used_numbers=used_tuple,
        )
    if value != Fraction(target, 1):
        return CountdownVerification(
            accepted=True,
            correct=False,
            reward=0.05,
            reason="wrong_value",
            expression=expression,
            value=value,
            used_numbers=used_tuple,
        )
    return CountdownVerification(
        accepted=True,
        correct=True,
        reward=1.0,
        reason="correct",
        expression=expression,
        value=value,
        used_numbers=used_tuple,
    )


def _extract_expression(raw_submission: str) -> str:
    stripped = raw_submission.strip()
    if not stripped:
        return ""
    fenced = re.search(
        r"```(?:text|python)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced is not None:
        stripped = fenced.group(1).strip()
    answer_match = re.search(
        r"(?:answer|expression)\s*[:=]\s*(.+)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match is not None:
        stripped = answer_match.group(1).strip()
    lines = nonempty_stripped_lines(stripped)
    if not lines:
        return ""
    candidate = lines[-1]
    if "=" in candidate:
        candidate = candidate.split("=")[0].strip()
    return candidate


def _evaluate_countdown_ast(node: ast.AST, used_numbers: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_countdown_ast(node.body, used_numbers)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise ValueError("Countdown expressions may only contain integer constants")
        used_numbers.append(node.value)
        return Fraction(node.value, 1)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate_countdown_ast(node.operand, used_numbers)
        if isinstance(node.op, ast.USub):
            return -value
        return value
    if isinstance(node, ast.BinOp):
        left = _evaluate_countdown_ast(node.left, used_numbers)
        right = _evaluate_countdown_ast(node.right, used_numbers)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
    raise ValueError("unsupported expression syntax")
