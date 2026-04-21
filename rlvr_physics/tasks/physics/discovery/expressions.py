"""Safe expression extraction and evaluation for physics discovery."""

from collections.abc import Callable
import ast
import json
import math
import re
from typing import Mapping

from rlvr_physics.tasks._shared.submissions import strip_code_fence_lines
from rlvr_physics.tasks.physics.discovery.utils import coerce_float


def extract_hypothesis_expression(text: str) -> str:
    """Extract a scalar expression from hypothesis text.

    Parameters
    ----------
    text:
        Raw final text or JSON-encoded hypothesis payload.
    """

    stripped = text.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        equation = parsed.get("equation", parsed.get("hypothesis"))
        if isinstance(equation, str):
            return extract_hypothesis_expression(equation)

    lines = strip_code_fence_lines(stripped).splitlines()
    if not lines:
        return ""
    joined = "\n".join(lines)
    return_match = re.search(r"\breturn\s+(.+)", joined)
    if return_match:
        return return_match.group(1).strip()
    candidate = lines[-1]
    assignment_match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)$", candidate)
    if assignment_match:
        candidate = assignment_match.group(1)
    return candidate.strip().rstrip(".")


def evaluate_expression(expression: str, variables: Mapping[str, float]) -> float:
    """Evaluate a restricted numeric expression."""

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid expression syntax") from error
    value = _evaluate_ast_node(parsed.body, variables)
    result = coerce_float(value, "expression result")
    _ensure_reasonable_number(result)
    return result


def _evaluate_ast_node(node: ast.AST, variables: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("only numeric constants are allowed")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        constant = _named_constant(node.id)
        if constant is not None:
            return constant
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_ast_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast_node(node.left, variables)
        right = _evaluate_ast_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return _checked_number(left + right)
        if isinstance(node.op, ast.Sub):
            return _checked_number(left - right)
        if isinstance(node.op, ast.Mult):
            return _checked_number(left * right)
        if isinstance(node.op, ast.Div):
            return _checked_number(left / right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 12:
                raise ValueError("exponent too large")
            return _checked_number(left**right)
        raise ValueError("unsupported binary operator")
    if isinstance(node, ast.Call):
        function = _allowed_function(node.func)
        args = [_evaluate_ast_node(argument, variables) for argument in node.args]
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        return _checked_number(function(*args))
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in ("np", "math"):
            if node.attr == "pi":
                return math.pi
            if node.attr == "e":
                return math.e
        raise ValueError("unsupported attribute")
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _allowed_function(node: ast.AST) -> Callable[..., float]:
    name = ""
    if isinstance(node, ast.Name):
        name = node.id
    elif (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("np", "math")
    ):
        name = node.attr
    functions: dict[str, Callable[..., float]] = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "abs": _absolute_value,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "arcsin": math.asin,
        "arccos": math.acos,
        "arctan": math.atan,
        "arccosh": math.acosh,
        "acosh": math.acosh,
    }
    if name not in functions:
        raise ValueError(f"unsupported function: {name}")
    return functions[name]


def _named_constant(name: str) -> float | None:
    constants = {
        "pi": math.pi,
        "e": math.e,
        "c": 299792458.0,
        "mu_0": 4.0 * math.pi * 1e-7,
    }
    return constants.get(name)


def _absolute_value(value: float) -> float:
    """Return the absolute value as a float."""

    return abs(value)


def _checked_number(value: float) -> float:
    result = coerce_float(value, "expression value")
    _ensure_reasonable_number(result)
    return result


def _ensure_reasonable_number(value: float) -> None:
    if not math.isfinite(value):
        raise ValueError("expression produced a non-finite value")
    if abs(value) > 1e100:
        raise ValueError("expression produced an unreasonably large value")
