"""Cart inference budget vocabulary and validation helpers."""

from typing import Mapping

TURN_BUDGET = "turns"
ACTION_BUDGET = "actions"
FINAL_ANSWER_BUDGET = "final_answers"
CART_BUDGET_NAMES = frozenset((TURN_BUDGET, ACTION_BUDGET, FINAL_ANSWER_BUDGET))


def cart_budget_limits(
    turn_budget: int,
    action_budget: int,
    final_answer_budget: int,
) -> dict[str, int]:
    """Return the canonical cart budget-limit mapping.

    Parameters
    ----------
    turn_budget:
        Maximum number of model submissions accepted before truncation.
    action_budget:
        Maximum number of measurement actions allowed.
    final_answer_budget:
        Maximum number of final-answer attempts allowed.

    Returns
    -------
    dict[str, int]
        Budget limits keyed by the cart task's public budget names.
    """

    return {
        TURN_BUDGET: turn_budget,
        ACTION_BUDGET: action_budget,
        FINAL_ANSWER_BUDGET: final_answer_budget,
    }


def validate_cart_budget_limits(budget_limits: Mapping[str, int]) -> None:
    """Validate the task-specific cart budget vocabulary and values.

    Parameters
    ----------
    budget_limits:
        Public budget limits from a cart task spec or instance.

    Raises
    ------
    ValueError
        Raised when a required cart budget is missing, an unknown budget is
        present, or the task-specific budget relationships are invalid.
    """

    names = set(budget_limits)
    missing_names = sorted(CART_BUDGET_NAMES - names)
    if len(missing_names) > 0:
        joined_names = ", ".join(missing_names)
        raise ValueError(f"cart inference budget_limits missing: {joined_names}")
    unknown_names = sorted(names - CART_BUDGET_NAMES)
    if len(unknown_names) > 0:
        joined_names = ", ".join(unknown_names)
        raise ValueError(f"cart inference budget_limits unknown: {joined_names}")

    turn_budget = required_cart_budget(budget_limits, TURN_BUDGET)
    action_budget = required_cart_budget(budget_limits, ACTION_BUDGET)
    final_answer_budget = required_cart_budget(budget_limits, FINAL_ANSWER_BUDGET)
    if turn_budget <= 0:
        raise ValueError("cart inference turns budget must be positive")
    if action_budget <= 0:
        raise ValueError("cart inference actions budget must be positive")
    if final_answer_budget <= 0:
        raise ValueError("cart inference final_answers budget must be positive")
    if final_answer_budget != 1:
        raise ValueError("cart inference final_answers budget must be 1")
    if action_budget + final_answer_budget > turn_budget:
        raise ValueError(
            "cart inference actions and final_answers budgets must fit within turns"
        )


def required_cart_budget(budget_limits: Mapping[str, int], name: str) -> int:
    """Return one required cart budget value.

    Parameters
    ----------
    budget_limits:
        Public budget limits from a cart task spec or instance.
    name:
        Cart budget name to read.

    Returns
    -------
    int
        The requested cart budget value.

    Raises
    ------
    RuntimeError
        Raised when a cart task instance is missing a required budget value.
    """

    value = budget_limits.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"cart inference instances require {name} budget")
    return value
