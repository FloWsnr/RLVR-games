"""Cart inference budget vocabulary and validation helpers."""

from typing import Mapping

from rlvr_physics.core.submissions import InvalidSubmissionPolicy

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


class CartRolloutBudgetState:
    """Mutable rollout-local budget counters for cart inference.

    Parameters
    ----------
    budget_limits:
        Public cart budget limits from the immutable task instance.

    Attributes
    ----------
    submissions_used:
        Number of consumed turn submissions.
    invalid_submissions:
        Number of rejected submissions observed by the task core.
    final_answers_used:
        Number of consumed final-answer attempts.
    """

    def __init__(self, budget_limits: Mapping[str, int]) -> None:
        """Initialize validated cart budget counters.

        Parameters
        ----------
        budget_limits:
            Public cart budget limits from the immutable task instance.
        """

        validate_cart_budget_limits(budget_limits)
        self._turn_budget = required_cart_budget(budget_limits, TURN_BUDGET)
        self._final_answer_budget = required_cart_budget(
            budget_limits, FINAL_ANSWER_BUDGET
        )
        self.submissions_used = 0
        self.invalid_submissions = 0
        self.final_answers_used = 0

    @property
    def final_answer_budget(self) -> int:
        """Return the final-answer attempt budget.

        Returns
        -------
        int
            Maximum number of final-answer attempts.
        """

        return self._final_answer_budget

    @property
    def turns_remaining(self) -> int:
        """Return the remaining turn budget.

        Returns
        -------
        int
            Non-negative remaining turn submissions.
        """

        return max(0, self._turn_budget - self.submissions_used)

    @property
    def final_answers_remaining(self) -> int:
        """Return the remaining final-answer budget.

        Returns
        -------
        int
            Non-negative remaining final-answer attempts.
        """

        return max(0, self._final_answer_budget - self.final_answers_used)

    def reset(self) -> None:
        """Reset all rollout-local counters."""

        self.submissions_used = 0
        self.invalid_submissions = 0
        self.final_answers_used = 0

    def record_action_submission(self) -> None:
        """Consume one turn for an action-mode submission."""

        self.submissions_used += 1

    def record_final_answer_submission(self) -> None:
        """Consume one turn and one final-answer attempt."""

        self.submissions_used += 1
        self.final_answers_used += 1

    def record_invalid_submission(self, policy: InvalidSubmissionPolicy) -> None:
        """Apply an invalid-submission policy to public counters.

        Parameters
        ----------
        policy:
            Public invalid-submission policy controlling the rejection.
        """

        _reject_action_budget_consumption(policy)
        self.invalid_submissions += 1
        self.submissions_used += policy.consumes_budget.get(TURN_BUDGET, 0)
        self.final_answers_used += policy.consumes_budget.get(FINAL_ANSWER_BUDGET, 0)

    def record_invalid_after_counted_submission(self) -> None:
        """Record a rejection after submission budgets have already been used."""

        self.invalid_submissions += 1

    def turn_budget_exhausted(self) -> bool:
        """Return whether the consumed turn budget has reached its limit.

        Returns
        -------
        bool
            ``True`` when no further turn should be emitted.
        """

        return self.submissions_used >= self._turn_budget

    def budget_usage(self, actions_used: int) -> dict[str, int]:
        """Return consumed public budgets.

        Parameters
        ----------
        actions_used:
            Accepted measurement action count owned by the backbone.

        Returns
        -------
        dict[str, int]
            Consumed budgets keyed by the cart budget namespace.
        """

        return {
            TURN_BUDGET: self.submissions_used,
            ACTION_BUDGET: actions_used,
            FINAL_ANSWER_BUDGET: self.final_answers_used,
        }

    def budget_remaining(self, actions_remaining: int) -> dict[str, int]:
        """Return remaining public budgets.

        Parameters
        ----------
        actions_remaining:
            Remaining accepted measurement actions owned by the backbone.

        Returns
        -------
        dict[str, int]
            Remaining budgets keyed by the cart budget namespace.
        """

        return {
            TURN_BUDGET: self.turns_remaining,
            ACTION_BUDGET: actions_remaining,
            FINAL_ANSWER_BUDGET: self.final_answers_remaining,
        }

    def public_status(
        self,
        actions_used: int,
        actions_remaining: int,
        extra_info: Mapping[str, object],
    ) -> dict[str, object]:
        """Return trainer-safe rollout counters and event metadata.

        Parameters
        ----------
        actions_used:
            Accepted measurement action count owned by the backbone.
        actions_remaining:
            Remaining accepted measurement actions owned by the backbone.
        extra_info:
            Event-specific public metadata merged into the returned status.

        Returns
        -------
        dict[str, object]
            Public budget usage, remaining budgets, counters, and event
            metadata.
        """

        status: dict[str, object] = {
            "budget_usage": self.budget_usage(actions_used),
            "budget_remaining": self.budget_remaining(actions_remaining),
            "actions_used": actions_used,
            "actions_remaining": actions_remaining,
            "final_answers_used": self.final_answers_used,
            "final_answers_remaining": self.final_answers_remaining,
            "submissions_used": self.submissions_used,
            "invalid_submissions": self.invalid_submissions,
        }
        status.update(extra_info)
        return status


def _reject_action_budget_consumption(policy: InvalidSubmissionPolicy) -> None:
    """Reject action budget effects that are owned by the backbone."""

    if policy.consumes_budget.get(ACTION_BUDGET, 0) != 0:
        raise ValueError(
            "cart invalid-submission policies cannot consume accepted action budget"
        )
