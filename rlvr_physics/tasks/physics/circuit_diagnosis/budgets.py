"""Budget vocabulary and rollout counters for circuit diagnosis."""

from collections.abc import Mapping

from rlvr_physics.core.submissions import InvalidSubmissionPolicy

TURN_BUDGET = "turns"
PROBE_BUDGET = "probe_actions"
REPAIR_BUDGET = "repair_actions"
FINAL_ANSWER_BUDGET = "final_answers"
CIRCUIT_BUDGET_NAMES = frozenset(
    (TURN_BUDGET, PROBE_BUDGET, REPAIR_BUDGET, FINAL_ANSWER_BUDGET)
)


def circuit_budget_limits(
    turn_budget: int,
    probe_budget: int,
    repair_budget: int,
    final_answer_budget: int,
) -> dict[str, int]:
    """Return the canonical circuit budget-limit mapping.

    Parameters
    ----------
    turn_budget:
        Maximum number of model submissions before truncation.
    probe_budget:
        Maximum number of accepted probing actions.
    repair_budget:
        Maximum number of accepted repair actions.
    final_answer_budget:
        Maximum number of final-answer attempts.

    Returns
    -------
    dict[str, int]
        Budget limits keyed by the circuit task's public budget names.
    """

    return {
        TURN_BUDGET: turn_budget,
        PROBE_BUDGET: probe_budget,
        REPAIR_BUDGET: repair_budget,
        FINAL_ANSWER_BUDGET: final_answer_budget,
    }


def validate_circuit_budget_limits(budget_limits: Mapping[str, int]) -> None:
    """Validate circuit diagnosis budget names and values.

    Parameters
    ----------
    budget_limits:
        Public budget limits from a circuit task spec or instance.

    Raises
    ------
    ValueError
        Raised when required budgets are missing, unknown budgets are present,
        or budget relationships are invalid.
    """

    names = set(budget_limits)
    missing_names = sorted(CIRCUIT_BUDGET_NAMES - names)
    if len(missing_names) > 0:
        joined_names = ", ".join(missing_names)
        raise ValueError(f"circuit diagnosis budget_limits missing: {joined_names}")
    unknown_names = sorted(names - CIRCUIT_BUDGET_NAMES)
    if len(unknown_names) > 0:
        joined_names = ", ".join(unknown_names)
        raise ValueError(f"circuit diagnosis budget_limits unknown: {joined_names}")

    turn_budget = required_circuit_budget(budget_limits, TURN_BUDGET)
    probe_budget = required_circuit_budget(budget_limits, PROBE_BUDGET)
    repair_budget = required_circuit_budget(budget_limits, REPAIR_BUDGET)
    final_answer_budget = required_circuit_budget(budget_limits, FINAL_ANSWER_BUDGET)
    if turn_budget <= 0:
        raise ValueError("circuit diagnosis turns budget must be positive")
    if probe_budget <= 0:
        raise ValueError("circuit diagnosis probe_actions budget must be positive")
    if repair_budget <= 0:
        raise ValueError("circuit diagnosis repair_actions budget must be positive")
    if final_answer_budget <= 0:
        raise ValueError("circuit diagnosis final_answers budget must be positive")
    if final_answer_budget != 1:
        raise ValueError("circuit diagnosis final_answers budget must be 1")
    if probe_budget + repair_budget + final_answer_budget > turn_budget:
        raise ValueError(
            "probe_actions, repair_actions, and final_answers budgets must fit "
            "within turns"
        )


def required_circuit_budget(budget_limits: Mapping[str, int], name: str) -> int:
    """Return one required circuit budget value.

    Parameters
    ----------
    budget_limits:
        Public budget limits from a circuit task spec or instance.
    name:
        Circuit budget name to read.

    Returns
    -------
    int
        The requested budget value.

    Raises
    ------
    RuntimeError
        Raised when a circuit task instance is missing a required budget value.
    """

    value = budget_limits.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"circuit diagnosis instances require {name} budget")
    return value


class CircuitRolloutBudgetState:
    """Mutable rollout-local budget counters for circuit diagnosis.

    Parameters
    ----------
    budget_limits:
        Public circuit budget limits from the immutable task instance.

    Attributes
    ----------
    submissions_used:
        Number of consumed turn submissions.
    invalid_submissions:
        Number of rejected submissions observed by the task core.
    probe_actions_used:
        Number of accepted probing actions.
    repair_actions_used:
        Number of accepted repair actions.
    final_answers_used:
        Number of consumed final-answer attempts.
    """

    def __init__(self, budget_limits: Mapping[str, int]) -> None:
        """Initialize validated circuit budget counters."""

        validate_circuit_budget_limits(budget_limits)
        self._turn_budget = required_circuit_budget(budget_limits, TURN_BUDGET)
        self._probe_budget = required_circuit_budget(budget_limits, PROBE_BUDGET)
        self._repair_budget = required_circuit_budget(budget_limits, REPAIR_BUDGET)
        self._final_answer_budget = required_circuit_budget(
            budget_limits, FINAL_ANSWER_BUDGET
        )
        self.submissions_used = 0
        self.invalid_submissions = 0
        self.probe_actions_used = 0
        self.repair_actions_used = 0
        self.final_answers_used = 0

    @property
    def turns_remaining(self) -> int:
        """Return the remaining turn budget."""

        return max(0, self._turn_budget - self.submissions_used)

    @property
    def probe_actions_remaining(self) -> int:
        """Return the remaining probing action budget."""

        return max(0, self._probe_budget - self.probe_actions_used)

    @property
    def repair_actions_remaining(self) -> int:
        """Return the remaining repair action budget."""

        return max(0, self._repair_budget - self.repair_actions_used)

    @property
    def final_answers_remaining(self) -> int:
        """Return the remaining final-answer budget."""

        return max(0, self._final_answer_budget - self.final_answers_used)

    def reset(self) -> None:
        """Reset all rollout-local counters."""

        self.submissions_used = 0
        self.invalid_submissions = 0
        self.probe_actions_used = 0
        self.repair_actions_used = 0
        self.final_answers_used = 0

    def record_turn_submission(self) -> None:
        """Consume one turn for a submission attempt."""

        self.submissions_used += 1

    def probe_budget_available(self) -> bool:
        """Return whether another probing action can be accepted."""

        return self.probe_actions_remaining > 0

    def repair_budget_available(self) -> bool:
        """Return whether another repair action can be accepted."""

        return self.repair_actions_remaining > 0

    def record_accepted_probe(self) -> None:
        """Consume one accepted probing action."""

        if self.probe_actions_remaining <= 0:
            raise RuntimeError("probe action budget exhausted")
        self.probe_actions_used += 1

    def record_accepted_repair(self) -> None:
        """Consume one accepted repair action."""

        if self.repair_actions_remaining <= 0:
            raise RuntimeError("repair action budget exhausted")
        self.repair_actions_used += 1

    def record_final_answer_submission(self) -> bool:
        """Consume one turn and final-answer attempt when possible.

        Returns
        -------
        bool
            ``True`` when the final-answer budget was available.
        """

        self.submissions_used += 1
        if self.final_answers_remaining <= 0:
            return False
        self.final_answers_used += 1
        return True

    def record_invalid_submission(self, policy: InvalidSubmissionPolicy) -> None:
        """Apply an invalid-submission policy to public counters."""

        _reject_action_budget_consumption(policy)
        self.invalid_submissions += 1
        self.submissions_used += policy.consumes_budget.get(TURN_BUDGET, 0)
        self.final_answers_used += policy.consumes_budget.get(FINAL_ANSWER_BUDGET, 0)

    def record_invalid_after_counted_submission(self) -> None:
        """Record a rejection after submission budget was already consumed."""

        self.invalid_submissions += 1

    def turn_budget_exhausted(self) -> bool:
        """Return whether the consumed turn budget has reached its limit."""

        return self.submissions_used >= self._turn_budget

    def budget_usage(self) -> dict[str, int]:
        """Return consumed public budgets."""

        return {
            TURN_BUDGET: self.submissions_used,
            PROBE_BUDGET: self.probe_actions_used,
            REPAIR_BUDGET: self.repair_actions_used,
            FINAL_ANSWER_BUDGET: self.final_answers_used,
        }

    def budget_remaining(self) -> dict[str, int]:
        """Return remaining public budgets."""

        return {
            TURN_BUDGET: self.turns_remaining,
            PROBE_BUDGET: self.probe_actions_remaining,
            REPAIR_BUDGET: self.repair_actions_remaining,
            FINAL_ANSWER_BUDGET: self.final_answers_remaining,
        }

    def public_status(self, extra_info: Mapping[str, object]) -> dict[str, object]:
        """Return trainer-safe rollout counters and event metadata."""

        status: dict[str, object] = {
            "budget_usage": self.budget_usage(),
            "budget_remaining": self.budget_remaining(),
            "probe_actions_used": self.probe_actions_used,
            "probe_actions_remaining": self.probe_actions_remaining,
            "repair_actions_used": self.repair_actions_used,
            "repair_actions_remaining": self.repair_actions_remaining,
            "final_answers_used": self.final_answers_used,
            "final_answers_remaining": self.final_answers_remaining,
            "submissions_used": self.submissions_used,
            "invalid_submissions": self.invalid_submissions,
        }
        status.update(extra_info)
        return status


def _reject_action_budget_consumption(policy: InvalidSubmissionPolicy) -> None:
    """Reject invalid policies that claim accepted action budget."""

    if policy.consumes_budget.get(PROBE_BUDGET, 0) != 0:
        raise ValueError(
            "circuit invalid-submission policies cannot consume probe_actions"
        )
    if policy.consumes_budget.get(REPAIR_BUDGET, 0) != 0:
        raise ValueError(
            "circuit invalid-submission policies cannot consume repair_actions"
        )
