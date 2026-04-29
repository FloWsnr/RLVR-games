"""Shared submission protocols and parsing helpers."""

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, cast

from rlvr_physics.core.payloads import freeze_mapping

ACTION_NAME_FIELD = "action"
ACTION_ARGUMENTS_FIELD = "arguments"
JSON_LINE_FORMAT = "json_line"


@dataclass(frozen=True)
class TaskSubmission:
    """A raw model submission plus optional interpreted payload.

    Attributes
    ----------
    kind:
        Submission category, such as ``final_text`` or ``action``.
    raw:
        Raw model text or action string.
    parsed:
        Task- or integration-interpreted payload.
    """

    kind: str
    raw: str
    parsed: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parsed payload after construction."""

        object.__setattr__(self, "parsed", freeze_mapping(self.parsed))

    @classmethod
    def final_text(cls, text: str) -> "TaskSubmission":
        """Create a raw final-text submission.

        Parameters
        ----------
        text:
            Final answer text emitted by the model.

        Returns
        -------
        TaskSubmission
            Submission with kind ``final_text`` and an empty parsed payload.
        """

        return cls(kind="final_text", raw=text, parsed={})

    @classmethod
    def action(cls, action: str) -> "TaskSubmission":
        """Create an action submission.

        Parameters
        ----------
        action:
            Action text emitted by the model.

        Returns
        -------
        TaskSubmission
            Submission with kind ``action`` and the action mirrored into the
            parsed payload.
        """

        return cls(kind="action", raw=action, parsed={ACTION_NAME_FIELD: action})


@dataclass(frozen=True)
class ParsedAction:
    """Canonical structured action parsed from a model submission.

    Parameters
    ----------
    name:
        Action name requested by the model.
    arguments:
        Structured action arguments.

    Attributes
    ----------
    name:
        Action name requested by the model.
    arguments:
        Frozen structured action arguments.
    """

    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze argument payloads after construction."""

        object.__setattr__(self, "arguments", freeze_mapping(self.arguments))


@dataclass(frozen=True)
class InvalidSubmissionPolicy:
    """Policy for one category of rejected model submission.

    Parameters
    ----------
    category:
        Stable public category for the rejected submission.
    consumes_budget:
        Named public budgets consumed by this rejected submission.
    reward:
        Scalar reward emitted for the rejected submission.
    terminal:
        Whether the rejection ends the task as a terminal result.
    truncated:
        Whether the rejection ends the task by truncation.

    Attributes
    ----------
    category:
        Stable public category for the rejected submission.
    consumes_budget:
        Frozen mapping of public budget names to consumed amounts.
    reward:
        Scalar reward emitted for the rejected submission.
    terminal:
        Whether the rejection ends the task as a terminal result.
    truncated:
        Whether the rejection ends the task by truncation.
    """

    category: str
    consumes_budget: Mapping[str, int]
    reward: float
    terminal: bool
    truncated: bool

    def __post_init__(self) -> None:
        """Validate and freeze budget effects."""

        if self.category == "":
            raise ValueError("invalid submission category must be non-empty")
        for name, amount in self.consumes_budget.items():
            if not isinstance(name, str) or name == "":
                raise ValueError("budget name must be a non-empty string")
            if isinstance(amount, bool) or not isinstance(amount, int):
                raise ValueError("budget consumption must be an integer")
            if amount < 0:
                raise ValueError("budget consumption must be non-negative")
        object.__setattr__(
            self, "consumes_budget", freeze_mapping(self.consumes_budget)
        )


def invalid_submission_policy_payload(
    policy: InvalidSubmissionPolicy,
) -> dict[str, object]:
    """Return public metadata for one invalid-submission policy.

    Parameters
    ----------
    policy:
        Invalid-submission policy to serialize.

    Returns
    -------
    dict[str, object]
        Plain trainer-safe policy payload.
    """

    return {
        "category": policy.category,
        "consumes_budget": dict(policy.consumes_budget),
        "reward": policy.reward,
        "terminal": policy.terminal,
        "truncated": policy.truncated,
    }


def invalid_submission_policies_payload(
    policies: Mapping[str, InvalidSubmissionPolicy],
) -> dict[str, object]:
    """Return a public mapping of invalid-submission policies.

    Parameters
    ----------
    policies:
        Policies keyed by public category.

    Returns
    -------
    dict[str, object]
        Plain trainer-safe policy payload mapping.
    """

    payload: dict[str, object] = {}
    for category, policy in policies.items():
        if category != policy.category:
            raise ValueError("invalid submission policy key must match category")
        payload[category] = invalid_submission_policy_payload(policy)
    return payload


def validate_invalid_submission_policies(
    policies: Mapping[str, InvalidSubmissionPolicy],
    public_limits: Mapping[str, object],
) -> None:
    """Validate that policy budget effects reference named public budgets.

    Parameters
    ----------
    policies:
        Policies keyed by public category.
    public_limits:
        Public rollout limits advertised for the current turn.

    Raises
    ------
    ValueError
        Raised when a policy references an unknown budget key or category key.
    """

    budget_names = _budget_names_from_public_limits(public_limits)

    for category, policy in policies.items():
        if category != policy.category:
            raise ValueError("invalid submission policy key must match category")
        _validate_consumes_budget_mapping(
            policy.consumes_budget,
            budget_names,
            "invalid submission policy",
        )


def validate_action_schema_budget_references(
    action_schema: Mapping[str, object],
    public_limits: Mapping[str, object],
) -> None:
    """Validate action-schema budget effects against public budget limits.

    Parameters
    ----------
    action_schema:
        Public action schema that may contain nested ``consumes_budget`` maps.
    public_limits:
        Public rollout limits advertised for the current turn.

    Raises
    ------
    ValueError
        Raised when a ``consumes_budget`` entry references an unknown budget or
        has a malformed budget effect.
    """

    budget_names = _budget_names_from_public_limits(public_limits)
    _validate_nested_consumes_budget(action_schema, budget_names)


def parse_action_submission(submission: TaskSubmission) -> ParsedAction | None:
    """Parse a submission as a canonical action envelope when possible.

    Parameters
    ----------
    submission:
        Raw model submission plus optional integration-parsed payload.

    Returns
    -------
    ParsedAction or None
        Parsed action envelope, or ``None`` when neither parsed payload nor raw
        text contains the canonical action envelope.
    """

    parsed_from_payload = parse_action_envelope(submission.parsed)
    if parsed_from_payload is not None:
        return parsed_from_payload

    parsed_json = parse_json_object(submission.raw)
    if parsed_json is None:
        return None
    return parse_action_envelope(parsed_json)


def parse_action_envelope(values: Mapping[str, object]) -> ParsedAction | None:
    """Parse the canonical action envelope from a mapping.

    Parameters
    ----------
    values:
        Mapping that may contain ``action`` and ``arguments`` envelope fields.

    Returns
    -------
    ParsedAction or None
        Parsed action envelope, or ``None`` when required envelope fields are
        absent or malformed.
    """

    name_value = values.get(ACTION_NAME_FIELD)
    if not isinstance(name_value, str):
        return None

    arguments_value = values.get(ACTION_ARGUMENTS_FIELD)
    if isinstance(arguments_value, Mapping):
        return ParsedAction(
            name=name_value,
            arguments=cast(Mapping[str, object], arguments_value),
        )

    return None


def parse_json_object(raw: str) -> Mapping[str, object] | None:
    """Parse raw strict JSON into a mapping when possible.

    Parameters
    ----------
    raw:
        Raw JSON text.

    Returns
    -------
    Mapping[str, object] or None
        Parsed JSON object, or ``None`` when the text is not a strict JSON
        object.
    """

    try:
        decoded = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(decoded, Mapping):
        return cast(Mapping[str, Any], decoded)
    return None


def _reject_json_constant(raw: str) -> object:
    """Reject non-standard JSON numeric constants."""

    raise ValueError(f"invalid JSON numeric constant: {raw}")


def _budget_names_from_public_limits(public_limits: Mapping[str, object]) -> set[str]:
    """Return validated budget names from public limits."""

    budget_limits = public_limits.get("budget_limits")
    if not isinstance(budget_limits, Mapping):
        raise ValueError("public_limits must include budget_limits")
    budget_names: set[str] = set()
    for budget_name in budget_limits:
        if not isinstance(budget_name, str) or budget_name == "":
            raise ValueError("public budget name must be a non-empty string")
        budget_names.add(budget_name)
    return budget_names


def _validate_nested_consumes_budget(value: object, budget_names: set[str]) -> None:
    """Validate every nested consumes-budget map in a public schema."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "consumes_budget":
                if not isinstance(item, Mapping):
                    raise ValueError("action schema consumes_budget must be a mapping")
                _validate_consumes_budget_mapping(
                    cast(Mapping[str, object], item),
                    budget_names,
                    "action schema",
                )
            else:
                _validate_nested_consumes_budget(item, budget_names)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_nested_consumes_budget(item, budget_names)


def _validate_consumes_budget_mapping(
    consumes_budget: Mapping[str, object],
    budget_names: set[str],
    context: str,
) -> None:
    """Validate one budget consumption map against known budget names."""

    for budget_name, amount in consumes_budget.items():
        if not isinstance(budget_name, str) or budget_name == "":
            raise ValueError(f"{context} budget name must be a non-empty string")
        if budget_name not in budget_names:
            raise ValueError(f"{context} references unknown budget: {budget_name}")
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError(f"{context} budget consumption must be an integer")
        if amount < 0:
            raise ValueError(f"{context} budget consumption must be non-negative")
