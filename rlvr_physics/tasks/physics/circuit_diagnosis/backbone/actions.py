"""Circuit diagnosis action parsing helpers."""

from math import isfinite

from rlvr_physics.core.submissions import (
    ParsedAction,
    TaskSubmission,
    parse_action_submission as parse_core_action_submission,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.constants import (
    ACTION_SUBMISSION_PARSE_ERROR,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    SubmissionParseError,
)


def parse_action_submission(submission: TaskSubmission) -> ParsedAction:
    """Parse a model submission as a structured action."""

    parsed_action = parse_core_action_submission(submission)
    if parsed_action is not None:
        return parsed_action
    raise SubmissionParseError(ACTION_SUBMISSION_PARSE_ERROR)


def required_str_argument(action: ParsedAction, name: str) -> str:
    """Read one required string action argument."""

    value = action.arguments.get(name)
    if isinstance(value, str) and value != "":
        return value
    raise SubmissionParseError(f"{name} must be a non-empty string")


def required_numeric_argument(action: ParsedAction, name: str) -> float:
    """Read one required numeric action argument."""

    value = action.arguments.get(name)
    if value is None:
        raise SubmissionParseError(f"missing argument: {name}")
    if isinstance(value, bool):
        raise SubmissionParseError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
    else:
        raise SubmissionParseError(f"{name} must be numeric")
    if not isfinite(numeric_value):
        raise SubmissionParseError(f"{name} must be finite")
    return numeric_value


def positive_numeric_argument(action: ParsedAction, name: str) -> float:
    """Read one positive numeric action argument."""

    value = required_numeric_argument(action, name)
    if value <= 0.0:
        raise SubmissionParseError(f"{name} must be positive")
    return value


def string_tuple_argument(
    action: ParsedAction, plural_name: str, singular_name: str
) -> tuple[str, ...]:
    """Read a plural list or singular string final-answer argument."""

    plural_value = action.arguments.get(plural_name)
    if isinstance(plural_value, list | tuple):
        values: list[str] = []
        for item in plural_value:
            if not isinstance(item, str) or item == "":
                raise SubmissionParseError(f"{plural_name} entries must be strings")
            values.append(item)
        return tuple(values)
    singular_value = action.arguments.get(singular_name)
    if isinstance(singular_value, str) and singular_value != "":
        return (singular_value,)
    raise SubmissionParseError(
        f"missing argument: {plural_name} must be a list of strings"
    )
