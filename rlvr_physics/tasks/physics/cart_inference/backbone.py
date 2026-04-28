"""Authoritative cart inference backbone, parsing, and verification."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Mapping

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSubmission

MEASURE_POSITION_ACTION = "measure_position"
FINAL_ANSWER_ACTION = "final_answer"

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class SubmissionParseError(ValueError):
    """Raised when a model submission cannot be interpreted for this task."""


class ActionBudgetExceeded(RuntimeError):
    """Raised when a rollout exceeds the cart measurement action budget."""


@dataclass(frozen=True)
class CartInferenceState:
    """Authoritative state for one cart inference task instance.

    Parameters
    ----------
    initial_position_m:
        Public initial cart position in meters.
    initial_velocity_mps:
        Public initial cart velocity in meters per second.
    acceleration_mps2:
        Privileged constant acceleration in meters per second squared.
    target_time_s:
        Public target prediction time in seconds.
    min_measurement_time_s:
        Public minimum valid measurement time in seconds.
    max_measurement_time_s:
        Public maximum valid measurement time in seconds.
    measurement_noise_abs_m:
        Public absolute bound on deterministic measurement noise.
    answer_tolerance_abs_m:
        Privileged absolute tolerance for the final answer.
    exact_target_position_m:
        Privileged exact target position in meters.
    measurement_noise_seed:
        Privileged seed used to produce deterministic measurement noise.

    Attributes
    ----------
    initial_position_m:
        Public initial cart position in meters.
    initial_velocity_mps:
        Public initial cart velocity in meters per second.
    acceleration_mps2:
        Privileged constant acceleration in meters per second squared.
    target_time_s:
        Public target prediction time in seconds.
    min_measurement_time_s:
        Public minimum valid measurement time in seconds.
    max_measurement_time_s:
        Public maximum valid measurement time in seconds.
    measurement_noise_abs_m:
        Public absolute bound on deterministic measurement noise.
    answer_tolerance_abs_m:
        Privileged absolute tolerance for the final answer.
    exact_target_position_m:
        Privileged exact target position in meters.
    measurement_noise_seed:
        Privileged seed used to produce deterministic measurement noise.
    """

    initial_position_m: float
    initial_velocity_mps: float
    acceleration_mps2: float
    target_time_s: float
    min_measurement_time_s: float
    max_measurement_time_s: float
    measurement_noise_abs_m: float
    answer_tolerance_abs_m: float
    exact_target_position_m: float
    measurement_noise_seed: int


@dataclass(frozen=True)
class MeasurementResult:
    """Result of one public cart position measurement.

    Parameters
    ----------
    measurement_index:
        Zero-based accepted measurement index in this rollout.
    time_s:
        Measurement time in seconds.
    true_position_m:
        Privileged exact position at the measurement time.
    measured_position_m:
        Public noisy sensor reading in meters.
    noise_m:
        Privileged deterministic sensor noise in meters.

    Attributes
    ----------
    measurement_index:
        Zero-based accepted measurement index in this rollout.
    time_s:
        Measurement time in seconds.
    true_position_m:
        Privileged exact position at the measurement time.
    measured_position_m:
        Public noisy sensor reading in meters.
    noise_m:
        Privileged deterministic sensor noise in meters.
    """

    measurement_index: int
    time_s: float
    true_position_m: float
    measured_position_m: float
    noise_m: float


@dataclass(frozen=True)
class FinalAnswerEvaluation:
    """Verifier evaluation for a final target-position answer.

    Parameters
    ----------
    submitted_position_m:
        Parsed submitted target position in meters.
    exact_position_m:
        Privileged exact target position in meters.
    absolute_error_m:
        Absolute answer error in meters.
    tolerance_abs_m:
        Absolute tolerance for full credit in meters.
    correct:
        Whether the answer is inside tolerance.
    Attributes
    ----------
    submitted_position_m:
        Parsed submitted target position in meters.
    exact_position_m:
        Privileged exact target position in meters.
    absolute_error_m:
        Absolute answer error in meters.
    tolerance_abs_m:
        Absolute tolerance for full credit in meters.
    correct:
        Whether the answer is inside tolerance.
    """

    submitted_position_m: float
    exact_position_m: float
    absolute_error_m: float
    tolerance_abs_m: float
    correct: bool


@dataclass(frozen=True)
class ParsedAction:
    """Structured action parsed from a model submission.

    Parameters
    ----------
    name:
        Action name, such as ``measure_position``.
    arguments:
        Action argument payload.

    Attributes
    ----------
    name:
        Action name, such as ``measure_position``.
    arguments:
        Action argument payload.
    """

    name: str
    arguments: Mapping[str, object]


class CartInferenceBackbone:
    """Authoritative executable backbone for one cart inference rollout.

    Parameters
    ----------
    instance:
        Immutable cart inference task instance used to initialize canonical
        task state and rollout limits.

    Attributes
    ----------
    instance:
        Immutable cart inference task instance.
    """

    def __init__(self, instance: TaskInstance) -> None:
        """Initialize canonical state and task-rule counters.

        Parameters
        ----------
        instance:
            Immutable cart inference task instance.
        """

        self.instance = instance
        self._state = state_from_instance(instance)
        self._action_budget = _required_action_budget(instance)
        self._measurements_used = 0

    @property
    def state(self) -> CartInferenceState:
        """Return the canonical immutable physics state.

        Returns
        -------
        CartInferenceState
            Authoritative state parsed from the task instance.
        """

        return self._state

    @property
    def action_budget(self) -> int:
        """Return the maximum number of measurement actions.

        Returns
        -------
        int
            Public measurement action budget for this rollout.
        """

        return self._action_budget

    @property
    def measurements_used(self) -> int:
        """Return the number of accepted measurements in this rollout.

        Returns
        -------
        int
            Count of accepted measurement actions.
        """

        return self._measurements_used

    @property
    def measurements_remaining(self) -> int:
        """Return the remaining public measurement action budget.

        Returns
        -------
        int
            Number of measurement actions still available.
        """

        return self._action_budget - self._measurements_used

    def reset_rollout(self) -> None:
        """Reset rollout-local task-rule counters."""

        self._measurements_used = 0

    def parse_action(self, submission: TaskSubmission) -> ParsedAction:
        """Parse a submission as a structured task action.

        Parameters
        ----------
        submission:
            Raw model submission plus optional integration-parsed payload.

        Returns
        -------
        ParsedAction
            Parsed action name and arguments.
        """

        return parse_action_submission(submission)

    def measure(self, action: ParsedAction) -> MeasurementResult:
        """Apply one public position measurement action.

        Parameters
        ----------
        action:
            Parsed measurement action.

        Returns
        -------
        MeasurementResult
            Public noisy reading plus privileged exact components.

        Raises
        ------
        ActionBudgetExceeded
            Raised when no measurement actions remain.
        SubmissionParseError
            Raised when the action is not a valid measurement request.
        ValueError
            Raised when the measurement time is outside the public range.
        """

        if action.name != MEASURE_POSITION_ACTION:
            raise SubmissionParseError(f"expected action: {MEASURE_POSITION_ACTION}")
        if self._measurements_used >= self._action_budget:
            raise ActionBudgetExceeded("action_budget_exceeded")
        time_s = _required_numeric_argument(action, "time")
        measurement = measure_position(
            state=self._state,
            time_s=time_s,
            measurement_index=self._measurements_used,
        )
        self._measurements_used += 1
        return measurement

    def final_answer_from_action(self, action: ParsedAction) -> float:
        """Extract a final-answer value from a parsed action.

        Parameters
        ----------
        action:
            Parsed final-answer action.

        Returns
        -------
        float
            Submitted target position in meters.

        Raises
        ------
        SubmissionParseError
            Raised when the action is not a valid final-answer request.
        """

        if action.name != FINAL_ANSWER_ACTION:
            raise SubmissionParseError(f"expected action: {FINAL_ANSWER_ACTION}")
        return _required_numeric_argument(action, "x")

    def evaluate_final_answer(
        self, submitted_position_m: float
    ) -> FinalAnswerEvaluation:
        """Evaluate a submitted target-position answer.

        Parameters
        ----------
        submitted_position_m:
            Submitted target position in meters.

        Returns
        -------
        FinalAnswerEvaluation
            Correctness and privileged error details.
        """

        return evaluate_final_answer(self._state, submitted_position_m)

    def position_at_time(self, time_s: float) -> float:
        """Return the exact cart position at one time.

        Parameters
        ----------
        time_s:
            Elapsed time in seconds.

        Returns
        -------
        float
            Exact position in meters.
        """

        return position_at_time(self._state, time_s)


def state_from_instance(instance: TaskInstance) -> CartInferenceState:
    """Build authoritative task state from an immutable instance.

    Parameters
    ----------
    instance:
        Immutable cart inference task instance.

    Returns
    -------
    CartInferenceState
        Authoritative state assembled from public and privileged payloads.

    Raises
    ------
    KeyError
        Raised when a required payload field is missing.
    TypeError
        Raised when a required payload field has the wrong type.
    """

    measurement_range = _mapping_field(
        instance.public_payload, "measurement_time_range_s"
    )
    return CartInferenceState(
        initial_position_m=_float_field(instance.public_payload, "initial_position_m"),
        initial_velocity_mps=_float_field(
            instance.public_payload, "initial_velocity_mps"
        ),
        acceleration_mps2=_float_field(
            instance.privileged_payload, "acceleration_mps2"
        ),
        target_time_s=_float_field(instance.public_payload, "target_time_s"),
        min_measurement_time_s=_float_field(measurement_range, "min"),
        max_measurement_time_s=_float_field(measurement_range, "max"),
        measurement_noise_abs_m=_float_field(
            instance.public_payload, "measurement_noise_abs_m"
        ),
        answer_tolerance_abs_m=_float_field(
            instance.privileged_payload, "answer_tolerance_abs_m"
        ),
        exact_target_position_m=_float_field(
            instance.privileged_payload, "exact_target_position_m"
        ),
        measurement_noise_seed=_int_field(
            instance.privileged_payload, "measurement_noise_seed"
        ),
    )


def position_from_values(
    initial_position_m: float,
    initial_velocity_mps: float,
    acceleration_mps2: float,
    time_s: float,
) -> float:
    """Return a constant-acceleration position from explicit values.

    Parameters
    ----------
    initial_position_m:
        Initial position in meters.
    initial_velocity_mps:
        Initial velocity in meters per second.
    acceleration_mps2:
        Constant acceleration in meters per second squared.
    time_s:
        Elapsed time in seconds.

    Returns
    -------
    float
        Position in meters at ``time_s``.
    """

    return (
        initial_position_m
        + initial_velocity_mps * time_s
        + 0.5 * acceleration_mps2 * time_s * time_s
    )


def position_at_time(state: CartInferenceState, time_s: float) -> float:
    """Return the exact cart position at one time.

    Parameters
    ----------
    state:
        Authoritative cart task state.
    time_s:
        Elapsed time in seconds.

    Returns
    -------
    float
        Exact position in meters.
    """

    return position_from_values(
        initial_position_m=state.initial_position_m,
        initial_velocity_mps=state.initial_velocity_mps,
        acceleration_mps2=state.acceleration_mps2,
        time_s=time_s,
    )


def measure_position(
    state: CartInferenceState, time_s: float, measurement_index: int
) -> MeasurementResult:
    """Measure cart position with deterministic bounded noise.

    Parameters
    ----------
    state:
        Authoritative cart task state.
    time_s:
        Measurement time in seconds.
    measurement_index:
        Zero-based accepted measurement index in the rollout.

    Returns
    -------
    MeasurementResult
        Public noisy reading plus privileged exact components.

    Raises
    ------
    ValueError
        Raised when ``time_s`` is outside the public measurement range.
    """

    if (
        not isfinite(time_s)
        or time_s < state.min_measurement_time_s
        or time_s > state.max_measurement_time_s
    ):
        raise ValueError(
            "measurement time must be between "
            f"{state.min_measurement_time_s:g}s and "
            f"{state.max_measurement_time_s:g}s"
        )
    true_position_m = position_at_time(state, time_s)
    noise_m = _measurement_noise_m(state, time_s, measurement_index)
    return MeasurementResult(
        measurement_index=measurement_index,
        time_s=time_s,
        true_position_m=true_position_m,
        measured_position_m=true_position_m + noise_m,
        noise_m=noise_m,
    )


def evaluate_final_answer(
    state: CartInferenceState, submitted_position_m: float
) -> FinalAnswerEvaluation:
    """Evaluate a submitted target-position answer.

    Parameters
    ----------
    state:
        Authoritative cart task state.
    submitted_position_m:
        Submitted target position in meters.

    Returns
    -------
    FinalAnswerEvaluation
        Correctness and privileged error details.
    """

    absolute_error_m = abs(submitted_position_m - state.exact_target_position_m)
    correct = absolute_error_m <= state.answer_tolerance_abs_m
    return FinalAnswerEvaluation(
        submitted_position_m=submitted_position_m,
        exact_position_m=state.exact_target_position_m,
        absolute_error_m=absolute_error_m,
        tolerance_abs_m=state.answer_tolerance_abs_m,
        correct=correct,
    )


def parse_action_submission(submission: TaskSubmission) -> ParsedAction:
    """Parse a model submission as a structured task action.

    Parameters
    ----------
    submission:
        Raw model submission plus optional integration-parsed payload.

    Returns
    -------
    ParsedAction
        Parsed action name and argument mapping.

    Raises
    ------
    SubmissionParseError
        Raised when no supported action shape can be decoded.
    """

    parsed_from_payload = _parse_action_mapping(submission.parsed)
    if parsed_from_payload is not None:
        return parsed_from_payload

    parsed_json = _json_object(submission.raw)
    if parsed_json is not None:
        parsed_from_json = _parse_action_mapping(parsed_json)
        if parsed_from_json is not None:
            return parsed_from_json

    raw = submission.raw.strip()
    command_match = re.fullmatch(
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<argument>.*)\)", raw
    )
    if command_match is not None:
        name = command_match.group("name")
        argument_name = "x" if name == FINAL_ANSWER_ACTION else "time"
        return ParsedAction(
            name=name,
            arguments={
                argument_name: _parse_single_number(command_match.group("argument"))
            },
        )

    raise SubmissionParseError("could not parse action submission")


def _measurement_noise_m(
    state: CartInferenceState, time_s: float, measurement_index: int
) -> float:
    """Return deterministic measurement noise for one tool call."""

    if state.measurement_noise_abs_m == 0.0:
        return 0.0
    encoded = (
        f"{state.measurement_noise_seed}|{measurement_index}|{time_s:.9f}"
    ).encode("utf-8")
    digest = sha256(encoded).hexdigest()
    unit = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return (2.0 * unit - 1.0) * state.measurement_noise_abs_m


def _required_action_budget(instance: TaskInstance) -> int:
    """Return the required measurement action budget from an instance."""

    if instance.action_budget is None:
        raise RuntimeError("cart inference instances require action_budget")
    return instance.action_budget


def _required_numeric_argument(action: ParsedAction, name: str) -> float:
    """Read one required numeric action argument."""

    value = action.arguments.get(name)
    if value is None:
        raise SubmissionParseError(f"missing argument: {name}")
    if isinstance(value, bool):
        raise SubmissionParseError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
    elif isinstance(value, str):
        try:
            numeric_value = float(value)
        except ValueError as error:
            raise SubmissionParseError(f"{name} must be numeric") from error
    else:
        raise SubmissionParseError(f"{name} must be numeric")
    if not isfinite(numeric_value):
        raise SubmissionParseError(f"{name} must be finite")
    return numeric_value


def _parse_action_mapping(values: Mapping[str, object]) -> ParsedAction | None:
    """Parse action data from a mapping-like payload."""

    name_value = values.get("name")
    if not isinstance(name_value, str):
        name_value = values.get("tool_name")
    if not isinstance(name_value, str):
        action_value = values.get("action")
        if isinstance(action_value, str) and action_value in {
            MEASURE_POSITION_ACTION,
            FINAL_ANSWER_ACTION,
        }:
            name_value = action_value
    if not isinstance(name_value, str):
        return None

    arguments_value = values.get("arguments")
    if isinstance(arguments_value, Mapping):
        return ParsedAction(name=name_value, arguments=arguments_value)

    arguments: dict[str, object] = {}
    time_value = values.get("time")
    if time_value is not None:
        arguments["time"] = time_value
    x_value = values.get("x")
    if x_value is not None:
        arguments["x"] = x_value
    return ParsedAction(name=name_value, arguments=arguments)


def _json_object(raw: str) -> Mapping[str, object] | None:
    """Parse raw JSON into a mapping when possible."""

    try:
        decoded = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(decoded, Mapping):
        return decoded
    return None


def _reject_json_constant(raw: str) -> object:
    """Reject non-standard JSON numeric constants."""

    raise ValueError(f"invalid JSON numeric constant: {raw}")


def _parse_single_number(raw: str) -> float:
    """Parse one number from raw text."""

    match = _NUMBER_PATTERN.search(raw)
    if match is None:
        raise SubmissionParseError("expected a numeric value")
    return float(match.group(0))


def _float_field(values: Mapping[str, object], name: str) -> float:
    """Return a required numeric field from a mapping."""

    value = values[name]
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"{name} must be numeric")


def _int_field(values: Mapping[str, object], name: str) -> int:
    """Return a required integer field from a mapping."""

    value = values[name]
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    raise TypeError(f"{name} must be an integer")


def _mapping_field(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    """Return a required mapping field from a mapping."""

    value = values[name]
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{name} must be a mapping")
