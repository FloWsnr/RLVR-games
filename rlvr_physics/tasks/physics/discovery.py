"""Interactive physics equation discovery tasks."""

from collections.abc import Callable
from dataclasses import dataclass, field
import ast
from functools import lru_cache
from importlib import resources
import json
import math
from random import Random
import re
from typing import Mapping

from rlvr_physics.core.instances import (
    TaskInstance,
    TaskLimits,
    freeze_mapping,
    mapping_to_dict,
    require_int,
    require_mapping,
    require_str,
    stable_hash,
)
from rlvr_physics.core.rendering import RenderedObservation, text_observation
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.core.trajectory import TaskTrajectory, TrajectoryEvent

PHYSICS_DISCOVERY_KIND = "physics.discovery.v1"
PHYSICS_DOMAIN = "physics"
PHYSICS_DISCOVERY_PRIOR_MODES = (
    "default",
    "no_context",
    "no_description",
    "no_description_anonymous",
)
PHYSICS_DISCOVERY_SOURCE = "physgym.curated_subset"
PHYSICS_DISCOVERY_RECORDS_FILE = "physgym_curated_records.json"
_DEFAULT_RANGE = (0.5, 5.0)
_ALLOWED_FUNCTION_NAMES = "sqrt, sin, cos, tan, exp, log, abs"


@dataclass(frozen=True)
class PhysicsDiscoveryRecord:
    """One source law used to build physics discovery instances.

    Parameters
    ----------
    source_id:
        Original PhysGym source identifier.
    tag:
        Coarse physics domain tag.
    context:
        Public problem context used in the richest prior mode.
    equation:
        Ground-truth scalar expression in terms of the input variables.
    input_variables:
        Mapping from variable name to public description.
    output_variable:
        Mapping with one output variable name and public description.
    """

    source_id: int
    tag: str
    context: str
    equation: str
    input_variables: Mapping[str, object]
    output_variable: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze record mappings after construction."""

        object.__setattr__(
            self, "input_variables", freeze_mapping(self.input_variables)
        )
        object.__setattr__(
            self, "output_variable", freeze_mapping(self.output_variable)
        )


@dataclass(frozen=True)
class ExperimentObservation:
    """Public result from one controlled experiment."""

    sample_id: int
    inputs: Mapping[str, object]
    output: float

    def __post_init__(self) -> None:
        """Freeze input mapping after construction."""

        object.__setattr__(self, "inputs", freeze_mapping(self.inputs))

    def as_public_dict(self) -> Mapping[str, object]:
        """Return trainer-safe observation data."""

        return freeze_mapping(
            {
                "sample_id": self.sample_id,
                "inputs": self.inputs,
                "output": self.output,
            }
        )


@dataclass(frozen=True)
class HypothesisAttempt:
    """Public result from one tested hypothesis."""

    hypothesis_id: int
    expression: str
    score: float
    correct: bool

    def as_public_dict(self) -> Mapping[str, object]:
        """Return trainer-safe hypothesis data."""

        return freeze_mapping(
            {
                "hypothesis_id": self.hypothesis_id,
                "expression": self.expression,
                "score": self.score,
                "correct": self.correct,
            }
        )


@dataclass(frozen=True)
class HypothesisEvaluation:
    """Numeric hidden-point evaluation result."""

    accepted: bool
    score: float
    correct: bool
    reason: str
    valid_points: int
    max_relative_error: float
    mean_relative_error: float


@dataclass(frozen=True)
class ParsedDiscoveryAction:
    """Interpreted discovery action."""

    action_type: str
    inputs: Mapping[str, object] = field(default_factory=dict)
    equation: str = ""

    def __post_init__(self) -> None:
        """Freeze parsed action inputs after construction."""

        object.__setattr__(self, "inputs", freeze_mapping(self.inputs))


def physics_discovery_records() -> tuple[PhysicsDiscoveryRecord, ...]:
    """Return the packaged PhysGym-derived discovery records."""

    return _load_physics_discovery_records()


def physics_discovery_task_spec(
    seed: int,
    sample_quota: int,
    hypothesis_quota: int,
    prior_mode: str,
) -> TaskSpec:
    """Return the task spec for interactive physics discovery.

    Parameters
    ----------
    seed:
        Seed used for deterministic hidden verification points.
    sample_quota:
        Maximum accepted experiments before hypotheses must rely on history.
    hypothesis_quota:
        Maximum accepted hypothesis tests.
    prior_mode:
        Information exposure mode.
    """

    _validate_prior_mode(prior_mode)
    _validate_positive_quota(sample_quota, "sample_quota")
    _validate_positive_quota(hypothesis_quota, "hypothesis_quota")
    return TaskSpec(
        kind=PHYSICS_DISCOVERY_KIND,
        domain=PHYSICS_DOMAIN,
        source=SourceSpec(
            source_type=PHYSICS_DISCOVERY_SOURCE,
            seed=seed,
            parameters={
                "sample_quota": sample_quota,
                "hypothesis_quota": hypothesis_quota,
                "prior_mode": prior_mode,
                "records": len(physics_discovery_records()),
            },
        ),
        renderers=(RendererSpec(renderer_type="text"),),
        verifier=VerifierSpec(
            verifier_type="hidden_numeric_equivalence",
            parameters={"hidden_points": 24, "relative_tolerance": 1e-5},
        ),
        reward=RewardSpec(
            reward_type="hypothesis_fit_with_experiment_cost",
            parameters={
                "correct": 1.0,
                "experiment_cost": -0.01,
                "invalid": -0.05,
            },
        ),
        limits=TaskLimits(
            max_turns=sample_quota + hypothesis_quota,
            action_budget=sample_quota + hypothesis_quota,
        ),
        metadata={
            "exports": {
                "environment": {"actions": ("run_experiment", "submit_hypothesis")}
            },
            "source": "PhysGym curated records",
        },
    )


def make_physics_discovery_instance(
    source_id: int,
    seed: int,
    prior_mode: str,
    sample_quota: int,
    hypothesis_quota: int,
) -> TaskInstance:
    """Create one immutable interactive physics discovery task instance.

    Parameters
    ----------
    source_id:
        Embedded PhysGym-derived source law id.
    seed:
        Seed used for deterministic hidden verification points.
    prior_mode:
        Information exposure mode.
    sample_quota:
        Maximum accepted experiments for the session.
    hypothesis_quota:
        Maximum accepted hypothesis tests for the session.
    """

    _validate_positive_quota(sample_quota, "sample_quota")
    _validate_positive_quota(hypothesis_quota, "hypothesis_quota")
    _validate_prior_mode(prior_mode)
    record = _record_by_source_id(source_id)
    variable_names = tuple(str(name) for name in record.input_variables.keys())
    output_name = _single_mapping_key(record.output_variable, "output_variable")

    if prior_mode == "no_description_anonymous":
        variable_mapping = {
            original: f"var_{index + 1}"
            for index, original in enumerate(variable_names)
        }
        visible_output_name = "var_obs"
    else:
        variable_mapping = {original: original for original in variable_names}
        visible_output_name = output_name

    visible_inputs = _visible_input_variables(record, prior_mode, variable_mapping)
    visible_output = _visible_output_variable(record, prior_mode, visible_output_name)
    visible_equation = _rename_symbols(record.equation, variable_mapping)
    parameter_ranges = {
        visible_name: _DEFAULT_RANGE for visible_name in visible_inputs.keys()
    }
    hidden_points = _make_hidden_points(
        seed=seed,
        source_id=source_id,
        variable_names=tuple(visible_inputs.keys()),
        count=24,
        parameter_ranges=parameter_ranges,
    )
    context = _visible_context(record.context, prior_mode)
    task_id = (
        "physics-discovery-"
        + stable_hash(
            {
                "source_id": source_id,
                "seed": seed,
                "prior_mode": prior_mode,
                "sample_quota": sample_quota,
                "hypothesis_quota": hypothesis_quota,
                "visible_equation": visible_equation,
                "hidden_points": hidden_points,
            }
        )[:16]
    )
    return TaskInstance(
        task_id=task_id,
        kind=PHYSICS_DISCOVERY_KIND,
        domain=PHYSICS_DOMAIN,
        seed=seed,
        public_payload={
            "source_id": source_id,
            "source": PHYSICS_DISCOVERY_SOURCE,
            "tag": record.tag,
            "prior_mode": prior_mode,
            "problem": context,
            "input_variables": visible_inputs,
            "output_variable": visible_output,
            "parameter_ranges": parameter_ranges,
            "sample_quota": sample_quota,
            "hypothesis_quota": hypothesis_quota,
        },
        privileged_payload={
            "equation": visible_equation,
            "source_equation": record.equation,
            "variable_mapping": {
                visible: original for original, visible in variable_mapping.items()
            },
            "hidden_points": hidden_points,
        },
        limits=TaskLimits(
            max_turns=sample_quota + hypothesis_quota,
            action_budget=sample_quota + hypothesis_quota,
        ),
        metadata={
            "source": PHYSICS_DISCOVERY_SOURCE,
            "source_id": source_id,
            "tag": record.tag,
            "prior_mode": prior_mode,
        },
    )


def render_physics_discovery_text(
    instance: TaskInstance,
    observations: tuple[ExperimentObservation, ...],
    hypotheses: tuple[HypothesisAttempt, ...],
    samples_used: int,
    hypotheses_used: int,
) -> RenderedObservation:
    """Render a physics discovery state as text.

    Parameters
    ----------
    instance:
        Immutable discovery instance.
    observations:
        Accepted experiment observations.
    hypotheses:
        Tested hypothesis records.
    samples_used:
        Number of accepted experiments.
    hypotheses_used:
        Number of accepted hypothesis tests.
    """

    input_variables = require_mapping(
        instance.public_payload["input_variables"], "input_variables"
    )
    output_variable = require_mapping(
        instance.public_payload["output_variable"], "output_variable"
    )
    parameter_ranges = require_mapping(
        instance.public_payload["parameter_ranges"], "parameter_ranges"
    )
    sample_quota = require_int(instance.public_payload["sample_quota"], "sample_quota")
    hypothesis_quota = require_int(
        instance.public_payload["hypothesis_quota"], "hypothesis_quota"
    )
    problem = require_str(instance.public_payload["problem"], "problem")
    prior_mode = require_str(instance.public_payload["prior_mode"], "prior_mode")
    output_name = _single_mapping_key(output_variable, "output_variable")
    lines = [
        "Physics discovery",
        f"Prior mode: {prior_mode}",
        "",
        "Problem:",
        problem,
        "",
        "Controllable variables:",
    ]
    for name, description in input_variables.items():
        low, high = _range_pair(parameter_ranges[name], f"range for {name}")
        lines.append(f"- {name}: {description} | allowed range [{low}, {high}]")
    lines.extend(
        [
            "",
            "Observable:",
            f"- {output_name}: {output_variable[output_name]}",
            "",
            "Budget:",
            f"- Experiments used: {samples_used}/{sample_quota}",
            f"- Hypotheses tested: {hypotheses_used}/{hypothesis_quota}",
            "",
            "Submit JSON to run an experiment:",
            _example_experiment_action(tuple(input_variables.keys())),
            "",
            "Submit JSON or final text to propose a hypothesis:",
            '{"action": "submit_hypothesis", "equation": "expression using the variables"}',
            (
                "Allowed expression syntax: +, -, *, /, **, parentheses, constants "
                f"pi/e, and functions {_ALLOWED_FUNCTION_NAMES}."
            ),
            "",
            "Observation history:",
        ]
    )
    if observations:
        for observation in observations:
            lines.append(_format_observation(observation, output_name))
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Hypothesis tests:")
    if hypotheses:
        for hypothesis in hypotheses:
            lines.append(
                f"- {hypothesis.hypothesis_id}: {hypothesis.expression} "
                f"| score={hypothesis.score:.6f} | correct={hypothesis.correct}"
            )
    else:
        lines.append("- none")
    return text_observation("text", "\n".join(lines))


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

    candidate = _extract_hypothesis_expression(expression)
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
        variables = _float_mapping(point)
        try:
            true_value = _evaluate_expression(true_equation, variables)
            candidate_value = _evaluate_expression(candidate, variables)
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


class PhysicsDiscoverySession:
    """Stateful interactive physics discovery task session."""

    def __init__(self, instance: TaskInstance, renderer: str) -> None:
        self._instance = instance
        self._renderer = renderer
        self._trajectory = TaskTrajectory(task_id=instance.task_id, session_id="")
        self._turn: TaskTurn | None = None
        self._observations: list[ExperimentObservation] = []
        self._hypotheses: list[HypothesisAttempt] = []
        self._submissions = 0
        self._samples_used = 0
        self._hypotheses_used = 0

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh discovery rollout."""

        session_id = new_session_id(self._instance.task_id, seed)
        self._trajectory = TaskTrajectory(
            task_id=self._instance.task_id, session_id=session_id
        )
        self._observations = []
        self._hypotheses = []
        self._submissions = 0
        self._samples_used = 0
        self._hypotheses_used = 0
        self._turn = self._make_turn()
        self._trajectory.append(
            "reset",
            0,
            {
                "task_id": self._instance.task_id,
                "renderer": self._renderer,
                "source_id": self._instance.public_payload["source_id"],
            },
            {"instance_hash": self._instance.content_hash()},
        )
        self._append_observation_event()
        return TaskResetResult(
            session_id=session_id, turn=self._turn, trajectory=self._trajectory
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current discovery turn."""

        return self._turn

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the verified discovery trajectory."""

        return self._trajectory

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Apply one discovery action or final hypothesis submission."""

        if self._turn is None:
            event = self._trajectory.append(
                "invalid_submission",
                self._submissions,
                {"reason": "session_finished"},
                {},
            )
            return TaskStepResult(
                accepted=False,
                reward=0.0,
                score=None,
                terminal=True,
                truncated=False,
                observation=None,
                public_info={"reason": "session_finished"},
                debug_info={},
                events=(event,),
            )

        turn_index = self._turn.turn_index
        self._submissions += 1
        submit_event = self._trajectory.append(
            "submission",
            turn_index,
            {"kind": submission.kind, "raw": submission.raw},
            {},
        )
        try:
            action = _parse_discovery_action(submission)
        except ValueError as error:
            invalid_event = self._trajectory.append(
                "invalid_action", turn_index, {"reason": str(error)}, {}
            )
            return self._invalid_result(str(error), (submit_event, invalid_event))

        if action.action_type == "run_experiment":
            return self._run_experiment(action, turn_index, submit_event)
        if action.action_type == "submit_hypothesis":
            return self._submit_hypothesis(action, turn_index, submit_event)
        invalid_event = self._trajectory.append(
            "invalid_action",
            turn_index,
            {"reason": "unknown_action", "action_type": action.action_type},
            {},
        )
        return self._invalid_result("unknown_action", (submit_event, invalid_event))

    def _run_experiment(
        self,
        action: ParsedDiscoveryAction,
        turn_index: int,
        submit_event: TrajectoryEvent,
    ) -> TaskStepResult:
        sample_quota = require_int(
            self._instance.public_payload["sample_quota"], "sample_quota"
        )
        if self._samples_used >= sample_quota:
            invalid_event = self._trajectory.append(
                "invalid_action", turn_index, {"reason": "sample_quota_exceeded"}, {}
            )
            return self._invalid_result(
                "sample_quota_exceeded", (submit_event, invalid_event)
            )
        try:
            inputs = _validate_experiment_inputs(self._instance, action.inputs)
            output = _evaluate_expression(
                require_str(self._instance.privileged_payload["equation"], "equation"),
                inputs,
            )
        except (ValueError, TypeError, ArithmeticError, OverflowError) as error:
            invalid_event = self._trajectory.append(
                "invalid_action",
                turn_index,
                {"reason": "invalid_experiment", "error": str(error)},
                {},
            )
            return self._invalid_result(
                "invalid_experiment", (submit_event, invalid_event)
            )

        self._samples_used += 1
        observation = ExperimentObservation(
            sample_id=self._samples_used,
            inputs=inputs,
            output=output,
        )
        self._observations.append(observation)
        truncated = self._submissions >= self._instance.limits.max_turns
        self._turn = None if truncated else self._make_turn()
        experiment_event = self._trajectory.append(
            "experiment",
            turn_index,
            {
                "sample_id": observation.sample_id,
                "inputs": observation.inputs,
                "output": observation.output,
                "samples_used": self._samples_used,
                "truncated": truncated,
            },
            {},
        )
        events = [submit_event, experiment_event]
        if self._turn is not None:
            events.append(self._append_observation_event())
        return TaskStepResult(
            accepted=True,
            reward=-0.01,
            score=None,
            terminal=False,
            truncated=truncated,
            observation=self._turn,
            public_info={
                "reason": "experiment_run"
                if not truncated
                else "turn_budget_exhausted",
                "sample_id": observation.sample_id,
                "inputs": observation.inputs,
                "output": observation.output,
                "samples_used": self._samples_used,
            },
            debug_info={},
            events=tuple(events),
        )

    def _submit_hypothesis(
        self,
        action: ParsedDiscoveryAction,
        turn_index: int,
        submit_event: TrajectoryEvent,
    ) -> TaskStepResult:
        hypothesis_quota = require_int(
            self._instance.public_payload["hypothesis_quota"], "hypothesis_quota"
        )
        if self._hypotheses_used >= hypothesis_quota:
            invalid_event = self._trajectory.append(
                "invalid_action",
                turn_index,
                {"reason": "hypothesis_quota_exceeded"},
                {},
            )
            return self._invalid_result(
                "hypothesis_quota_exceeded", (submit_event, invalid_event)
            )

        evaluation = evaluate_physics_hypothesis(self._instance, action.equation)
        if not evaluation.accepted:
            invalid_event = self._trajectory.append(
                "invalid_hypothesis",
                turn_index,
                {"reason": evaluation.reason, "equation": action.equation},
                {"true_equation": self._instance.privileged_payload["equation"]},
            )
            return self._invalid_result(
                evaluation.reason, (submit_event, invalid_event)
            )

        self._hypotheses_used += 1
        attempt = HypothesisAttempt(
            hypothesis_id=self._hypotheses_used,
            expression=_extract_hypothesis_expression(action.equation),
            score=evaluation.score,
            correct=evaluation.correct,
        )
        self._hypotheses.append(attempt)
        terminal = evaluation.correct
        truncated = not terminal and (
            self._hypotheses_used >= hypothesis_quota
            or self._submissions >= self._instance.limits.max_turns
        )
        self._turn = None if terminal or truncated else self._make_turn()
        hypothesis_event = self._trajectory.append(
            "hypothesis",
            turn_index,
            {
                "hypothesis_id": attempt.hypothesis_id,
                "expression": attempt.expression,
                "score": evaluation.score,
                "correct": evaluation.correct,
                "terminal": terminal,
                "truncated": truncated,
            },
            {
                "true_equation": self._instance.privileged_payload["equation"],
                "valid_points": evaluation.valid_points,
                "max_relative_error": evaluation.max_relative_error,
                "mean_relative_error": evaluation.mean_relative_error,
            },
        )
        events = [submit_event, hypothesis_event]
        if self._turn is not None:
            events.append(self._append_observation_event())
        reason = _hypothesis_reason(evaluation.correct, truncated)
        return TaskStepResult(
            accepted=True,
            reward=evaluation.score,
            score=evaluation.score,
            terminal=terminal,
            truncated=truncated,
            observation=self._turn,
            public_info={
                "reason": reason,
                "hypothesis_id": attempt.hypothesis_id,
                "score": evaluation.score,
                "correct": evaluation.correct,
                "valid_points": evaluation.valid_points,
                "max_relative_error": evaluation.max_relative_error,
                "mean_relative_error": evaluation.mean_relative_error,
                "hypotheses_used": self._hypotheses_used,
            },
            debug_info={"true_equation": self._instance.privileged_payload["equation"]},
            events=tuple(events),
        )

    def _invalid_result(
        self, reason: str, events: tuple[TrajectoryEvent, ...]
    ) -> TaskStepResult:
        truncated = self._submissions >= self._instance.limits.max_turns
        if truncated:
            self._turn = None
        return TaskStepResult(
            accepted=False,
            reward=-0.05,
            score=None,
            terminal=False,
            truncated=truncated,
            observation=None if truncated else self._turn,
            public_info={"reason": reason, "submissions": self._submissions},
            debug_info={},
            events=events,
        )

    def _make_turn(self) -> TaskTurn:
        if self._renderer != "text":
            raise ValueError(f"unknown physics discovery renderer: {self._renderer}")
        observation = render_physics_discovery_text(
            instance=self._instance,
            observations=tuple(self._observations),
            hypotheses=tuple(self._hypotheses),
            samples_used=self._samples_used,
            hypotheses_used=self._hypotheses_used,
        )
        return TaskTurn(
            turn_index=self._submissions,
            observation=observation,
            submission_modes=("action", "final_text"),
            action_schema=_make_action_schema(self._instance),
            public_limits=self._instance.limits.as_public_dict(),
            public_info={
                "task_id": self._instance.task_id,
                "kind": self._instance.kind,
                "source_id": self._instance.public_payload["source_id"],
                "samples_used": self._samples_used,
                "hypotheses_used": self._hypotheses_used,
            },
        )

    def _append_observation_event(self) -> TrajectoryEvent:
        if self._turn is None:
            raise ValueError("cannot append observation event without a turn")
        return self._trajectory.append(
            "observation",
            self._turn.turn_index,
            {
                "renderer": self._renderer,
                "content_digests": self._turn.observation.content_digests(),
            },
            {
                "observations": [
                    observation.as_public_dict() for observation in self._observations
                ],
                "hypotheses": [
                    hypothesis.as_public_dict() for hypothesis in self._hypotheses
                ],
            },
        )


def _record_by_source_id(source_id: int) -> PhysicsDiscoveryRecord:
    for record in physics_discovery_records():
        if record.source_id == source_id:
            return record
    raise ValueError(f"unknown physics discovery source id: {source_id}")


@lru_cache(maxsize=1)
def _load_physics_discovery_records() -> tuple[PhysicsDiscoveryRecord, ...]:
    raw_text = (
        resources.files("rlvr_physics.tasks.physics.data")
        .joinpath(PHYSICS_DISCOVERY_RECORDS_FILE)
        .read_text(encoding="utf-8")
    )
    raw_records = json.loads(raw_text)
    if not isinstance(raw_records, list):
        raise ValueError("physics discovery records JSON must contain a list")
    records: list[PhysicsDiscoveryRecord] = []
    source_ids: set[int] = set()
    for index, raw_record in enumerate(raw_records):
        record = _parse_physics_discovery_record(raw_record, index)
        if record.source_id in source_ids:
            raise ValueError(
                f"duplicate physics discovery source_id: {record.source_id}"
            )
        source_ids.add(record.source_id)
        records.append(record)
    return tuple(records)


def _parse_physics_discovery_record(
    raw_record: object, index: int
) -> PhysicsDiscoveryRecord:
    if not isinstance(raw_record, dict):
        raise ValueError(f"physics discovery record {index} must be an object")
    source_id = raw_record.get("source_id")
    if not isinstance(source_id, int) or isinstance(source_id, bool):
        raise ValueError(f"physics discovery record {index} has invalid source_id")
    tag = raw_record.get("tag")
    context = raw_record.get("context")
    equation = raw_record.get("equation")
    input_variables = _require_string_mapping(
        raw_record.get("input_variables"),
        f"physics discovery record {source_id} has invalid input_variables",
    )
    output_variable = _require_string_mapping(
        raw_record.get("output_variable"),
        f"physics discovery record {source_id} has invalid output_variable",
    )
    if not isinstance(tag, str):
        raise ValueError(f"physics discovery record {source_id} has invalid tag")
    if not isinstance(context, str):
        raise ValueError(f"physics discovery record {source_id} has invalid context")
    if not isinstance(equation, str):
        raise ValueError(f"physics discovery record {source_id} has invalid equation")
    if len(output_variable) != 1:
        raise ValueError(
            f"physics discovery record {source_id} must have one output_variable"
        )
    return PhysicsDiscoveryRecord(
        source_id=source_id,
        tag=tag,
        context=context,
        equation=equation,
        input_variables=input_variables,
        output_variable=output_variable,
    )


def _require_string_mapping(value: object, error_message: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not value:
        raise ValueError(error_message)
    values: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(error_message)
        values[key] = item
    return values


def _validate_prior_mode(prior_mode: str) -> None:
    if prior_mode not in PHYSICS_DISCOVERY_PRIOR_MODES:
        raise ValueError(f"unknown physics discovery prior mode: {prior_mode}")


def _validate_positive_quota(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _visible_context(context: str, prior_mode: str) -> str:
    if prior_mode == "default":
        return context
    return "Unknown context."


def _visible_input_variables(
    record: PhysicsDiscoveryRecord,
    prior_mode: str,
    variable_mapping: Mapping[str, str],
) -> Mapping[str, object]:
    visible: dict[str, object] = {}
    for original, description in record.input_variables.items():
        visible_name = variable_mapping[str(original)]
        if prior_mode in ("no_description", "no_description_anonymous"):
            visible[visible_name] = "Some variable."
        else:
            visible[visible_name] = description
    return freeze_mapping(visible)


def _visible_output_variable(
    record: PhysicsDiscoveryRecord, prior_mode: str, visible_output_name: str
) -> Mapping[str, object]:
    description = record.output_variable[
        _single_mapping_key(record.output_variable, "output_variable")
    ]
    if prior_mode in ("no_description", "no_description_anonymous"):
        description = "Some variable."
    return freeze_mapping({visible_output_name: description})


def _rename_symbols(expression: str, variable_mapping: Mapping[str, str]) -> str:
    renamed = expression
    for original, visible in sorted(
        variable_mapping.items(), key=lambda item: len(item[0]), reverse=True
    ):
        renamed = re.sub(rf"\b{re.escape(original)}\b", visible, renamed)
    return renamed


def _make_hidden_points(
    seed: int,
    source_id: int,
    variable_names: tuple[str, ...],
    count: int,
    parameter_ranges: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    rng_seed = int(
        stable_hash(
            {
                "seed": seed,
                "source_id": source_id,
                "variable_names": variable_names,
                "count": count,
            }
        )[:16],
        16,
    )
    rng = Random(rng_seed)
    points: list[Mapping[str, object]] = []
    for _index in range(count):
        point: dict[str, object] = {}
        for name in variable_names:
            low, high = _range_pair(parameter_ranges[name], f"range for {name}")
            point[name] = round(rng.uniform(low, high), 6)
        points.append(freeze_mapping(point))
    return tuple(points)


def _hidden_points(instance: TaskInstance) -> tuple[Mapping[str, object], ...]:
    value = instance.privileged_payload["hidden_points"]
    if not isinstance(value, tuple):
        raise TypeError("hidden_points must be a tuple")
    points: list[Mapping[str, object]] = []
    for point in value:
        points.append(require_mapping(point, "hidden point"))
    return tuple(points)


def _validate_experiment_inputs(
    instance: TaskInstance, inputs: Mapping[str, object]
) -> Mapping[str, float]:
    input_variables = require_mapping(
        instance.public_payload["input_variables"], "input_variables"
    )
    parameter_ranges = require_mapping(
        instance.public_payload["parameter_ranges"], "parameter_ranges"
    )
    expected_names = set(str(name) for name in input_variables.keys())
    provided_names = set(str(name) for name in inputs.keys())
    missing = sorted(expected_names - provided_names)
    extra = sorted(provided_names - expected_names)
    if missing:
        raise ValueError(f"missing inputs: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unknown inputs: {', '.join(extra)}")
    validated: dict[str, float] = {}
    for name in sorted(expected_names):
        value = _coerce_float(inputs[name], name)
        low, high = _range_pair(parameter_ranges[name], f"range for {name}")
        if value < low or value > high:
            raise ValueError(f"{name} must be in [{low}, {high}]")
        validated[name] = value
    return validated


def _parse_discovery_action(submission: TaskSubmission) -> ParsedDiscoveryAction:
    if submission.kind == "final_text":
        return ParsedDiscoveryAction(
            action_type="submit_hypothesis",
            equation=_extract_hypothesis_expression(submission.raw),
        )
    raw = submission.raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("action_must_be_json") from error
    if not isinstance(parsed, dict):
        raise ValueError("action_must_be_json_object")
    action_value = parsed.get("action", parsed.get("type"))
    if not isinstance(action_value, str):
        if isinstance(parsed.get("equation"), str):
            action_value = "submit_hypothesis"
        else:
            raise ValueError("missing_action")
    action_type = action_value.strip()
    if action_type == "run_experiment":
        inputs = parsed.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("experiment_inputs_must_be_object")
        return ParsedDiscoveryAction(action_type=action_type, inputs=inputs)
    if action_type == "submit_hypothesis":
        equation = parsed.get("equation", parsed.get("hypothesis"))
        if not isinstance(equation, str):
            raise ValueError("hypothesis_equation_must_be_string")
        return ParsedDiscoveryAction(action_type=action_type, equation=equation)
    return ParsedDiscoveryAction(action_type=action_type)


def _make_action_schema(instance: TaskInstance) -> Mapping[str, object]:
    input_variables = require_mapping(
        instance.public_payload["input_variables"], "input_variables"
    )
    parameter_ranges = require_mapping(
        instance.public_payload["parameter_ranges"], "parameter_ranges"
    )
    input_properties: dict[str, object] = {}
    for name in input_variables.keys():
        low, high = _range_pair(parameter_ranges[name], f"range for {name}")
        input_properties[str(name)] = {
            "type": "number",
            "minimum": low,
            "maximum": high,
        }
    return {
        "type": "object",
        "oneOf": (
            {
                "properties": {
                    "action": {"const": "run_experiment"},
                    "inputs": {
                        "type": "object",
                        "properties": input_properties,
                        "required": tuple(input_properties.keys()),
                        "additionalProperties": False,
                    },
                },
                "required": ("action", "inputs"),
                "additionalProperties": False,
            },
            {
                "properties": {
                    "action": {"const": "submit_hypothesis"},
                    "equation": {"type": "string"},
                },
                "required": ("action", "equation"),
                "additionalProperties": False,
            },
        ),
    }


def _extract_hypothesis_expression(text: str) -> str:
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
            return _extract_hypothesis_expression(equation)

    lines = [
        line.strip()
        for line in stripped.replace("```python", "```").splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
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


def _evaluate_expression(expression: str, variables: Mapping[str, float]) -> float:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid expression syntax") from error
    value = _evaluate_ast_node(parsed.body, variables)
    result = _coerce_float(value, "expression result")
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
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
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
    }
    if name not in functions:
        raise ValueError(f"unsupported function: {name}")
    return functions[name]


def _absolute_value(value: float) -> float:
    """Return the absolute value as a float."""

    return abs(value)


def _checked_number(value: float) -> float:
    result = _coerce_float(value, "expression value")
    _ensure_reasonable_number(result)
    return result


def _ensure_reasonable_number(value: float) -> None:
    if not math.isfinite(value):
        raise ValueError("expression produced a non-finite value")
    if abs(value) > 1e100:
        raise ValueError("expression produced an unreasonably large value")


def _coerce_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_mapping(values: Mapping[str, object]) -> Mapping[str, float]:
    floats: dict[str, float] = {}
    for key, value in values.items():
        floats[str(key)] = _coerce_float(value, str(key))
    return floats


def _range_pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    low = _coerce_float(value[0], f"{name} lower bound")
    high = _coerce_float(value[1], f"{name} upper bound")
    if low >= high:
        raise ValueError(f"{name} lower bound must be less than upper bound")
    return low, high


def _single_mapping_key(values: Mapping[str, object], name: str) -> str:
    if len(values) != 1:
        raise ValueError(f"{name} must contain exactly one item")
    return str(next(iter(values.keys())))


def _format_observation(observation: ExperimentObservation, output_name: str) -> str:
    inputs = mapping_to_dict(observation.inputs)
    input_text = ", ".join(f"{name}={inputs[name]}" for name in sorted(inputs))
    return f"- {observation.sample_id}: inputs({input_text}) -> {output_name}={observation.output:.10g}"


def _example_experiment_action(variable_names: tuple[str, ...]) -> str:
    values = {
        name: round(1.0 + 0.5 * index, 3) for index, name in enumerate(variable_names)
    }
    return json.dumps({"action": "run_experiment", "inputs": values})


def _hypothesis_reason(correct: bool, truncated: bool) -> str:
    if correct:
        return "correct_hypothesis"
    if truncated:
        return "hypothesis_budget_exhausted"
    return "hypothesis_tested"
