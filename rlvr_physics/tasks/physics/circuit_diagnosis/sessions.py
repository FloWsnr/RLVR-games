"""Scalar session implementation for the circuit diagnosis task."""

from collections.abc import Mapping

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.submissions import (
    ACTION_ARGUMENTS_FIELD,
    ACTION_NAME_FIELD,
    InvalidSubmissionPolicy,
    JSON_LINE_FORMAT,
    ParsedAction,
    TaskSubmission,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.constants import (
    FINAL_ANSWER_ACTION,
    MEASURE_CURRENT_ACTION,
    MEASURE_VOLTAGE_ACTION,
    REPLACE_COMPONENT_ACTION,
    SET_SOURCE_ACTION,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    CircuitSimulationError,
    SubmissionParseError,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.payloads import (
    fault_payload,
    replacement_payload,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.repairs import (
    nominal_replacement_for_component,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.runtime import (
    CircuitDiagnosisBackbone,
    CurrentMeasurement,
    VoltageMeasurement,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitDefinition,
    ReplacementSpec,
    SourceSetting,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.verification import (
    FinalCircuitEvaluation,
    target_check_debug_payloads,
    target_check_public_payloads,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.budgets import (
    FINAL_ANSWER_BUDGET,
    PROBE_BUDGET,
    REPAIR_BUDGET,
    TURN_BUDGET,
    CircuitRolloutBudgetState,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.prompting import (
    circuit_initial_feedback,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.renderers import (
    CircuitRenderContext,
    render_circuit_observation,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.rewards import (
    CircuitRewardConfig,
    reward_accepted_probe,
    reward_accepted_repair,
    reward_budget_exceeded,
    reward_final_verification,
    reward_invalid_submission,
    reward_session_already_done,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    validate_circuit_renderer_type,
)

RETRYABLE_INVALID_SUBMISSION_POLICY = InvalidSubmissionPolicy(
    category="retryable_invalid_submission",
    consumes_budget={TURN_BUDGET: 1},
    terminal=False,
    truncated=False,
)
INVALID_FINAL_ANSWER_POLICY = InvalidSubmissionPolicy(
    category="invalid_final_answer",
    consumes_budget={TURN_BUDGET: 1, FINAL_ANSWER_BUDGET: 1},
    terminal=True,
    truncated=False,
)
BUDGET_EXCEEDED_POLICY = InvalidSubmissionPolicy(
    category="budget_exceeded",
    consumes_budget={TURN_BUDGET: 1},
    terminal=False,
    truncated=True,
)


def _invalid_policies() -> dict[str, InvalidSubmissionPolicy]:
    """Return invalid-submission policies keyed by category."""

    policies = (
        RETRYABLE_INVALID_SUBMISSION_POLICY,
        INVALID_FINAL_ANSWER_POLICY,
        BUDGET_EXCEEDED_POLICY,
    )
    return {policy.category: policy for policy in policies}


class CircuitDiagnosisSession:
    """Scalar runtime session for one circuit diagnosis instance."""

    def __init__(
        self,
        instance: TaskInstance,
        renderer_type: str,
        reward_config: CircuitRewardConfig,
    ) -> None:
        """Initialize a mutable scalar runtime session."""

        validate_circuit_renderer_type(renderer_type)
        self.instance = instance
        self._renderer_type = renderer_type
        self._backbone = CircuitDiagnosisBackbone(instance)
        self._budget_state = CircuitRolloutBudgetState(instance.budget_limits)
        self._reward_config = reward_config
        self._session_id: str | None = None
        self._turn: TaskTurn | None = None

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh rollout and return the first model-facing turn."""

        self._session_id = new_session_id(self.instance.task_id, seed)
        self._budget_state.reset()
        self._backbone.reset_rollout()
        self._turn = self._build_turn(
            turn_index=0,
            feedback=circuit_initial_feedback(
                self._backbone.state.public_view.fault_count_range
            ),
        )
        return TaskResetResult(
            session_id=self._session_id,
            turn=self._turn,
            public_info={
                "task_id": self.instance.task_id,
                "kind": self.instance.kind,
                "domain": self.instance.domain,
                "renderer": self._renderer_type,
                "limits": self._public_limits(),
                "diagnosis_options": self._diagnosis_options(),
            },
            debug_info={
                "rollout_seed": seed,
                "faults": [
                    fault_payload(fault)
                    for fault in self._backbone.state.truth.hidden_faults
                ],
                "generation_debug": self.instance.privileged_payload.get(
                    "generation_debug", {}
                ),
            },
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current model-facing turn."""

        return self._turn

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Apply one model submission to the task session."""

        self._require_reset()
        if self._turn is None:
            return self._already_done_result()

        turn_index = self._turn.turn_index
        if submission.kind != "action":
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=f"unsupported submission kind: {submission.kind}",
                policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
                reason_category="unsupported_submission_kind",
            )

        try:
            action = self._backbone.parse_action(submission)
        except SubmissionParseError as error:
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
                policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
                reason_category=self._unparseable_action_category(submission),
            )

        if action.name in {
            SET_SOURCE_ACTION,
            MEASURE_VOLTAGE_ACTION,
            MEASURE_CURRENT_ACTION,
        }:
            return self._handle_probe_action(action, turn_index)
        if action.name == REPLACE_COMPONENT_ACTION:
            return self._handle_repair_action(action, turn_index)
        if action.name == FINAL_ANSWER_ACTION:
            return self._handle_final_action(action)
        return self._invalid_submission_result(
            turn_index=turn_index,
            reason=f"unknown action: {action.name}",
            policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
            reason_category="unknown_action",
        )

    def _handle_probe_action(
        self, action: ParsedAction, turn_index: int
    ) -> TaskStepResult:
        """Handle source-setting and measurement actions."""

        if not self._budget_state.probe_budget_available():
            self._budget_state.record_turn_submission()
            return self._budget_exceeded_result(
                attempted_action=action.name,
                reward_result=reward_budget_exceeded(
                    policy_category=BUDGET_EXCEEDED_POLICY.category,
                    reason_category="probe_budget_exhausted",
                    config=self._reward_config,
                ),
            )
        self._budget_state.record_turn_submission()
        try:
            public_info, debug_info, feedback = self._apply_probe_action(action)
        except (SubmissionParseError, CircuitSimulationError) as error:
            self._budget_state.record_invalid_after_counted_submission()
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
                policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
                reason_category="invalid_probe_action",
                counts_applied=True,
            )

        self._budget_state.record_accepted_probe()
        return self._continue_or_truncate(
            turn_index=turn_index,
            feedback=feedback,
            public_info=public_info,
            debug_info=debug_info,
            reward_result=reward_accepted_probe(action.name, self._reward_config),
            accepted=True,
        )

    def _apply_probe_action(
        self, action: ParsedAction
    ) -> tuple[dict[str, object], dict[str, object], str]:
        """Apply one accepted probe action to the backbone."""

        if action.name == SET_SOURCE_ACTION:
            source = self._backbone.set_source_from_action(action)
            return (
                {
                    "accepted_action": SET_SOURCE_ACTION,
                    "source": _source_public_payload(source),
                },
                {},
                (
                    f"Source set: {source.node_plus} relative to "
                    f"{source.node_minus} = {source.voltage_V:.6g} V."
                ),
            )
        if action.name == MEASURE_VOLTAGE_ACTION:
            measurement = self._backbone.measure_voltage_from_action(action)
            return (
                {
                    "accepted_action": MEASURE_VOLTAGE_ACTION,
                    "measurement": _voltage_public_payload(measurement),
                },
                {},
                (
                    f"Voltage V({measurement.node_a},{measurement.node_b}) = "
                    f"{measurement.voltage_V:.6g} V."
                ),
            )
        if action.name == MEASURE_CURRENT_ACTION:
            measurement = self._backbone.measure_current_from_action(action)
            return (
                {
                    "accepted_action": MEASURE_CURRENT_ACTION,
                    "measurement": _current_public_payload(measurement),
                },
                {},
                (
                    f"Current I({measurement.component_id}) = "
                    f"{measurement.current_A:.6g} A."
                ),
            )
        raise SubmissionParseError(f"unknown probe action: {action.name}")

    def _handle_repair_action(
        self, action: ParsedAction, turn_index: int
    ) -> TaskStepResult:
        """Handle a component replacement action."""

        if not self._budget_state.repair_budget_available():
            self._budget_state.record_turn_submission()
            return self._budget_exceeded_result(
                attempted_action=REPLACE_COMPONENT_ACTION,
                reward_result=reward_budget_exceeded(
                    policy_category=BUDGET_EXCEEDED_POLICY.category,
                    reason_category="repair_budget_exhausted",
                    config=self._reward_config,
                ),
            )
        self._budget_state.record_turn_submission()
        try:
            replacement = self._backbone.replace_component_from_action(action)
        except SubmissionParseError as error:
            self._budget_state.record_invalid_after_counted_submission()
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
                policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
                reason_category="invalid_repair_action",
                counts_applied=True,
            )
        self._budget_state.record_accepted_repair()
        feedback = (
            f"Component {replacement.component_id} replaced as {replacement.kind}."
        )
        return self._continue_or_truncate(
            turn_index=turn_index,
            feedback=feedback,
            public_info={
                "accepted_action": REPLACE_COMPONENT_ACTION,
                "repair": replacement_payload(replacement),
            },
            debug_info={},
            reward_result=reward_accepted_repair(
                replacement.component_id, self._reward_config
            ),
            accepted=True,
        )

    def _handle_final_action(self, action: ParsedAction) -> TaskStepResult:
        """Handle a structured final-answer action."""

        budget_available = self._budget_state.record_final_answer_submission()
        if not budget_available:
            return self._budget_exceeded_result(
                attempted_action=FINAL_ANSWER_ACTION,
                reward_result=reward_budget_exceeded(
                    policy_category=BUDGET_EXCEEDED_POLICY.category,
                    reason_category="final_answer_budget_exhausted",
                    config=self._reward_config,
                ),
            )
        try:
            submitted_faults, submitted_repairs = (
                self._backbone.final_answer_from_action(action)
            )
            self._validate_final_answer_labels(submitted_faults, submitted_repairs)
        except SubmissionParseError as error:
            self._budget_state.record_invalid_after_counted_submission()
            return self._invalid_submission_result(
                turn_index=self._turn.turn_index if self._turn is not None else 0,
                reason=str(error),
                policy=INVALID_FINAL_ANSWER_POLICY,
                reason_category="invalid_final_answer_arguments",
                counts_applied=True,
            )
        return self._final_result(submitted_faults, submitted_repairs)

    def _final_result(
        self,
        submitted_faults: tuple[str, ...],
        submitted_repairs: tuple[str, ...],
    ) -> TaskStepResult:
        """Evaluate repairs and terminate on a final answer."""

        evaluation = self._backbone.evaluate_final_answer(
            submitted_faults=submitted_faults,
            submitted_repairs=submitted_repairs,
        )
        reward = reward_final_verification(
            target_restored=evaluation.target_restored,
            diagnosis_correct=evaluation.diagnosis_correct,
            config=self._reward_config,
        )
        self._turn = None
        return TaskStepResult(
            accepted=True,
            reward_result=reward,
            terminal=True,
            truncated=False,
            observation=None,
            public_info=self._final_public_info(evaluation, reward),
            debug_info=self._final_debug_info(evaluation),
        )

    def _invalid_submission_result(
        self,
        turn_index: int,
        reason: str,
        policy: InvalidSubmissionPolicy,
        reason_category: str | None = None,
        counts_applied: bool = False,
    ) -> TaskStepResult:
        """Return a rejected-submission result."""

        category = reason_category if reason_category is not None else policy.category
        if not counts_applied:
            self._budget_state.record_invalid_submission(policy)
        if policy.terminal or policy.truncated:
            self._turn = None
            return TaskStepResult(
                accepted=False,
                reward_result=self._reward_invalid_submission(policy, category),
                terminal=policy.terminal,
                truncated=policy.truncated,
                observation=None,
                public_info=self._public_status(
                    {
                        "reason": reason,
                        "invalid_submission_category": category,
                        "invalid_submission_policy": policy.category,
                    }
                ),
                debug_info={},
            )
        return self._continue_or_truncate(
            turn_index=turn_index,
            feedback=f"Submission was not accepted: {reason}.",
            public_info={
                "reason": reason,
                "invalid_submission_category": category,
                "invalid_submission_policy": policy.category,
            },
            debug_info={},
            reward_result=self._reward_invalid_submission(policy, category),
            accepted=False,
        )

    def _reward_invalid_submission(
        self, policy: InvalidSubmissionPolicy, reason_category: str
    ) -> RewardResult:
        """Return the reward for one invalid-submission event."""

        if policy.category == BUDGET_EXCEEDED_POLICY.category:
            return reward_budget_exceeded(
                policy_category=policy.category,
                reason_category=reason_category,
                config=self._reward_config,
            )
        return reward_invalid_submission(
            policy_category=policy.category,
            reason_category=reason_category,
            config=self._reward_config,
        )

    def _continue_or_truncate(
        self,
        turn_index: int,
        feedback: str,
        public_info: dict[str, object],
        debug_info: dict[str, object],
        reward_result: RewardResult,
        accepted: bool,
    ) -> TaskStepResult:
        """Continue to the next turn unless the turn limit is exhausted."""

        if self._budget_state.turn_budget_exhausted():
            return self._truncated_result(
                reason="turn_budget_exhausted",
                public_info=public_info,
                debug_info=debug_info,
                accepted=accepted,
                reward_result=reward_result,
            )
        next_turn_index = turn_index + 1
        self._turn = self._build_turn(turn_index=next_turn_index, feedback=feedback)
        return TaskStepResult(
            accepted=accepted,
            reward_result=reward_result,
            terminal=False,
            truncated=False,
            observation=self._turn,
            public_info=self._public_status(public_info),
            debug_info=debug_info,
        )

    def _truncated_result(
        self,
        reason: str,
        public_info: dict[str, object],
        debug_info: dict[str, object],
        reward_result: RewardResult,
        accepted: bool,
    ) -> TaskStepResult:
        """Terminate the rollout due to a public limit."""

        self._turn = None
        result_info = dict(public_info)
        if "reason" in result_info:
            result_info["submission_reason"] = result_info["reason"]
        result_info["reason"] = reason
        return TaskStepResult(
            accepted=accepted,
            reward_result=reward_result,
            terminal=False,
            truncated=True,
            observation=None,
            public_info=self._public_status(result_info),
            debug_info=debug_info,
        )

    def _budget_exceeded_result(
        self, attempted_action: str, reward_result: RewardResult
    ) -> TaskStepResult:
        """Return a truncated budget-exceeded result."""

        self._budget_state.record_invalid_after_counted_submission()
        return self._truncated_result(
            reason="budget_exceeded",
            public_info={
                "attempted_action": attempted_action,
                "invalid_submission_category": BUDGET_EXCEEDED_POLICY.category,
                "invalid_submission_policy": BUDGET_EXCEEDED_POLICY.category,
            },
            debug_info={},
            reward_result=reward_result,
            accepted=False,
        )

    def _already_done_result(self) -> TaskStepResult:
        """Return a rejected result for submissions after completion."""

        return TaskStepResult(
            accepted=False,
            reward_result=reward_session_already_done(self._reward_config),
            terminal=True,
            truncated=False,
            observation=None,
            public_info={"reason": "session_already_done"},
            debug_info={},
        )

    def _build_turn(self, turn_index: int, feedback: str) -> TaskTurn:
        """Build the next model-facing turn."""

        render_context = CircuitRenderContext(
            public_view=self._backbone.state.public_view,
            diagnosis_options=self._diagnosis_options(),
            feedback=feedback,
            source_setting=self._backbone.source_setting,
            repairs=self._backbone.repairs,
            budget_status=self._budget_status_text(),
        )
        public_limits = self._public_limits()
        return TaskTurn(
            turn_index=turn_index,
            observation=render_circuit_observation(self._renderer_type, render_context),
            submission_modes=("action",),
            submission_format=self._submission_format(),
            action_schema=self._action_schema(),
            invalid_submission_policies=_invalid_policies(),
            public_limits=public_limits,
            public_info=self._public_status(
                {"diagnosis_options": self._diagnosis_options()}
            ),
        )

    def _submission_format(self) -> dict[str, object]:
        """Return the canonical public JSONL action submission format."""

        definition = self._backbone.state.public_view.definition
        example_component = definition.components[0]
        example_replacement = nominal_replacement_for_component(example_component)
        return {
            "type": JSON_LINE_FORMAT,
            "required_fields": (ACTION_NAME_FIELD, ACTION_ARGUMENTS_FIELD),
            "examples": (
                {
                    ACTION_NAME_FIELD: SET_SOURCE_ACTION,
                    ACTION_ARGUMENTS_FIELD: _source_example_arguments(definition),
                },
                {
                    ACTION_NAME_FIELD: MEASURE_VOLTAGE_ACTION,
                    ACTION_ARGUMENTS_FIELD: _voltage_example_arguments(definition),
                },
                {
                    ACTION_NAME_FIELD: MEASURE_CURRENT_ACTION,
                    ACTION_ARGUMENTS_FIELD: {
                        "component": example_component.component_id
                    },
                },
                {
                    ACTION_NAME_FIELD: REPLACE_COMPONENT_ACTION,
                    ACTION_ARGUMENTS_FIELD: _repair_example_arguments(
                        example_replacement
                    ),
                },
                {
                    ACTION_NAME_FIELD: FINAL_ANSWER_ACTION,
                    ACTION_ARGUMENTS_FIELD: {
                        "faults": [self._diagnosis_fault_ids()[0]],
                        "repairs": [self._diagnosis_repair_codes()[0]],
                    },
                },
            ),
        }

    def _action_schema(self) -> dict[str, object]:
        """Return the public structured action schema."""

        return {
            "actions": {
                SET_SOURCE_ACTION: {
                    "consumes_budget": {TURN_BUDGET: 1, PROBE_BUDGET: 1},
                    "arguments": {
                        "node_plus": {"type": "string"},
                        "node_minus": {"type": "string"},
                        "voltage_V": {
                            "type": "number",
                            "minimum": -24.0,
                            "maximum": 24.0,
                            "units": "V",
                        },
                    },
                },
                MEASURE_VOLTAGE_ACTION: {
                    "consumes_budget": {TURN_BUDGET: 1, PROBE_BUDGET: 1},
                    "arguments": {
                        "node_a": {"type": "string"},
                        "node_b": {"type": "string"},
                    },
                },
                MEASURE_CURRENT_ACTION: {
                    "consumes_budget": {TURN_BUDGET: 1, PROBE_BUDGET: 1},
                    "arguments": {"component": {"type": "string"}},
                },
                REPLACE_COMPONENT_ACTION: {
                    "consumes_budget": {TURN_BUDGET: 1, REPAIR_BUDGET: 1},
                    "arguments": {
                        "component": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "allowed": (
                                "resistor",
                                "capacitor",
                                "diode",
                                "switch",
                                "voltage_source",
                                "current_source",
                            ),
                        },
                        "kind_parameters": {
                            "resistor": {
                                "required": ("value_ohm",),
                                "fields": {
                                    "value_ohm": {
                                        "type": "number",
                                        "minimum_exclusive": 0,
                                        "units": "ohm",
                                        "must_match": "nominal component value",
                                    }
                                },
                            },
                            "capacitor": {
                                "required": ("value_F",),
                                "fields": {
                                    "value_F": {
                                        "type": "number",
                                        "minimum_exclusive": 0,
                                        "units": "F",
                                        "must_match": "nominal component value",
                                    }
                                },
                            },
                            "diode": {
                                "required": ("forward_drop_V",),
                                "fields": {
                                    "forward_drop_V": {
                                        "type": "number",
                                        "minimum_exclusive": 0,
                                        "units": "V",
                                        "must_match": "nominal component value",
                                    }
                                },
                            },
                            "switch": {
                                "required": ("closed",),
                                "fields": {
                                    "closed": {
                                        "type": "boolean",
                                        "must_match": "nominal component value",
                                    }
                                },
                            },
                            "voltage_source": {
                                "required": ("voltage_V",),
                                "fields": {
                                    "voltage_V": {
                                        "type": "number",
                                        "units": "V",
                                        "must_match": "nominal component value",
                                    }
                                },
                            },
                            "current_source": {
                                "required": ("current_A",),
                                "fields": {
                                    "current_A": {
                                        "type": "number",
                                        "units": "A",
                                        "must_match": "nominal component value",
                                    }
                                },
                            },
                        },
                    },
                },
                FINAL_ANSWER_ACTION: {
                    "consumes_budget": {
                        TURN_BUDGET: 1,
                        FINAL_ANSWER_BUDGET: 1,
                    },
                    "arguments": {
                        "faults": {
                            "type": "array",
                            "items": "string",
                            "allowed": self._diagnosis_fault_ids(),
                        },
                        "repairs": {
                            "type": "array",
                            "items": "string",
                            "allowed": self._diagnosis_repair_codes(),
                        },
                    },
                },
            }
        }

    def _public_limits(self) -> dict[str, object]:
        """Return circuit-specific public limits."""

        return dict(self.instance.public_limits())

    def _public_status(self, extra_info: dict[str, object]) -> dict[str, object]:
        """Return public rollout counters with event metadata."""

        return self._budget_state.public_status(extra_info=extra_info)

    def _diagnosis_options(self) -> Mapping[str, object]:
        """Return public final-answer diagnosis options from the instance."""

        options = self.instance.public_payload["diagnosis_options"]
        if isinstance(options, Mapping):
            return options
        raise TypeError("diagnosis_options must be a mapping")

    def _diagnosis_fault_ids(self) -> tuple[str, ...]:
        """Return allowed public final-answer fault IDs."""

        options = self._diagnosis_options()
        return _string_tuple_field(options, "fault_ids")

    def _diagnosis_repair_codes(self) -> tuple[str, ...]:
        """Return allowed public final-answer repair codes."""

        options = self._diagnosis_options()
        return _string_tuple_field(options, "repair_codes")

    def _validate_final_answer_labels(
        self, submitted_faults: tuple[str, ...], submitted_repairs: tuple[str, ...]
    ) -> None:
        """Reject malformed final-answer labels before verifier evaluation."""

        min_fault_count, max_fault_count = (
            self._backbone.state.public_view.fault_count_range
        )
        _reject_duplicate_strings(submitted_faults, "faults")
        _reject_duplicate_strings(submitted_repairs, "repairs")
        if len(submitted_faults) < min_fault_count:
            raise SubmissionParseError(
                f"faults must contain at least {min_fault_count} label(s)"
            )
        if len(submitted_faults) > max_fault_count:
            raise SubmissionParseError(
                f"faults must contain at most {max_fault_count} label(s)"
            )
        if len(submitted_repairs) != len(submitted_faults):
            raise SubmissionParseError(
                "repairs must contain one repair code per submitted fault"
            )
        _reject_unknown_strings(submitted_faults, self._diagnosis_fault_ids(), "faults")
        _reject_unknown_strings(
            submitted_repairs, self._diagnosis_repair_codes(), "repairs"
        )

    def _budget_status_text(self) -> str:
        """Return public budget status lines for renderers."""

        usage = self._budget_state.budget_usage()
        remaining = self._budget_state.budget_remaining()
        return "\n".join(
            [
                f"- turns: {usage[TURN_BUDGET]} used, {remaining[TURN_BUDGET]} remaining",
                (
                    f"- probe actions: {usage[PROBE_BUDGET]} used, "
                    f"{remaining[PROBE_BUDGET]} remaining"
                ),
                (
                    f"- repair actions: {usage[REPAIR_BUDGET]} used, "
                    f"{remaining[REPAIR_BUDGET]} remaining"
                ),
                (
                    f"- final answers: {usage[FINAL_ANSWER_BUDGET]} used, "
                    f"{remaining[FINAL_ANSWER_BUDGET]} remaining"
                ),
            ]
        )

    def _unparseable_action_category(self, submission: TaskSubmission) -> str:
        """Return the public reason category for an unparseable action."""

        raw = submission.raw.lstrip()
        if raw.startswith("{"):
            return "malformed_transport"
        return "unparseable_action"

    def _final_public_info(
        self, evaluation: FinalCircuitEvaluation, reward: RewardResult
    ) -> dict[str, object]:
        """Return public final-answer metadata."""

        return {
            "target_restored": evaluation.target_restored,
            "diagnosis_correct": evaluation.diagnosis_correct,
            "score": reward.score,
            "target_checks": target_check_public_payloads(evaluation.check_results),
            **self._public_status({}),
        }

    def _final_debug_info(
        self, evaluation: FinalCircuitEvaluation
    ) -> dict[str, object]:
        """Return privileged final-answer metadata."""

        return {
            "submitted_faults": evaluation.submitted_faults,
            "submitted_repairs": evaluation.submitted_repairs,
            "expected_faults": evaluation.expected_faults,
            "expected_repairs": evaluation.expected_repairs,
            "target_checks": target_check_debug_payloads(evaluation.check_results),
            "simulation_error": evaluation.simulation_error,
            "repairs": [
                replacement_payload(repair)
                for repair in self._backbone.repairs.values()
            ],
            "faults": [
                fault_payload(fault)
                for fault in self._backbone.state.truth.hidden_faults
            ],
        }

    def _require_reset(self) -> None:
        """Raise a usage error when the session has not been reset."""

        if self._session_id is None:
            raise RuntimeError("session has not been reset")


def _source_public_payload(source: SourceSetting) -> dict[str, object]:
    """Return public source metadata."""

    return {
        "node_plus": source.node_plus,
        "node_minus": source.node_minus,
        "voltage_V": source.voltage_V,
    }


def _voltage_public_payload(measurement: VoltageMeasurement) -> dict[str, object]:
    """Return public voltage measurement metadata."""

    return {
        "node_a": measurement.node_a,
        "node_b": measurement.node_b,
        "voltage_V": measurement.voltage_V,
    }


def _current_public_payload(measurement: CurrentMeasurement) -> dict[str, object]:
    """Return public current measurement metadata."""

    return {
        "component": measurement.component_id,
        "current_A": measurement.current_A,
    }


def _source_example_arguments(definition: CircuitDefinition) -> dict[str, object]:
    """Return a valid source action example for a public circuit."""

    if definition.target_source is not None:
        source = definition.target_source
        return {
            "node_plus": source.node_plus,
            "node_minus": source.node_minus,
            "voltage_V": source.voltage_V,
        }
    non_ground = next(
        node for node in definition.nodes if node != definition.ground_node
    )
    return {
        "node_plus": non_ground,
        "node_minus": definition.ground_node,
        "voltage_V": 5.0,
    }


def _voltage_example_arguments(definition: CircuitDefinition) -> dict[str, object]:
    """Return a valid voltage measurement example for a public circuit."""

    if definition.target_source is not None:
        return {
            "node_a": definition.target_source.node_plus,
            "node_b": definition.target_source.node_minus,
        }
    non_ground = next(
        node for node in definition.nodes if node != definition.ground_node
    )
    return {"node_a": non_ground, "node_b": definition.ground_node}


def _repair_example_arguments(replacement: ReplacementSpec) -> dict[str, object]:
    """Return a valid repair action example from a nominal replacement."""

    return {
        "component": replacement.component_id,
        "kind": replacement.kind,
        **dict(replacement.parameters),
    }


def _reject_duplicate_strings(values: tuple[str, ...], field_name: str) -> None:
    """Reject duplicate strings in a final-answer field."""

    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise SubmissionParseError(f"{field_name} must not contain duplicates")
        seen.add(value)


def _reject_unknown_strings(
    values: tuple[str, ...], allowed_values: tuple[str, ...], field_name: str
) -> None:
    """Reject strings outside a final-answer allowed vocabulary."""

    allowed = set(allowed_values)
    unknown_values = tuple(value for value in values if value not in allowed)
    if len(unknown_values) > 0:
        joined_values = ", ".join(unknown_values)
        raise SubmissionParseError(
            f"{field_name} contain unsupported label(s): {joined_values}"
        )


def _string_tuple_field(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    """Return a tuple of strings from a public payload field."""

    raw_values = values[name]
    if not isinstance(raw_values, tuple):
        raise TypeError(f"{name} must be a tuple")
    result: list[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            raise TypeError(f"{name} values must be strings")
        result.append(value)
    return tuple(result)
