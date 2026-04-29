"""Scalar session implementation for the cart inference task."""

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.submissions import (
    ACTION_ARGUMENTS_FIELD,
    ACTION_NAME_FIELD,
    InvalidSubmissionPolicy,
    JSON_LINE_FORMAT,
    ParsedAction,
    TaskSubmission,
)
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FINAL_ANSWER_ACTION,
    MEASURE_POSITION_ACTION,
    ActionBudgetExceeded,
    CartInferenceBackbone,
    FinalAnswerEvaluation,
    SubmissionParseError,
)
from rlvr_physics.tasks.physics.cart_inference.budgets import (
    ACTION_BUDGET,
    FINAL_ANSWER_BUDGET,
    TURN_BUDGET,
    CartRolloutBudgetState,
)
from rlvr_physics.tasks.physics.cart_inference.rewards import (
    CartRewardConfig,
    reward_accepted_measurement,
    reward_budget_exceeded,
    reward_final_answer,
    reward_invalid_submission,
    reward_session_already_done,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CartMeasurementView,
    CartRenderContext,
    render_cart_observation,
    validate_cart_renderer_type,
)
from rlvr_physics.tasks.physics.cart_inference.prompting import cart_initial_feedback


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

    policies = [
        RETRYABLE_INVALID_SUBMISSION_POLICY,
        INVALID_FINAL_ANSWER_POLICY,
        BUDGET_EXCEEDED_POLICY,
    ]
    return {policy.category: policy for policy in policies}


class CartInferenceSession:
    """Scalar runtime session for one cart inference instance.

    Parameters
    ----------
    instance:
        Immutable cart inference task instance.

    Attributes
    ----------
    instance:
        Immutable cart inference task instance.
    """

    def __init__(
        self,
        instance: TaskInstance,
        renderer_type: str,
        reward_config: CartRewardConfig,
    ) -> None:
        """Initialize a mutable scalar runtime session.

        Parameters
        ----------
        instance:
            Immutable cart inference task instance.
        renderer_type:
            Renderer identifier used for every observation in this session.
        reward_config:
            Reward policy configuration used for every step result in this
            session.
        """

        validate_cart_renderer_type(renderer_type)
        self.instance = instance
        self._renderer_type = renderer_type
        self._backbone = CartInferenceBackbone(instance)
        self._budget_state = CartRolloutBudgetState(instance.budget_limits)
        self._reward_config = reward_config
        self._session_id: str | None = None
        self._turn: TaskTurn | None = None

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh rollout and return the first model-facing turn.

        Parameters
        ----------
        seed:
            Deterministic rollout seed.

        Returns
        -------
        TaskResetResult
            Session identifier, first turn, and reset metadata.
        """

        self._session_id = new_session_id(self.instance.task_id, seed)
        self._budget_state.reset()
        self._backbone.reset_rollout()
        self._turn = self._build_turn(
            turn_index=0,
            feedback=cart_initial_feedback(),
            current_measurement=None,
        )
        return TaskResetResult(
            session_id=self._session_id,
            turn=self._turn,
            public_info={
                "task_id": self.instance.task_id,
                "kind": self.instance.kind,
                "domain": self.instance.domain,
                "instance_hash": self.instance.content_hash(),
                "renderer": self._renderer_type,
                "limits": self._public_limits(),
            },
            debug_info={
                "rollout_seed": seed,
                "acceleration_mps2": self._backbone.state.acceleration_mps2,
                "exact_target_position_m": (
                    self._backbone.state.exact_target_position_m
                ),
                "measurement_noise_seed": self._backbone.state.measurement_noise_seed,
            },
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current model-facing turn.

        Returns
        -------
        TaskTurn or None
            Current turn, or ``None`` after terminal or truncated completion.
        """

        return self._turn

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Apply one model submission to the task session.

        Parameters
        ----------
        submission:
            Raw and optionally interpreted model output.

        Returns
        -------
        TaskStepResult
            Step outcome, reward, next observation, and metadata.
        """

        self._require_reset()
        if self._turn is None:
            return self._already_done_result()

        turn_index = self._turn.turn_index

        if submission.kind == "action":
            return self._handle_action(submission, turn_index)
        return self._invalid_submission_result(
            turn_index=turn_index,
            reason=f"unsupported submission kind: {submission.kind}",
            policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
            reason_category="unsupported_submission_kind",
        )

    def _handle_action(
        self,
        submission: TaskSubmission,
        turn_index: int,
    ) -> TaskStepResult:
        """Handle an action-mode submission."""

        try:
            action = self._backbone.parse_action(submission)
        except SubmissionParseError as error:
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
                policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
                reason_category=self._unparseable_action_category(submission),
            )

        if action.name == MEASURE_POSITION_ACTION:
            return self._handle_measurement_action(
                action=action,
                turn_index=turn_index,
            )
        if action.name == FINAL_ANSWER_ACTION:
            return self._handle_final_action(
                action=action,
                turn_index=turn_index,
            )
        return self._invalid_submission_result(
            turn_index=turn_index,
            reason=f"unknown action: {action.name}",
            policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
            reason_category="unknown_action",
        )

    def _handle_measurement_action(
        self,
        action: ParsedAction,
        turn_index: int,
    ) -> TaskStepResult:
        """Handle a public position measurement action."""

        self._budget_state.record_action_submission()
        try:
            measurement = self._backbone.measure(action)
        except ActionBudgetExceeded as error:
            self._budget_state.record_invalid_after_counted_submission()
            return self._truncated_result(
                reason=str(error),
                public_info={
                    "attempted_action": MEASURE_POSITION_ACTION,
                    "invalid_submission_category": BUDGET_EXCEEDED_POLICY.category,
                    "invalid_submission_policy": BUDGET_EXCEEDED_POLICY.category,
                },
                debug_info={},
                reward_result=reward_budget_exceeded(
                    policy_category=BUDGET_EXCEEDED_POLICY.category,
                    reason_category=BUDGET_EXCEEDED_POLICY.category,
                    config=self._reward_config,
                ),
                accepted=False,
            )
        except (SubmissionParseError, ValueError) as error:
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
                policy=RETRYABLE_INVALID_SUBMISSION_POLICY,
                reason_category="invalid_action_arguments",
                counts_applied=True,
            )

        current_measurement = CartMeasurementView(
            time_s=measurement.time_s,
            measured_position_m=measurement.measured_position_m,
        )
        feedback = (
            f"Measurement at t={measurement.time_s:g}s: "
            f"x={measurement.measured_position_m:.6g} m."
        )
        return self._continue_or_truncate(
            turn_index=turn_index,
            feedback=feedback,
            current_measurement=current_measurement,
            public_info={
                "accepted_action": MEASURE_POSITION_ACTION,
                "measurement": {
                    "time_s": measurement.time_s,
                    "measured_position_m": measurement.measured_position_m,
                },
                "actions_remaining": self._actions_remaining(),
            },
            debug_info={
                "true_position_m": measurement.true_position_m,
                "noise_m": measurement.noise_m,
            },
            reward_result=reward_accepted_measurement(
                measurement=measurement,
                config=self._reward_config,
            ),
        )

    def _handle_final_action(
        self,
        action: ParsedAction,
        turn_index: int,
    ) -> TaskStepResult:
        """Handle a structured final-answer action."""

        self._budget_state.record_final_answer_submission()
        try:
            submitted_position_m = self._backbone.final_answer_from_action(action)
        except SubmissionParseError as error:
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
                policy=INVALID_FINAL_ANSWER_POLICY,
                counts_applied=True,
            )
        return self._final_result(
            submitted_position_m=submitted_position_m,
        )

    def _final_result(
        self,
        submitted_position_m: float,
    ) -> TaskStepResult:
        """Evaluate and terminate on a final answer."""

        evaluation = self._backbone.evaluate_final_answer(submitted_position_m)
        reward = reward_final_answer(evaluation, self._reward_config)
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
        else:
            self._budget_state.record_invalid_after_counted_submission()
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
            current_measurement=None,
            public_info={
                "reason": reason,
                "invalid_submission_category": category,
                "invalid_submission_policy": policy.category,
            },
            debug_info={},
            reward_result=self._reward_invalid_submission(policy, category),
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
        current_measurement: CartMeasurementView | None,
        public_info: dict[str, object],
        debug_info: dict[str, object],
        reward_result: RewardResult,
    ) -> TaskStepResult:
        """Continue to the next turn unless the turn limit is exhausted."""

        if self._budget_state.turn_budget_exhausted():
            return self._truncated_result(
                reason="turn_budget_exhausted",
                public_info=public_info,
                debug_info=debug_info,
                accepted="reason" not in public_info,
                reward_result=reward_result,
            )

        next_turn_index = turn_index + 1
        self._turn = self._build_turn(
            turn_index=next_turn_index,
            feedback=feedback,
            current_measurement=current_measurement,
        )
        public_result = self._public_status(public_info)
        return TaskStepResult(
            accepted="reason" not in public_info,
            reward_result=reward_result,
            terminal=False,
            truncated=False,
            observation=self._turn,
            public_info=public_result,
            debug_info=debug_info,
        )

    def _truncated_result(
        self,
        reason: str,
        public_info: dict[str, object],
        debug_info: dict[str, object],
        reward_result: RewardResult,
        accepted: bool = False,
    ) -> TaskStepResult:
        """Terminate the rollout due to a public limit."""

        self._turn = None
        result_info = self._public_status({"reason": reason, **public_info})
        return TaskStepResult(
            accepted=accepted,
            reward_result=reward_result,
            terminal=False,
            truncated=True,
            observation=None,
            public_info=result_info,
            debug_info=debug_info,
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

    def _build_turn(
        self,
        turn_index: int,
        feedback: str,
        current_measurement: CartMeasurementView | None,
    ) -> TaskTurn:
        """Build the next model-facing turn."""

        state = self._backbone.state
        public_limits = self._public_limits()
        invalid_submission_policies = _invalid_policies()
        render_context = CartRenderContext(
            initial_position_m=state.initial_position_m,
            initial_velocity_mps=state.initial_velocity_mps,
            target_time_s=state.target_time_s,
            min_measurement_time_s=state.min_measurement_time_s,
            max_measurement_time_s=state.max_measurement_time_s,
            measurement_noise_abs_m=state.measurement_noise_abs_m,
            feedback=feedback,
            current_measurement=current_measurement,
            actions_used=self._backbone.measurements_used,
            action_budget=self._backbone.action_budget,
            actions_remaining=self._actions_remaining(),
            final_answers_used=self._budget_state.final_answers_used,
            final_answer_budget=self._budget_state.final_answer_budget,
            final_answers_remaining=self._budget_state.final_answers_remaining,
        )
        return TaskTurn(
            turn_index=turn_index,
            observation=render_cart_observation(self._renderer_type, render_context),
            submission_modes=("action",),
            submission_format=self._submission_format(),
            action_schema={
                "actions": {
                    MEASURE_POSITION_ACTION: {
                        "consumes_budget": {
                            TURN_BUDGET: 1,
                            ACTION_BUDGET: 1,
                        },
                        "arguments": {
                            "time": {
                                "type": "number",
                                "minimum": state.min_measurement_time_s,
                                "maximum": state.max_measurement_time_s,
                                "units": "s",
                            }
                        },
                    },
                    FINAL_ANSWER_ACTION: {
                        "consumes_budget": {
                            TURN_BUDGET: 1,
                            FINAL_ANSWER_BUDGET: 1,
                        },
                        "arguments": {
                            "x": {
                                "type": "number",
                                "units": "m",
                            }
                        },
                    },
                }
            },
            invalid_submission_policies=invalid_submission_policies,
            public_limits=public_limits,
            public_info=self._public_status({}),
        )

    def _submission_format(self) -> dict[str, object]:
        """Return the canonical public JSONL action submission format."""

        state = self._backbone.state
        return {
            "type": JSON_LINE_FORMAT,
            "required_fields": (ACTION_NAME_FIELD, ACTION_ARGUMENTS_FIELD),
            "examples": (
                {
                    ACTION_NAME_FIELD: MEASURE_POSITION_ACTION,
                    ACTION_ARGUMENTS_FIELD: {"time": state.max_measurement_time_s},
                },
                {
                    ACTION_NAME_FIELD: FINAL_ANSWER_ACTION,
                    ACTION_ARGUMENTS_FIELD: {"x": 0.0},
                },
            ),
        }

    def _public_limits(self) -> dict[str, object]:
        """Return cart-specific public limits with semantic budget names."""

        return dict(self.instance.public_limits())

    def _actions_remaining(self) -> int:
        """Return the remaining non-final task action budget."""

        return self._backbone.measurements_remaining

    def _unparseable_action_category(self, submission: TaskSubmission) -> str:
        """Return the public reason category for an unparseable action."""

        raw = submission.raw.lstrip()
        if raw.startswith("{"):
            return "malformed_transport"
        return "unparseable_action"

    def _public_status(self, extra_info: dict[str, object]) -> dict[str, object]:
        """Return public rollout counters with additional event metadata."""

        return self._budget_state.public_status(
            actions_used=self._backbone.measurements_used,
            actions_remaining=self._actions_remaining(),
            extra_info=extra_info,
        )

    def _require_reset(self) -> None:
        """Raise a usage error when the session has not been reset."""

        if self._session_id is None:
            raise RuntimeError("session has not been reset")

    def _final_public_info(
        self, evaluation: FinalAnswerEvaluation, reward: RewardResult
    ) -> dict[str, object]:
        """Return public final-answer metadata."""

        return {
            "correct": evaluation.correct,
            "absolute_error_m": evaluation.absolute_error_m,
            "score": reward.score,
            **self._public_status({}),
        }

    def _final_debug_info(self, evaluation: FinalAnswerEvaluation) -> dict[str, object]:
        """Return privileged final-answer metadata."""

        return {
            "submitted_position_m": evaluation.submitted_position_m,
            "exact_position_m": evaluation.exact_position_m,
            "tolerance_abs_m": evaluation.tolerance_abs_m,
            "acceleration_mps2": self._backbone.state.acceleration_mps2,
        }
