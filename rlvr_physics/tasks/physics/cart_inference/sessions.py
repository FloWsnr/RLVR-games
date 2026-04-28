"""Scalar session implementation for the cart inference task."""

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FINAL_ANSWER_ACTION,
    MEASURE_POSITION_ACTION,
    ActionBudgetExceeded,
    CartInferenceBackbone,
    FinalAnswerEvaluation,
    ParsedAction,
    SubmissionParseError,
)
from rlvr_physics.tasks.physics.cart_inference.rewards import (
    reward_final_answer,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CartMeasurementView,
    CartRenderContext,
    render_cart_observation,
    validate_cart_renderer_type,
)
from rlvr_physics.tasks.physics.cart_inference.prompting import cart_initial_feedback


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

    def __init__(self, instance: TaskInstance, renderer_type: str) -> None:
        """Initialize a mutable scalar runtime session.

        Parameters
        ----------
        instance:
            Immutable cart inference task instance.
        renderer_type:
            Renderer identifier used for every observation in this session.
        """

        validate_cart_renderer_type(renderer_type)
        self.instance = instance
        self._renderer_type = renderer_type
        self._backbone = CartInferenceBackbone(instance)
        self._session_id: str | None = None
        self._turn: TaskTurn | None = None
        self._submissions_used: int = 0

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
        self._submissions_used = 0
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
                "rollout_seed": seed,
                "renderer": self._renderer_type,
                "limits": self.instance.public_limits(),
            },
            debug_info={
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
        self._submissions_used += 1

        if submission.kind == "action":
            return self._handle_action(submission, turn_index)
        return self._invalid_submission_result(
            turn_index=turn_index,
            reason=f"unsupported submission kind: {submission.kind}",
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
        )

    def _handle_measurement_action(
        self,
        action: ParsedAction,
        turn_index: int,
    ) -> TaskStepResult:
        """Handle a public position measurement action."""

        try:
            measurement = self._backbone.measure(action)
        except ActionBudgetExceeded as error:
            return self._truncated_result(
                reason=str(error),
                public_info={"accepted_action": MEASURE_POSITION_ACTION},
                debug_info={},
            )
        except (SubmissionParseError, ValueError) as error:
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
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
                "measurements_remaining": self._measurements_remaining(),
            },
            debug_info={
                "true_position_m": measurement.true_position_m,
                "noise_m": measurement.noise_m,
            },
        )

    def _handle_final_action(
        self,
        action: ParsedAction,
        turn_index: int,
    ) -> TaskStepResult:
        """Handle a structured final-answer action."""

        try:
            submitted_position_m = self._backbone.final_answer_from_action(action)
        except SubmissionParseError as error:
            return self._invalid_submission_result(
                turn_index=turn_index,
                reason=str(error),
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
        reward = reward_final_answer(evaluation)
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
    ) -> TaskStepResult:
        """Return a rejected-submission result."""

        return self._continue_or_truncate(
            turn_index=turn_index,
            feedback=f"Submission was not accepted: {reason}.",
            current_measurement=None,
            public_info={"reason": reason},
            debug_info={},
        )

    def _continue_or_truncate(
        self,
        turn_index: int,
        feedback: str,
        current_measurement: CartMeasurementView | None,
        public_info: dict[str, object],
        debug_info: dict[str, object],
    ) -> TaskStepResult:
        """Continue to the next turn unless the turn limit is exhausted."""

        if self._submissions_used >= self.instance.max_turns:
            return self._truncated_result(
                reason="max_turns_exhausted",
                public_info=public_info,
                debug_info=debug_info,
            )

        next_turn_index = turn_index + 1
        self._turn = self._build_turn(
            turn_index=next_turn_index,
            feedback=feedback,
            current_measurement=current_measurement,
        )
        public_result: dict[str, object] = {
            "measurements_used": self._backbone.measurements_used,
            "measurements_remaining": self._measurements_remaining(),
            "submissions_used": self._submissions_used,
        }
        public_result.update(public_info)
        return TaskStepResult(
            accepted="reason" not in public_info,
            reward_result=RewardResult(reward=0.0, score=None),
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
    ) -> TaskStepResult:
        """Terminate the rollout due to a public limit."""

        self._turn = None
        result_info: dict[str, object] = {
            "reason": reason,
            "measurements_used": self._backbone.measurements_used,
            "measurements_remaining": self._measurements_remaining(),
            "submissions_used": self._submissions_used,
        }
        result_info.update(public_info)
        return TaskStepResult(
            accepted=False,
            reward_result=RewardResult(reward=0.0, score=None),
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
            reward_result=RewardResult(reward=0.0, score=None),
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
        render_context = CartRenderContext(
            initial_position_m=state.initial_position_m,
            initial_velocity_mps=state.initial_velocity_mps,
            target_time_s=state.target_time_s,
            min_measurement_time_s=state.min_measurement_time_s,
            max_measurement_time_s=state.max_measurement_time_s,
            measurement_noise_abs_m=state.measurement_noise_abs_m,
            feedback=feedback,
            current_measurement=current_measurement,
            measurements_used=self._backbone.measurements_used,
            action_budget=self._backbone.action_budget,
            measurements_remaining=self._measurements_remaining(),
        )
        return TaskTurn(
            turn_index=turn_index,
            observation=render_cart_observation(self._renderer_type, render_context),
            submission_modes=("action",),
            action_schema={
                "actions": {
                    MEASURE_POSITION_ACTION: {
                        "arguments": {
                            "time": {
                                "type": "number",
                                "minimum": state.min_measurement_time_s,
                                "maximum": state.max_measurement_time_s,
                                "units": "s",
                            }
                        }
                    },
                    FINAL_ANSWER_ACTION: {
                        "arguments": {
                            "x": {
                                "type": "number",
                                "units": "m",
                            }
                        }
                    },
                }
            },
            public_limits=self.instance.public_limits(),
            public_info={
                "measurements_used": self._backbone.measurements_used,
                "measurements_remaining": self._measurements_remaining(),
            },
        )

    def _measurements_remaining(self) -> int:
        """Return the remaining measurement action budget."""

        return self._backbone.measurements_remaining

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
            "measurements_used": self._backbone.measurements_used,
            "submissions_used": self._submissions_used,
        }

    def _final_debug_info(self, evaluation: FinalAnswerEvaluation) -> dict[str, object]:
        """Return privileged final-answer metadata."""

        return {
            "submitted_position_m": evaluation.submitted_position_m,
            "exact_position_m": evaluation.exact_position_m,
            "tolerance_abs_m": evaluation.tolerance_abs_m,
            "acceleration_mps2": self._backbone.state.acceleration_mps2,
        }
