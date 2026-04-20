"""Scalar session for physics discovery."""

from rlvr_physics.core.instances import TaskInstance, require_int, require_str
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.trajectory import TaskTrajectory, TrajectoryEvent
from rlvr_physics.tasks.physics.discovery.actions import (
    make_action_schema,
    parse_discovery_action,
    validate_experiment_inputs,
)
from rlvr_physics.tasks.physics.discovery.evaluation import evaluate_physics_hypothesis
from rlvr_physics.tasks.physics.discovery.expressions import (
    evaluate_expression,
    extract_hypothesis_expression,
)
from rlvr_physics.tasks.physics.discovery.renderers import render_physics_discovery_text
from rlvr_physics.tasks.physics.discovery.types import (
    ExperimentObservation,
    HypothesisAttempt,
    ParsedDiscoveryAction,
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
            action = parse_discovery_action(submission)
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
            inputs = validate_experiment_inputs(self._instance, action.inputs)
            output = evaluate_expression(
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
            expression=extract_hypothesis_expression(action.equation),
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
            action_schema=make_action_schema(self._instance),
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


def _hypothesis_reason(correct: bool, truncated: bool) -> str:
    if correct:
        return "correct_hypothesis"
    if truncated:
        return "hypothesis_budget_exhausted"
    return "hypothesis_tested"
