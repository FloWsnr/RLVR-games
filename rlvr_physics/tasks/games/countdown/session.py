"""Scalar session for Countdown."""

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.trajectory import TaskTrajectory
from rlvr_physics.tasks.games.countdown.renderers import (
    render_countdown_image,
    render_countdown_text,
)
from rlvr_physics.tasks.games.countdown.verifier import verify_countdown_submission


class CountdownSession:
    """Single-step Countdown task session."""

    def __init__(self, instance: TaskInstance, renderer: str) -> None:
        self._instance = instance
        self._renderer = renderer
        self._turn: TaskTurn | None = None
        self._trajectory = TaskTrajectory(task_id=instance.task_id, session_id="")
        self._terminal = False

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh Countdown rollout."""

        session_id = new_session_id(self._instance.task_id, seed)
        self._trajectory = TaskTrajectory(
            task_id=self._instance.task_id, session_id=session_id
        )
        self._terminal = False
        self._turn = self._make_turn(0)
        self._trajectory.append(
            "reset",
            0,
            {"task_id": self._instance.task_id, "renderer": self._renderer},
            {"instance_hash": self._instance.content_hash()},
        )
        self._trajectory.append(
            "observation",
            0,
            {
                "renderer": self._renderer,
                "content_digests": self._turn.observation.content_digests(),
            },
            {},
        )
        return TaskResetResult(
            session_id=session_id, turn=self._turn, trajectory=self._trajectory
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current turn, or ``None`` after completion."""

        return self._turn

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the verified session trajectory."""

        return self._trajectory

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Verify a single Countdown submission."""

        if self._turn is None or self._terminal:
            event = self._trajectory.append(
                "invalid_submission",
                0,
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
        verification = verify_countdown_submission(self._instance, submission)
        submit_event = self._trajectory.append(
            "submission",
            turn_index,
            {"kind": submission.kind, "raw": submission.raw},
            {"parsed_expression": verification.expression},
        )
        reward_event = self._trajectory.append(
            "reward",
            turn_index,
            {
                "reward": verification.reward,
                "reason": verification.reason,
                "correct": verification.correct,
            },
            {
                "value": str(verification.value)
                if verification.value is not None
                else None,
                "used_numbers": verification.used_numbers,
                "reference_expression": self._instance.privileged_payload[
                    "reference_expression"
                ],
            },
        )
        self._terminal = True
        self._turn = None
        return TaskStepResult(
            accepted=verification.accepted,
            reward=verification.reward,
            score=1.0 if verification.correct else 0.0,
            terminal=True,
            truncated=False,
            observation=None,
            public_info={
                "reason": verification.reason,
                "correct": verification.correct,
            },
            debug_info={
                "expression": verification.expression,
                "value": str(verification.value)
                if verification.value is not None
                else None,
                "used_numbers": verification.used_numbers,
            },
            events=(submit_event, reward_event),
        )

    def _make_turn(self, turn_index: int) -> TaskTurn:
        if self._renderer == "text":
            observation = render_countdown_text(self._instance)
        elif self._renderer == "image":
            observation = render_countdown_image(self._instance)
        else:
            raise ValueError(f"unknown Countdown renderer: {self._renderer}")
        return TaskTurn(
            turn_index=turn_index,
            observation=observation,
            submission_modes=("final_text",),
            action_schema={},
            public_limits=self._instance.limits.as_public_dict(),
            public_info={
                "task_id": self._instance.task_id,
                "kind": self._instance.kind,
                "numbers": self._instance.public_payload["numbers"],
                "target": self._instance.public_payload["target"],
            },
        )
