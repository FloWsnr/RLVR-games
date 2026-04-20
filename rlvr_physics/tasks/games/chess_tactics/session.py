"""Scalar session for chess tactics."""

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.trajectory import TaskTrajectory
from rlvr_physics.tasks.games.chess_tactics.renderers import (
    render_chess_tactics_image,
    render_chess_tactics_text,
)
from rlvr_physics.tasks.games.chess_tactics.verifier import (
    verify_chess_tactic_submission,
)


class ChessTacticsSession:
    """Single-step chess tactic task session."""

    def __init__(self, instance: TaskInstance, renderer: str) -> None:
        self._instance = instance
        self._renderer = renderer
        self._turn: TaskTurn | None = None
        self._terminal = False
        self._trajectory = TaskTrajectory(task_id=instance.task_id, session_id="")

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh chess tactic rollout."""

        session_id = new_session_id(self._instance.task_id, seed)
        self._terminal = False
        self._trajectory = TaskTrajectory(
            task_id=self._instance.task_id, session_id=session_id
        )
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
            {"fen": self._instance.public_payload["fen"]},
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
        """Verify one chess tactic move."""

        if self._turn is None or self._terminal:
            event = self._trajectory.append(
                "invalid_submission", 0, {"reason": "session_finished"}, {}
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
        verification = verify_chess_tactic_submission(self._instance, submission)
        submit_event = self._trajectory.append(
            "submission",
            turn_index,
            {"kind": submission.kind, "raw": submission.raw},
            {"move_uci": verification.move_uci, "move_san": verification.move_san},
        )
        reward_event = self._trajectory.append(
            "reward",
            turn_index,
            {
                "reward": verification.reward,
                "reason": verification.reason,
                "correct": verification.correct,
            },
            {"solutions": self._instance.privileged_payload["solution_moves_uci"]},
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
                "move_uci": verification.move_uci,
                "move_san": verification.move_san,
            },
            events=(submit_event, reward_event),
        )

    def _make_turn(self, turn_index: int) -> TaskTurn:
        if self._renderer == "text":
            observation = render_chess_tactics_text(self._instance)
        elif self._renderer == "image":
            observation = render_chess_tactics_image(self._instance)
        else:
            raise ValueError(f"unknown chess renderer: {self._renderer}")
        return TaskTurn(
            turn_index=turn_index,
            observation=observation,
            submission_modes=("final_text",),
            action_schema={},
            public_limits=self._instance.limits.as_public_dict(),
            public_info={
                "task_id": self._instance.task_id,
                "kind": self._instance.kind,
                "fen": self._instance.public_payload["fen"],
                "side_to_move": self._instance.public_payload["side_to_move"],
                "allowed_notation": self._instance.public_payload["allowed_notation"],
            },
        )
