"""Scalar session for seeded 2048."""

from rlvr_physics.core.instances import TaskInstance, require_int
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.trajectory import TaskTrajectory, TrajectoryEvent
from rlvr_physics.tasks.games.game2048.constants import GAME2048_ACTIONS
from rlvr_physics.tasks.games.game2048.instances import initial_2048_state
from rlvr_physics.tasks.games.game2048.renderers import (
    render_2048_image,
    render_2048_text,
)
from rlvr_physics.tasks.games.game2048.rules import (
    apply_spawn,
    legal_2048_actions,
    move_board,
    parse_action,
    require_spawn_tape,
    terminal_reason,
)
from rlvr_physics.tasks.games.game2048.types import Game2048State


class Game2048Session:
    """Stateful seeded 2048 task session."""

    def __init__(self, instance: TaskInstance, renderer: str) -> None:
        self._instance = instance
        self._renderer = renderer
        self._state: Game2048State | None = None
        self._turn: TaskTurn | None = None
        self._trajectory = TaskTrajectory(task_id=instance.task_id, session_id="")
        self._submissions = 0

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh 2048 rollout."""

        session_id = new_session_id(self._instance.task_id, seed)
        self._trajectory = TaskTrajectory(
            task_id=self._instance.task_id, session_id=session_id
        )
        self._state = initial_2048_state(self._instance)
        self._submissions = 0
        self._turn = self._make_turn()
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
            {"board": self._state.board},
        )
        return TaskResetResult(
            session_id=session_id, turn=self._turn, trajectory=self._trajectory
        )

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current 2048 turn."""

        return self._turn

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the verified session trajectory."""

        return self._trajectory

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Apply one 2048 action submission."""

        if self._state is None or self._turn is None:
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

        self._submissions += 1
        action = parse_action(submission)
        turn_index = self._state.turns
        submit_event = self._trajectory.append(
            "submission",
            turn_index,
            {"kind": submission.kind, "raw": submission.raw},
            {"parsed_action": action},
        )
        if action not in GAME2048_ACTIONS:
            invalid_event = self._trajectory.append(
                "invalid_action", turn_index, {"reason": "unknown_action"}, {}
            )
            return self._invalid_result("unknown_action", (submit_event, invalid_event))

        move = move_board(self._state.board, action)
        if not move.changed:
            invalid_event = self._trajectory.append(
                "invalid_action",
                turn_index,
                {"reason": "move_did_not_change_board", "action": action},
                {"board": self._state.board},
            )
            return self._invalid_result(
                "move_did_not_change_board", (submit_event, invalid_event)
            )

        spawn_tape = require_spawn_tape(
            self._instance.privileged_payload["spawn_tape"], "spawn_tape"
        )
        board = move.board
        spawn_used = False
        if self._state.spawn_index < len(spawn_tape):
            board, spawn_used = apply_spawn(board, spawn_tape[self._state.spawn_index])
        next_spawn_index = self._state.spawn_index + int(spawn_used)
        max_tile = max(max(row) for row in board)
        next_turns = self._state.turns + 1
        score = self._state.score + move.score_delta
        target_tile = require_int(
            self._instance.public_payload["target_tile"], "target_tile"
        )
        next_state = Game2048State(
            board=board,
            score=score,
            turns=next_turns,
            max_tile=max_tile,
            spawn_index=next_spawn_index,
        )
        terminal = max_tile >= target_tile or len(legal_2048_actions(board)) == 0
        truncated = (
            self._submissions >= self._instance.limits.max_turns and not terminal
        )
        self._state = next_state
        self._turn = None if terminal or truncated else self._make_turn()
        transition_event = self._trajectory.append(
            "transition",
            turn_index,
            {
                "action": action,
                "score_delta": move.score_delta,
                "score": score,
                "max_tile": max_tile,
                "terminal": terminal,
                "truncated": truncated,
            },
            {"board": board, "spawn_used": spawn_used, "spawn_index": next_spawn_index},
        )
        step_events = [submit_event, transition_event]
        if self._turn is not None:
            observation_event = self._trajectory.append(
                "observation",
                self._turn.turn_index,
                {
                    "renderer": self._renderer,
                    "content_digests": self._turn.observation.content_digests(),
                },
                {"board": board},
            )
            step_events.append(observation_event)
        return TaskStepResult(
            accepted=True,
            reward=float(move.score_delta),
            score=float(score),
            terminal=terminal,
            truncated=truncated,
            observation=self._turn,
            public_info={
                "action": action,
                "score_delta": move.score_delta,
                "score": score,
                "max_tile": max_tile,
                "reason": terminal_reason(terminal, truncated, max_tile, target_tile),
                "submissions": self._submissions,
            },
            debug_info={
                "board": board,
                "spawn_index": next_spawn_index,
                "legal_actions": legal_2048_actions(board),
            },
            events=tuple(step_events),
        )

    def _invalid_result(
        self, reason: str, events: tuple[TrajectoryEvent, ...]
    ) -> TaskStepResult:
        truncated = self._submissions >= self._instance.limits.max_turns
        if truncated:
            self._turn = None
        return TaskStepResult(
            accepted=False,
            reward=-1.0,
            score=float(self._state.score) if self._state is not None else None,
            terminal=False,
            truncated=truncated,
            observation=None if truncated else self._turn,
            public_info={"reason": reason, "submissions": self._submissions},
            debug_info={"board": self._state.board if self._state is not None else ()},
            events=events,
        )

    def _make_turn(self) -> TaskTurn:
        if self._state is None:
            raise ValueError("session has not been reset")
        target_tile = require_int(
            self._instance.public_payload["target_tile"], "target_tile"
        )
        if self._renderer == "text":
            observation = render_2048_text(self._state, target_tile)
        elif self._renderer == "image":
            observation = render_2048_image(self._state, target_tile)
        else:
            raise ValueError(f"unknown 2048 renderer: {self._renderer}")
        return TaskTurn(
            turn_index=self._state.turns,
            observation=observation,
            submission_modes=("action", "final_text"),
            action_schema={"type": "string", "enum": GAME2048_ACTIONS},
            public_limits=self._instance.limits.as_public_dict(),
            public_info={
                "task_id": self._instance.task_id,
                "kind": self._instance.kind,
                "score": self._state.score,
                "max_tile": self._state.max_tile,
                "legal_actions": legal_2048_actions(self._state.board),
            },
        )
