"""Chess tactics task using python-chess."""

from dataclasses import dataclass

import chess
import chess.svg

from rlvr_physics.core.instances import (
    TaskInstance,
    TaskLimits,
    require_int,
    require_str,
    stable_hash,
)
from rlvr_physics.core.rendering import (
    ImageContent,
    RenderedObservation,
    TextContent,
    text_observation,
)
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
from rlvr_physics.core.trajectory import TaskTrajectory

CHESS_KIND = "games.chess_tactics.v1"
CHESS_DOMAIN = "games"
FOOLS_MATE_IN_ONE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"


@dataclass(frozen=True)
class ChessVerification:
    """Verification result for one chess tactic submission."""

    accepted: bool
    correct: bool
    reward: float
    reason: str
    move_uci: str | None
    move_san: str | None


def chess_tactics_task_spec(seed: int) -> TaskSpec:
    """Return the Python task spec for chess tactics.

    Parameters
    ----------
    seed:
        Source seed for deterministic puzzle selection.
    """

    return TaskSpec(
        kind=CHESS_KIND,
        domain=CHESS_DOMAIN,
        source=SourceSpec(
            source_type="builtin.chess_tactics", seed=seed, parameters={"mate_depth": 1}
        ),
        renderers=(
            RendererSpec(renderer_type="text"),
            RendererSpec(renderer_type="image"),
        ),
        verifier=VerifierSpec(
            verifier_type="python_chess_mate_in_one",
            parameters={"accepted_notation": ("SAN", "UCI")},
        ),
        reward=RewardSpec(
            reward_type="mate_in_one",
            parameters={
                "correct": 1.0,
                "legal_non_solution": 0.2,
                "illegal_or_parse_failure": 0.0,
            },
        ),
        limits=TaskLimits(max_turns=1, token_budget=64),
        metadata={"exports": {"dataset": {"ability": "chess_tactics"}}},
    )


def make_chess_tactics_instance(seed: int, puzzle_id: str) -> TaskInstance:
    """Create one immutable mate-in-one chess tactic instance."""

    fen = _select_puzzle_fen(seed, puzzle_id)
    board = chess.Board(fen)
    solutions = _mate_in_one_moves(board)
    if not solutions:
        raise ValueError("selected puzzle has no mate-in-one solution")
    task_id = (
        "chess-"
        + stable_hash(
            {"seed": seed, "puzzle_id": puzzle_id, "fen": fen, "solutions": solutions}
        )[:16]
    )
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    return TaskInstance(
        task_id=task_id,
        kind=CHESS_KIND,
        domain=CHESS_DOMAIN,
        seed=seed,
        public_payload={
            "fen": fen,
            "side_to_move": side_to_move,
            "mate_depth": 1,
            "allowed_notation": ("SAN", "UCI"),
        },
        privileged_payload={
            "puzzle_id": puzzle_id,
            "solution_moves_uci": solutions,
            "source": "builtin",
        },
        limits=TaskLimits(max_turns=1, token_budget=64),
        metadata={"puzzle_id": puzzle_id, "mate_depth": 1},
    )


def render_chess_tactics_text(instance: TaskInstance) -> RenderedObservation:
    """Render a chess tactic as plain text."""

    fen = require_str(instance.public_payload["fen"], "fen")
    side_to_move = require_str(instance.public_payload["side_to_move"], "side_to_move")
    mate_depth = require_int(instance.public_payload["mate_depth"], "mate_depth")
    board = chess.Board(fen)
    prompt = (
        f"Chess tactic: mate in {mate_depth}\n"
        f"Side to move: {side_to_move}\n"
        f"FEN: {fen}\n\n"
        f"{board}\n\n"
        "Submit the winning move in SAN or UCI notation."
    )
    return text_observation("text", prompt)


def render_chess_tactics_image(instance: TaskInstance) -> RenderedObservation:
    """Render a chess tactic as an SVG image observation."""

    fen = require_str(instance.public_payload["fen"], "fen")
    board = chess.Board(fen)
    alt_text = render_chess_tactics_text(instance).text()
    svg = chess.svg.board(board=board, orientation=chess.WHITE, size=640)
    return RenderedObservation(
        renderer_name="image",
        contents=(
            ImageContent(
                data=svg.encode("utf-8"),
                mime_type="image/svg+xml",
                alt_text=alt_text,
            ),
            TextContent(text=alt_text),
        ),
    )


def verify_chess_tactic_submission(
    instance: TaskInstance, submission: TaskSubmission
) -> ChessVerification:
    """Verify one chess tactic submission."""

    fen = require_str(instance.public_payload["fen"], "fen")
    board = chess.Board(fen)
    raw = submission.raw.strip()
    if not raw:
        return ChessVerification(False, False, 0.0, "empty_submission", None, None)
    move = _parse_move(board, raw)
    if move is None:
        return ChessVerification(False, False, 0.0, "parse_failure", None, None)
    if move not in board.legal_moves:
        return ChessVerification(False, False, 0.0, "illegal_move", move.uci(), None)
    san = board.san(move)
    board.push(move)
    if board.is_checkmate():
        return ChessVerification(True, True, 1.0, "checkmate", move.uci(), san)
    return ChessVerification(True, False, 0.2, "legal_non_solution", move.uci(), san)


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


def _select_puzzle_fen(seed: int, puzzle_id: str) -> str:
    if puzzle_id == "fools_mate_mate_in_one":
        return FOOLS_MATE_IN_ONE_FEN
    if puzzle_id == "seeded":
        return FOOLS_MATE_IN_ONE_FEN
    raise ValueError(f"unknown chess tactic puzzle id: {puzzle_id} (seed={seed})")


def _mate_in_one_moves(board: chess.Board) -> tuple[str, ...]:
    solutions: list[str] = []
    for move in board.legal_moves:
        candidate = board.copy(stack=False)
        candidate.push(move)
        if candidate.is_checkmate():
            solutions.append(move.uci())
    return tuple(sorted(solutions))


def _parse_move(board: chess.Board, raw: str) -> chess.Move | None:
    cleaned = raw.strip()
    try:
        return board.parse_san(cleaned)
    except ValueError:
        pass
    try:
        return chess.Move.from_uci(cleaned.lower())
    except ValueError:
        return None
