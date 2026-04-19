"""Seeded 2048 task."""

from dataclasses import dataclass
from io import BytesIO
from random import Random

from PIL import Image, ImageDraw, ImageFont

from rlvr_physics.core.instances import (
    TaskInstance,
    TaskLimits,
    require_int,
    stable_hash,
)
from rlvr_physics.core.rendering import (
    RenderedObservation,
    image_observation,
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
from rlvr_physics.core.trajectory import TaskTrajectory, TrajectoryEvent

GAME2048_KIND = "games.2048.v1"
GAME2048_DOMAIN = "games"
GAME2048_ACTIONS = ("up", "down", "left", "right")
Board = tuple[tuple[int, ...], ...]
SpawnTape = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Game2048State:
    """Canonical 2048 runtime state."""

    board: Board
    score: int
    turns: int
    max_tile: int
    spawn_index: int


@dataclass(frozen=True)
class MoveResult:
    """Result of applying one 2048 action before tile spawn."""

    board: Board
    score_delta: int
    changed: bool


def game2048_task_spec(seed: int, max_turns: int, target_tile: int) -> TaskSpec:
    """Return the Python task spec for seeded 2048.

    Parameters
    ----------
    seed:
        Seed used to generate the spawn tape.
    max_turns:
        Maximum valid moves before truncation.
    target_tile:
        Tile value that ends the task successfully.
    """

    return TaskSpec(
        kind=GAME2048_KIND,
        domain=GAME2048_DOMAIN,
        source=SourceSpec(
            source_type="procedural.spawn_tape",
            seed=seed,
            parameters={"max_turns": max_turns, "target_tile": target_tile},
        ),
        renderers=(
            RendererSpec(renderer_type="text"),
            RendererSpec(renderer_type="image"),
        ),
        verifier=VerifierSpec(
            verifier_type="deterministic_2048_rules",
            parameters={"actions": GAME2048_ACTIONS},
        ),
        reward=RewardSpec(
            reward_type="score_delta", parameters={"invalid_action": -1.0}
        ),
        limits=TaskLimits(max_turns=max_turns, action_budget=max_turns),
        metadata={"exports": {"environment": {"action_space": GAME2048_ACTIONS}}},
    )


def make_2048_instance(seed: int, max_turns: int, target_tile: int) -> TaskInstance:
    """Create one deterministic 2048 task instance."""

    size = 4
    spawn_tape = _make_spawn_tape(seed, max_turns + 16, size)
    board: Board = tuple(tuple(0 for _ in range(size)) for _ in range(size))
    board, first_used = _apply_spawn(board, spawn_tape[0])
    board, second_used = _apply_spawn(board, spawn_tape[1])
    initial_spawn_count = int(first_used) + int(second_used)
    task_id = (
        "2048-"
        + stable_hash(
            {
                "seed": seed,
                "max_turns": max_turns,
                "target_tile": target_tile,
                "spawn_tape": spawn_tape,
                "initial_board": board,
            }
        )[:16]
    )
    return TaskInstance(
        task_id=task_id,
        kind=GAME2048_KIND,
        domain=GAME2048_DOMAIN,
        seed=seed,
        public_payload={
            "size": size,
            "initial_board": board,
            "target_tile": target_tile,
        },
        privileged_payload={
            "spawn_tape": spawn_tape,
            "initial_spawn_count": initial_spawn_count,
        },
        limits=TaskLimits(max_turns=max_turns, action_budget=max_turns),
        metadata={"source": "procedural.spawn_tape"},
    )


def initial_2048_state(instance: TaskInstance) -> Game2048State:
    """Build the canonical initial state for a 2048 instance."""

    board = _require_board(instance.public_payload["initial_board"], "initial_board")
    max_tile = max(max(row) for row in board)
    spawn_index = require_int(
        instance.privileged_payload["initial_spawn_count"], "initial_spawn_count"
    )
    return Game2048State(
        board=board, score=0, turns=0, max_tile=max_tile, spawn_index=spawn_index
    )


def render_2048_text(state: Game2048State, target_tile: int) -> RenderedObservation:
    """Render a 2048 state as plain text."""

    rows = []
    for row in state.board:
        rows.append(" ".join(f"{value:4d}" if value else "   ." for value in row))
    prompt = (
        "2048\n"
        f"Score: {state.score} | Moves: {state.turns} | Max tile: {state.max_tile} | Target: {target_tile}\n\n"
        + "\n".join(rows)
        + "\n\nSubmit one action: up, down, left, or right."
    )
    return text_observation("text", prompt)


def render_2048_image(state: Game2048State, target_tile: int) -> RenderedObservation:
    """Render a 2048 state as a PNG image observation."""

    tile = 104
    gap = 12
    margin = 32
    header = 92
    board_px = 4 * tile + 3 * gap
    width = board_px + 2 * margin
    height = board_px + 2 * margin + header
    image = Image.new("RGB", (width, height), "#f3f5f7")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default(size=30)
    font_tile = ImageFont.load_default(size=36)
    font_body = ImageFont.load_default(size=20)
    draw.text((margin, 24), "2048", fill="#1f2937", font=font_title)
    draw.text(
        (margin + 104, 31),
        f"Score {state.score}  Moves {state.turns}  Target {target_tile}",
        fill="#334155",
        font=font_body,
    )
    palette = {
        0: ("#d7dde3", "#d7dde3"),
        2: ("#eef2ff", "#1f2937"),
        4: ("#e0f2fe", "#1f2937"),
        8: ("#bbf7d0", "#14532d"),
        16: ("#fde68a", "#713f12"),
        32: ("#fed7aa", "#7c2d12"),
        64: ("#fecaca", "#7f1d1d"),
        128: ("#ddd6fe", "#312e81"),
        256: ("#c7d2fe", "#1e3a8a"),
        512: ("#bfdbfe", "#172554"),
        1024: ("#a7f3d0", "#064e3b"),
        2048: ("#fef08a", "#713f12"),
    }
    for row_index, row in enumerate(state.board):
        for col_index, value in enumerate(row):
            x = margin + col_index * (tile + gap)
            y = margin + header + row_index * (tile + gap)
            fill, text_color = palette.get(value, ("#99f6e4", "#134e4a"))
            draw.rounded_rectangle((x, y, x + tile, y + tile), radius=8, fill=fill)
            if value:
                label = str(value)
                bbox = draw.textbbox((0, 0), label, font=font_tile)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text(
                    (x + (tile - text_width) / 2, y + (tile - text_height) / 2 - 4),
                    label,
                    fill=text_color,
                    font=font_tile,
                )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return image_observation(
        "image", buffer.getvalue(), render_2048_text(state, target_tile).text()
    )


def move_board(board: Board, action: str) -> MoveResult:
    """Apply one 2048 move without spawning a new tile."""

    if action not in GAME2048_ACTIONS:
        raise ValueError(f"unknown 2048 action: {action}")
    if action == "left":
        moved_rows = [_merge_line(row) for row in board]
        moved_board = tuple(result[0] for result in moved_rows)
        score_delta = sum(result[1] for result in moved_rows)
    elif action == "right":
        moved_rows = [_merge_line(tuple(reversed(row))) for row in board]
        moved_board = tuple(tuple(reversed(result[0])) for result in moved_rows)
        score_delta = sum(result[1] for result in moved_rows)
    elif action == "up":
        columns = _transpose(board)
        moved_columns = [_merge_line(column) for column in columns]
        moved_board = _transpose(tuple(result[0] for result in moved_columns))
        score_delta = sum(result[1] for result in moved_columns)
    else:
        columns = _transpose(board)
        moved_columns = [_merge_line(tuple(reversed(column))) for column in columns]
        moved_board = _transpose(
            tuple(tuple(reversed(result[0])) for result in moved_columns)
        )
        score_delta = sum(result[1] for result in moved_columns)
    return MoveResult(
        board=moved_board, score_delta=score_delta, changed=moved_board != board
    )


def legal_2048_actions(board: Board) -> tuple[str, ...]:
    """Return actions that change the board."""

    legal: list[str] = []
    for action in GAME2048_ACTIONS:
        if move_board(board, action).changed:
            legal.append(action)
    return tuple(legal)


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
        action = _parse_action(submission)
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

        spawn_tape = _require_spawn_tape(
            self._instance.privileged_payload["spawn_tape"], "spawn_tape"
        )
        board = move.board
        spawn_used = False
        if self._state.spawn_index < len(spawn_tape):
            board, spawn_used = _apply_spawn(board, spawn_tape[self._state.spawn_index])
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
                "reason": _terminal_reason(terminal, truncated, max_tile, target_tile),
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


def _make_spawn_tape(seed: int, length: int, size: int) -> SpawnTape:
    rng = Random(seed)
    tape: list[tuple[int, int]] = []
    cells = size * size
    for _ in range(length):
        empty_index = rng.randrange(cells)
        value = 4 if rng.random() < 0.1 else 2
        tape.append((empty_index, value))
    return tuple(tape)


def _apply_spawn(board: Board, spawn: tuple[int, int]) -> tuple[Board, bool]:
    empties: list[tuple[int, int]] = []
    for row_index, row in enumerate(board):
        for col_index, value in enumerate(row):
            if value == 0:
                empties.append((row_index, col_index))
    if not empties:
        return board, False
    empty_index, value = spawn
    row_index, col_index = empties[empty_index % len(empties)]
    mutable = [list(row) for row in board]
    mutable[row_index][col_index] = value
    return tuple(tuple(row) for row in mutable), True


def _merge_line(line: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    values = [value for value in line if value != 0]
    merged: list[int] = []
    score_delta = 0
    index = 0
    while index < len(values):
        value = values[index]
        if index + 1 < len(values) and values[index + 1] == value:
            merged_value = value * 2
            merged.append(merged_value)
            score_delta += merged_value
            index += 2
        else:
            merged.append(value)
            index += 1
    merged.extend(0 for _ in range(len(line) - len(merged)))
    return tuple(merged), score_delta


def _transpose(board: Board) -> Board:
    return tuple(
        tuple(row[column_index] for row in board)
        for column_index in range(len(board[0]))
    )


def _parse_action(submission: TaskSubmission) -> str:
    if submission.kind == "action":
        parsed_action = submission.parsed.get("action")
        if isinstance(parsed_action, str):
            return parsed_action.strip().lower()
    return submission.raw.strip().lower()


def _terminal_reason(
    terminal: bool, truncated: bool, max_tile: int, target_tile: int
) -> str:
    if terminal and max_tile >= target_tile:
        return "target_tile_reached"
    if terminal:
        return "no_legal_moves"
    if truncated:
        return "max_turns"
    return "continue"


def _require_board(value: object, name: str) -> Board:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple board")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, tuple):
            raise TypeError(f"{name} rows must be tuples")
        ints: list[int] = []
        for item in row:
            if not isinstance(item, int):
                raise TypeError(f"{name} cells must be integers")
            ints.append(item)
        rows.append(tuple(ints))
    return tuple(rows)


def _require_spawn_tape(value: object, name: str) -> SpawnTape:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    tape: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} entries must be integer pairs")
        empty_index, tile_value = item
        if not isinstance(empty_index, int) or not isinstance(tile_value, int):
            raise TypeError(f"{name} entries must be integer pairs")
        tape.append((empty_index, tile_value))
    return tuple(tape)
