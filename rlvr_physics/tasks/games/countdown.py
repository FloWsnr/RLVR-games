"""Reasoning Gym Countdown task wrapper."""

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
import ast
import re

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.games.countdown import CountdownConfig, CountdownDataset

from rlvr_physics.core.instances import (
    TaskInstance,
    TaskLimits,
    require_int,
    require_tuple_of_ints,
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
from rlvr_physics.core.trajectory import TaskTrajectory

COUNTDOWN_KIND = "games.countdown.v1"
COUNTDOWN_DOMAIN = "games"


@dataclass(frozen=True)
class CountdownVerification:
    """Verification result for one Countdown expression."""

    accepted: bool
    correct: bool
    reward: float
    reason: str
    expression: str
    value: Fraction | None
    used_numbers: tuple[int, ...]


def countdown_task_spec(seed: int, size: int) -> TaskSpec:
    """Return the Python task spec for Countdown instances.

    Parameters
    ----------
    seed:
        Procedural Reasoning Gym seed.
    size:
        Virtual source dataset size.
    """

    return TaskSpec(
        kind=COUNTDOWN_KIND,
        domain=COUNTDOWN_DOMAIN,
        source=SourceSpec(
            source_type="reasoning_gym.countdown", seed=seed, parameters={"size": size}
        ),
        renderers=(
            RendererSpec(renderer_type="text"),
            RendererSpec(renderer_type="image"),
        ),
        verifier=VerifierSpec(
            verifier_type="arithmetic_expression",
            parameters={"uses_all_numbers_once": True, "exact_target": True},
        ),
        reward=RewardSpec(
            reward_type="graded_countdown",
            parameters={
                "correct": 1.0,
                "wrong_numbers": 0.05,
                "wrong_value": 0.05,
                "invalid": 0.01,
            },
        ),
        limits=TaskLimits(max_turns=1, token_budget=256),
        metadata={"exports": {"dataset": {"ability": "countdown"}}},
    )


def make_countdown_instance(seed: int, source_index: int) -> TaskInstance:
    """Sample one immutable Countdown task instance from Reasoning Gym.

    Parameters
    ----------
    seed:
        Procedural Reasoning Gym seed.
    source_index:
        Deterministic source index within the virtual dataset.
    """

    config = CountdownConfig(seed=seed, size=max(source_index + 1, 1))
    dataset = CountdownDataset(config)
    entry = dataset[source_index]
    metadata = entry["metadata"]
    numbers = tuple(int(number) for number in metadata["numbers"])
    target = int(metadata["target"])
    reference_expression = str(entry["answer"])
    task_id = (
        "countdown-"
        + stable_hash(
            {
                "seed": seed,
                "source_index": source_index,
                "numbers": numbers,
                "target": target,
                "reference_expression": reference_expression,
            }
        )[:16]
    )
    return TaskInstance(
        task_id=task_id,
        kind=COUNTDOWN_KIND,
        domain=COUNTDOWN_DOMAIN,
        seed=seed,
        public_payload={
            "numbers": numbers,
            "target": target,
            "question": str(entry["question"]),
        },
        privileged_payload={
            "reference_expression": reference_expression,
            "source_dataset": str(metadata["source_dataset"]),
            "source_index": source_index,
        },
        limits=TaskLimits(max_turns=1, token_budget=256),
        metadata={
            "source_dataset": str(metadata["source_dataset"]),
            "source_index": source_index,
        },
    )


def render_countdown_text(instance: TaskInstance) -> RenderedObservation:
    """Render a Countdown instance as plain text."""

    numbers = require_tuple_of_ints(instance.public_payload["numbers"], "numbers")
    target = require_int(instance.public_payload["target"], "target")
    prompt = (
        "Countdown numbers game\n"
        f"Target: {target}\n"
        f"Numbers: {', '.join(str(number) for number in numbers)}\n\n"
        "Submit one arithmetic expression that uses every listed number exactly once "
        "and evaluates to the target. Allowed operators: +, -, *, /, and parentheses."
    )
    return text_observation("text", prompt)


def render_countdown_image(instance: TaskInstance) -> RenderedObservation:
    """Render a Countdown instance as a PNG image observation."""

    numbers = require_tuple_of_ints(instance.public_payload["numbers"], "numbers")
    target = require_int(instance.public_payload["target"], "target")
    image = Image.new("RGB", (720, 420), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default(size=34)
    font_large = ImageFont.load_default(size=54)
    font_body = ImageFont.load_default(size=24)

    draw.rectangle((0, 0, 720, 420), fill="#f7f8fa")
    draw.text((40, 34), "Countdown", fill="#1c2331", font=font_title)
    draw.text((40, 92), f"Target {target}", fill="#0c4a6e", font=font_large)

    tile_width = 96
    tile_height = 72
    gap = 18
    total_width = len(numbers) * tile_width + (len(numbers) - 1) * gap
    start_x = (720 - total_width) // 2
    y = 210
    for index, number in enumerate(numbers):
        x = start_x + index * (tile_width + gap)
        draw.rounded_rectangle(
            (x, y, x + tile_width, y + tile_height),
            radius=8,
            fill="#ffffff",
            outline="#8aa0b4",
            width=2,
        )
        label = str(number)
        bbox = draw.textbbox((0, 0), label, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (
                x + (tile_width - text_width) / 2,
                y + (tile_height - text_height) / 2 - 4,
            ),
            label,
            fill="#1f2937",
            font=font_large,
        )

    draw.text(
        (40, 340), "Use each number once with + - * /", fill="#364152", font=font_body
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    alt_text = render_countdown_text(instance).text()
    return image_observation("image", buffer.getvalue(), alt_text)


def verify_countdown_submission(
    instance: TaskInstance, submission: TaskSubmission
) -> CountdownVerification:
    """Verify one Countdown model submission."""

    numbers = require_tuple_of_ints(instance.public_payload["numbers"], "numbers")
    target = require_int(instance.public_payload["target"], "target")
    expression = _extract_expression(submission.raw)
    if not expression:
        return CountdownVerification(
            accepted=False,
            correct=False,
            reward=0.01,
            reason="empty_submission",
            expression="",
            value=None,
            used_numbers=(),
        )
    try:
        parsed = ast.parse(expression, mode="eval")
        used_numbers: list[int] = []
        value = _evaluate_countdown_ast(parsed.body, used_numbers)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError):
        return CountdownVerification(
            accepted=False,
            correct=False,
            reward=0.01,
            reason="invalid_expression",
            expression=expression,
            value=None,
            used_numbers=(),
        )

    used_tuple = tuple(used_numbers)
    if sorted(used_tuple) != sorted(numbers):
        return CountdownVerification(
            accepted=True,
            correct=False,
            reward=0.05,
            reason="wrong_numbers",
            expression=expression,
            value=value,
            used_numbers=used_tuple,
        )
    if value != Fraction(target, 1):
        return CountdownVerification(
            accepted=True,
            correct=False,
            reward=0.05,
            reason="wrong_value",
            expression=expression,
            value=value,
            used_numbers=used_tuple,
        )
    return CountdownVerification(
        accepted=True,
        correct=True,
        reward=1.0,
        reason="correct",
        expression=expression,
        value=value,
        used_numbers=used_tuple,
    )


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


def _extract_expression(raw_submission: str) -> str:
    stripped = raw_submission.strip()
    if not stripped:
        return ""
    fenced = re.search(
        r"```(?:text|python)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced is not None:
        stripped = fenced.group(1).strip()
    answer_match = re.search(
        r"(?:answer|expression)\s*[:=]\s*(.+)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match is not None:
        stripped = answer_match.group(1).strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    if "=" in candidate:
        candidate = candidate.split("=")[0].strip()
    return candidate


def _evaluate_countdown_ast(node: ast.AST, used_numbers: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_countdown_ast(node.body, used_numbers)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise ValueError("Countdown expressions may only contain integer constants")
        used_numbers.append(node.value)
        return Fraction(node.value, 1)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate_countdown_ast(node.operand, used_numbers)
        if isinstance(node.op, ast.USub):
            return -value
        return value
    if isinstance(node, ast.BinOp):
        left = _evaluate_countdown_ast(node.left, used_numbers)
        right = _evaluate_countdown_ast(node.right, used_numbers)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
    raise ValueError("unsupported expression syntax")
