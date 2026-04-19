"""Single-step verifier session tests."""

from dataclasses import dataclass

import pytest

from rlvr_games.core import (
    EpisodeFinishedError,
    EnvironmentNotResetError,
    Observation,
    SingleStepTask,
    SingleStepVerifierSession,
    TaskInstance,
    TextMessagePart,
    VerificationResult,
)


@dataclass(slots=True, frozen=True)
class AdditionPayload:
    """Tiny arithmetic payload for single-step session tests."""

    left: int
    right: int

    @property
    def answer(self) -> int:
        """Return the expected sum."""
        return self.left + self.right


class AdditionTaskSource:
    """Deterministic task source for addition prompts."""

    def sample(self, *, seed: int) -> SingleStepTask[AdditionPayload]:
        """Return one deterministic addition task."""
        payload = AdditionPayload(left=seed, right=2)
        return SingleStepTask(
            instance=TaskInstance(
                task_instance_id=f"addition:{seed}",
                task_kind="addition",
                seed=seed,
                prompt_key=f"{seed}+2",
                metadata={"left": seed, "right": 2},
            ),
            payload=payload,
        )


class FixedAdditionTaskSource:
    """Task source that returns the same task instance for every seed."""

    def __init__(self, task: SingleStepTask[AdditionPayload]) -> None:
        self._task = task

    def sample(self, *, seed: int) -> SingleStepTask[AdditionPayload]:
        """Return the fixed task."""
        del seed
        return self._task


class AdditionPromptRenderer:
    """Render addition tasks as prompt observations."""

    def render(self, task: SingleStepTask[AdditionPayload]) -> Observation:
        """Render the addition prompt."""
        return Observation(
            text=f"What is {task.payload.left} + {task.payload.right}?",
            metadata={"task_instance_id": task.instance.task_instance_id},
        )


class AdditionVerifier:
    """Verify integer addition completions."""

    def verify(
        self,
        *,
        task: SingleStepTask[AdditionPayload],
        completion: str,
    ) -> VerificationResult:
        """Reward exact integer answers."""
        try:
            parsed = int(completion.strip())
        except ValueError:
            return VerificationResult(
                parsed_output=None,
                valid_submission=False,
                reward=-1.0,
                info={"error": "not_an_integer"},
                debug_info={"expected": task.payload.answer},
            )
        correct = parsed == task.payload.answer
        return VerificationResult(
            parsed_output=parsed,
            valid_submission=True,
            reward=1.0 if correct else 0.0,
            info={"correct": correct},
            debug_info={"expected": task.payload.answer},
        )


def _make_addition_session() -> SingleStepVerifierSession[AdditionPayload]:
    """Return a standard addition verifier session."""
    return SingleStepVerifierSession(
        task_source=AdditionTaskSource(),
        prompt_renderer=AdditionPromptRenderer(),
        verifier=AdditionVerifier(),
    )


def test_single_step_session_reset_prepares_prompt_turn() -> None:
    session = _make_addition_session()

    reset_result = session.reset(seed=2)

    assert reset_result.task_instance_id == "addition:2"
    assert reset_result.observation is not None
    assert reset_result.observation.text == "What is 2 + 2?"
    assert reset_result.info["task_kind"] == "addition"
    assert reset_result.info["prompt_key"] == "2+2"
    assert reset_result.turn is not None
    text_part = reset_result.turn.messages[0].content[0]
    assert isinstance(text_part, TextMessagePart)
    assert text_part.text == "Observation:\nWhat is 2 + 2?"
    assert session.task_instance_id == "addition:2"
    assert session.done is False


def test_single_step_session_correct_completion_finishes() -> None:
    session = _make_addition_session()
    session.reset(seed=2)

    result = session.submit("4")

    assert result.task_instance_id == "addition:2"
    assert result.assistant_output == "4"
    assert result.raw_submission == "4"
    assert result.parsed_output == 4
    assert result.valid_submission is True
    assert result.reward == 1.0
    assert result.done is True
    assert result.turn is None
    assert session.done is True
    assert session.episode_return == 1.0
    assert session.trajectory.total_reward == 1.0
    assert session.trajectory.submissions[0].details == {"adapter": "single_step"}


def test_single_step_session_wrong_answer_is_valid_low_reward() -> None:
    session = _make_addition_session()
    session.reset(seed=2)

    result = session.submit("5")

    assert result.parsed_output == 5
    assert result.valid_submission is True
    assert result.reward == 0.0
    assert result.info == {"correct": False}
    assert result.debug_info == {"expected": 4}


def test_single_step_session_malformed_answer_is_invalid_submission() -> None:
    session = _make_addition_session()
    session.reset(seed=2)

    result = session.submit("not a number")

    assert result.parsed_output is None
    assert result.valid_submission is False
    assert result.reward == -1.0
    assert result.info == {"error": "not_an_integer"}
    assert result.debug_info == {"expected": 4}


def test_single_step_session_lifecycle_errors() -> None:
    session = _make_addition_session()

    with pytest.raises(EnvironmentNotResetError):
        _ = session.turn
    with pytest.raises(EnvironmentNotResetError):
        session.submit("4")

    session.reset(seed=2)
    session.submit("4")

    with pytest.raises(EpisodeFinishedError):
        session.submit("4")


def test_single_step_sessions_can_share_task_instance_id() -> None:
    task = SingleStepTask(
        instance=TaskInstance(
            task_instance_id="addition:fixed",
            task_kind="addition",
            seed=99,
            prompt_key="fixed",
        ),
        payload=AdditionPayload(left=2, right=2),
    )
    source = FixedAdditionTaskSource(task=task)
    first_session = SingleStepVerifierSession(
        task_source=source,
        prompt_renderer=AdditionPromptRenderer(),
        verifier=AdditionVerifier(),
    )
    second_session = SingleStepVerifierSession(
        task_source=source,
        prompt_renderer=AdditionPromptRenderer(),
        verifier=AdditionVerifier(),
    )

    first_session.reset(seed=1)
    second_session.reset(seed=2)
    first_result = first_session.submit("4")
    second_result = second_session.submit("5")

    assert first_session.task_instance_id == "addition:fixed"
    assert second_session.task_instance_id == "addition:fixed"
    assert first_result.reward == 1.0
    assert second_result.reward == 0.0
    assert first_session.trajectory is not second_session.trajectory


def test_verification_result_requires_exactly_one_boundary() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        VerificationResult(
            parsed_output=None,
            valid_submission=False,
            reward=0.0,
            terminated=False,
            truncated=False,
        )
