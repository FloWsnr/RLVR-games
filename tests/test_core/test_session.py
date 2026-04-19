"""Task-native session contract tests."""

import pytest

from rlvr_games.core import (
    ActionContext,
    ChatMessage,
    MessageRole,
    EnvironmentTaskSession,
    EpisodeConfig,
    EpisodeFinishedError,
    EnvironmentNotResetError,
    Observation,
    TaskInstance,
    TaskResetResult,
    TaskSubmissionResult,
    TaskTrajectory,
    TaskTurn,
    TextMessagePart,
)
from tests.test_core.support import CounterBackend, make_counter_env


def _task_turn() -> TaskTurn:
    """Return a minimal task turn for session type tests."""
    observation = Observation(text="What is 2 + 2?", metadata={"kind": "prompt"})
    return TaskTurn(
        observation=observation,
        action_context=ActionContext(turn_index=0),
        messages=(
            ChatMessage(
                role=MessageRole.USER,
                content=(TextMessagePart(text="What is 2 + 2?"),),
            ),
        ),
    )


def test_task_instance_validates_identity_and_copies_metadata() -> None:
    metadata: dict[str, object] = {"difficulty": "easy"}

    task_instance = TaskInstance(
        task_instance_id="arithmetic:7",
        task_kind="arithmetic",
        seed=7,
        prompt_key="add-2-2",
        metadata=metadata,
    )
    metadata["difficulty"] = "hard"

    assert task_instance.task_instance_id == "arithmetic:7"
    assert task_instance.task_kind == "arithmetic"
    assert task_instance.seed == 7
    assert task_instance.prompt_key == "add-2-2"
    assert task_instance.metadata == {"difficulty": "easy"}


def test_task_instance_rejects_empty_identity() -> None:
    with pytest.raises(ValueError, match="task_instance_id"):
        TaskInstance(task_instance_id="", task_kind="arithmetic", seed=0)


def test_task_reset_result_exposes_observation_and_copies_info() -> None:
    turn = _task_turn()
    info: dict[str, object] = {"seed": 3}

    reset_result = TaskResetResult(
        task_instance_id="arithmetic:3",
        observation=turn.observation,
        turn=turn,
        info=info,
    )
    info["seed"] = 4

    assert reset_result.observation is turn.observation
    assert reset_result.turn is turn
    assert reset_result.info == {"seed": 3}


def test_task_submission_result_done_and_metadata_are_stable() -> None:
    info: dict[str, object] = {"correct": True}
    debug_info: dict[str, object] = {"expected": 4}

    result = TaskSubmissionResult(
        task_instance_id="arithmetic:1",
        assistant_output="The answer is 4",
        raw_submission="4",
        parsed_output=4,
        valid_submission=True,
        reward=1.0,
        terminated=True,
        truncated=False,
        observation=None,
        turn=None,
        info=info,
        debug_info=debug_info,
    )
    info["correct"] = False
    debug_info["expected"] = 5

    assert result.done is True
    assert result.info == {"correct": True}
    assert result.debug_info == {"expected": 4}


def test_task_submission_result_rejects_finished_result_with_next_turn() -> None:
    with pytest.raises(ValueError, match="Finished task submissions"):
        TaskSubmissionResult(
            task_instance_id="arithmetic:1",
            assistant_output="The answer is 4",
            raw_submission="4",
            parsed_output=4,
            valid_submission=True,
            reward=1.0,
            terminated=True,
            truncated=False,
            observation=None,
            turn=_task_turn(),
        )


def test_task_submission_result_distinguishes_validity_from_reward() -> None:
    result = TaskSubmissionResult(
        task_instance_id="arithmetic:1",
        assistant_output="The answer is 5",
        raw_submission="5",
        parsed_output=5,
        valid_submission=True,
        reward=0.0,
        terminated=True,
        truncated=False,
        observation=None,
        turn=None,
    )

    assert result.valid_submission is True
    assert result.reward == 0.0
    assert result.done is True


def test_task_trajectory_records_submission_results() -> None:
    trajectory = TaskTrajectory(
        task_instance_id="arithmetic:1",
        initial_turn=_task_turn(),
        reset_info={"seed": 1},
        debug_reset_info={"answer": 4},
    )
    result = TaskSubmissionResult(
        task_instance_id="arithmetic:1",
        assistant_output="The answer is 4",
        raw_submission="4",
        parsed_output=4,
        valid_submission=True,
        reward=1.0,
        terminated=True,
        truncated=False,
        observation=None,
        turn=None,
        info={"correct": True},
        debug_info={"answer": 4},
    )

    trajectory.append(result, details={"adapter": "single_step"})

    assert trajectory.total_reward == 1.0
    assert trajectory.done is True
    assert len(trajectory.submissions) == 1
    record = trajectory.submissions[0]
    assert record.assistant_output == "The answer is 4"
    assert record.raw_submission == "4"
    assert record.parsed_output == 4
    assert record.valid_submission is True
    assert record.info == {"correct": True}
    assert record.debug_info == {"answer": 4}
    assert record.details == {"adapter": "single_step"}


def test_task_trajectory_rejects_mismatched_task_instance_id() -> None:
    trajectory = TaskTrajectory(
        task_instance_id="arithmetic:1",
        initial_turn=_task_turn(),
    )
    result = TaskSubmissionResult(
        task_instance_id="arithmetic:2",
        assistant_output="4",
        raw_submission="4",
        parsed_output=4,
        valid_submission=True,
        reward=1.0,
        terminated=True,
        truncated=False,
        observation=None,
        turn=None,
    )

    with pytest.raises(ValueError, match="does not match trajectory"):
        trajectory.append(result)


def test_environment_task_session_reset_prepares_task_turn() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = EnvironmentTaskSession(env=env, task_kind="counter")

    reset_result = session.reset(seed=11)

    assert reset_result.task_instance_id == "counter:seed=11:episode=0"
    assert reset_result.info == {"scenario": "counter", "seed": 11}
    assert reset_result.observation is not None
    assert reset_result.observation.metadata["value"] == 0
    assert reset_result.turn is not None
    assert reset_result.turn.action_context.turn_index == 0
    assert session.task_instance_id == "counter:seed=11:episode=0"
    assert session.turn is reset_result.turn
    assert session.done is False
    assert session.trajectory.initial_turn is reset_result.turn


def test_environment_task_session_submit_maps_step_result_to_task_result() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = EnvironmentTaskSession(
        env=env,
        task_kind="counter",
        action_extractor=lambda assistant_output: assistant_output.removeprefix(
            "move: "
        ),
    )
    session.reset(seed=3)

    result = session.submit("move: 1")

    assert result.task_instance_id == "counter:seed=3:episode=0"
    assert result.assistant_output == "move: 1"
    assert result.raw_submission == "1"
    assert result.parsed_output is not None
    assert result.valid_submission is True
    assert result.reward == 1.0
    assert result.done is False
    assert result.observation is not None
    assert result.observation.metadata["value"] == 1
    assert result.turn is not None
    assert result.turn.action_context.turn_index == 1
    assert session.episode_return == 1.0
    assert len(session.trajectory.submissions) == 1
    assert session.trajectory.submissions[0].raw_submission == "1"
    assert session.trajectory.submissions[0].details == {
        "adapter": "environment",
        "accepted": True,
        "transition_count": 1,
    }


def test_environment_task_session_terminal_submit_keeps_final_observation() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = EnvironmentTaskSession(env=env, task_kind="counter")
    session.reset(seed=5)

    session.submit("1")
    session.submit("1")
    result = session.submit("1")

    assert result.done is True
    assert result.terminated is True
    assert result.truncated is False
    assert result.turn is None
    assert result.observation is not None
    assert result.observation.metadata["value"] == 3
    assert session.turn is None
    assert session.done is True
    assert session.trajectory.done is True
    assert session.trajectory.total_reward == 3.0


def test_environment_task_session_lifecycle_errors() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = EnvironmentTaskSession(env=env, task_kind="counter")

    with pytest.raises(EnvironmentNotResetError):
        _ = session.turn
    with pytest.raises(EnvironmentNotResetError):
        session.submit("1")

    session.reset(seed=5)
    session.submit("1")
    session.submit("1")
    session.submit("1")

    with pytest.raises(EpisodeFinishedError):
        session.submit("1")


def test_environment_task_session_reset_after_done_starts_new_task_instance() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )
    session = EnvironmentTaskSession(env=env, task_kind="counter")
    session.reset(seed=5)
    session.submit("1")
    session.submit("1")
    session.submit("1")

    reset_result = session.reset(seed=5)

    assert reset_result.task_instance_id == "counter:seed=5:episode=1"
    assert session.task_instance_id == "counter:seed=5:episode=1"
    assert session.done is False
    assert session.episode_return == 0.0
    assert session.trajectory.submissions == []
