"""Tests for generic multi-turn adapter helpers."""

import pytest

from rlvr_physics.adapters.multiturn import (
    ScalarSessionEnvironment,
    SessionStepRecord,
    format_step_feedback,
    run_action_rollout,
)
from rlvr_physics.core.factory import ConfiguredTaskFactory
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.tasks.games.game2048 import (
    Game2048Session,
    game2048_task_spec,
    make_2048_instance,
)


def _game2048_text_session(instance: TaskInstance) -> TaskSession:
    return Game2048Session(instance, "text")


def _game2048_factory() -> ConfiguredTaskFactory:
    return ConfiguredTaskFactory(
        spec=game2048_task_spec(seed=5, max_turns=2, target_tile=2048),
        session_builder=_game2048_text_session,
    )


def test_scalar_session_environment_resets_from_nested_row_mapping() -> None:
    instance = make_2048_instance(seed=5, max_turns=2, target_tile=2048)
    environment = ScalarSessionEnvironment(
        instances={instance.task_id: instance},
        task_factory=_game2048_factory(),
        seed=4,
    )

    observation = environment.reset_from_mapping(
        {"extra_info": {"task_id": instance.task_id}}
    )

    assert "2048" in observation
    assert environment.task_id == instance.task_id
    assert environment.initial_observation == observation
    assert environment.steps == ()
    assert environment.step_rewards == ()
    assert environment.total_reward == 0
    assert environment.final_score is None
    assert not environment.done


def test_scalar_session_environment_records_steps_and_clears_on_reset() -> None:
    instance = make_2048_instance(seed=5, max_turns=2, target_tile=2048)
    environment = ScalarSessionEnvironment(
        instances={instance.task_id: instance},
        task_factory=_game2048_factory(),
        seed=4,
    )
    environment.reset(instance.task_id)

    first_step = environment.submit_action("up")
    feedback = environment.submit_action_text("right")

    assert first_step.submission_kind == "action"
    assert first_step.raw_submission == "up"
    assert first_step.accepted
    assert first_step.reward == 4.0
    assert first_step.score == 4.0
    assert not first_step.done
    assert first_step.observation is not None
    assert "2048" in first_step.observation
    assert "reward: 0.0" in feedback
    assert "done: True" in feedback
    assert environment.step_rewards == (4.0, 0.0)
    assert environment.total_reward == 4.0
    assert environment.final_score == 4.0
    assert environment.done

    environment.reset(instance.task_id)

    assert environment.steps == ()
    assert environment.step_rewards == ()
    assert environment.total_reward == 0
    assert environment.final_score is None
    assert not environment.done


def test_scalar_session_environment_requires_reset_before_submit() -> None:
    instance = make_2048_instance(seed=5, max_turns=2, target_tile=2048)
    environment = ScalarSessionEnvironment(
        instances={instance.task_id: instance},
        task_factory=_game2048_factory(),
        seed=4,
    )

    with pytest.raises(ValueError, match="must be reset"):
        environment.submit_action("up")


def test_run_action_rollout_stops_after_done_step() -> None:
    instance = make_2048_instance(seed=5, max_turns=2, target_tile=2048)

    initial_observation, steps = run_action_rollout(
        instance=instance,
        task_factory=_game2048_factory(),
        seed=4,
        actions=("up", "right", "left"),
    )

    assert "2048" in initial_observation
    assert len(steps) == 2
    assert tuple(step.raw_submission for step in steps) == ("up", "right")
    assert tuple(step.reward for step in steps) == (4.0, 0.0)
    assert not steps[0].done
    assert steps[1].done
    assert steps[1].truncated


def test_step_record_as_dict_and_feedback_are_plain_public_surfaces() -> None:
    step = SessionStepRecord(
        submission_kind="action",
        raw_submission="up",
        accepted=True,
        reward=4.0,
        score=4.0,
        done=False,
        terminal=False,
        truncated=False,
        observation="next state",
        public_info={"reason": "playing"},
        debug_info={"board": ((4, 0), (0, 0))},
    )

    data = step.as_dict()
    feedback = format_step_feedback(step)

    assert data["submission_kind"] == "action"
    assert data["public_info"] == {"reason": "playing"}
    assert data["debug_info"] == {"board": [[4, 0], [0, 0]]}
    assert "reason: playing" in feedback
    assert "observation:\nnext state" in feedback
