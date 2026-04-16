"""Connect 4 environment integration tests."""

from dataclasses import dataclass

import pytest

from rlvr_games.core import (
    AgentVisibleEvent,
    ProjectedActionContext,
    PublicResetEvent,
    build_action_context,
)
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.connect4 import (
    BitBullySolver,
    Connect4Action,
    Connect4SolverAutoAdvancePolicy,
    Connect4State,
    FixedBoardScenario,
    RandomPositionScenario,
    TerminalOutcomeReward,
    make_connect4_env,
)
from tests.test_games.test_connect4.support import PRE_WIN_BOARD, X_WIN_BOARD


def make_reward() -> TerminalOutcomeReward:
    """Return a sparse terminal reward used by env tests."""
    return TerminalOutcomeReward(
        perspective="mover",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    )


def make_empty_start_env() -> TurnBasedEnv[Connect4State, Connect4Action]:
    """Return a standard Connect 4 environment from the empty board."""
    return make_connect4_env(
        scenario=RandomPositionScenario(
            rows=6,
            columns=7,
            connect_length=4,
            min_start_moves=0,
            max_start_moves=0,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )


@dataclass(slots=True)
class Connect4OpeningProjector:
    """Return one fixed opening event for factory wiring tests."""

    def project_action_context(
        self,
        *,
        state: Connect4State,
        reset_events: tuple[PublicResetEvent, ...],
    ) -> ProjectedActionContext:
        """Project one simple opening event."""
        del state
        del reset_events
        return ProjectedActionContext(
            opening_events=(
                AgentVisibleEvent(
                    kind="opening_event",
                    source="setup",
                    text="factory wired projector",
                ),
            ),
        )


def test_horizontal_win_sequence_terminates_with_reward_and_metadata() -> None:
    env = make_connect4_env(
        scenario=FixedBoardScenario(
            initial_board=PRE_WIN_BOARD,
            connect_length=4,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )
    env.reset(seed=1)

    result = env.step("4")

    assert result.accepted is True
    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["player"] == "x"
    assert result.info["column"] == 4
    assert result.info["winner"] == "x"
    assert result.info["termination"] == "connect_length"
    assert env.state.board[5][:4] == ("x", "x", "x", "x")
    assert result.observation.metadata["is_terminal"] is True
    assert result.observation.metadata["winner"] == "x"


def test_terminal_reset_marks_episode_finished_and_rejects_steps() -> None:
    env = make_connect4_env(
        scenario=FixedBoardScenario(
            initial_board=X_WIN_BOARD,
            connect_length=4,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )

    observation, _ = env.reset(seed=23)

    assert env.episode_finished is True
    assert observation.metadata["is_terminal"] is True
    assert observation.metadata["winner"] == "x"

    with pytest.raises(EpisodeFinishedError):
        env.step("1")


def test_env_records_trajectory_with_real_backend() -> None:
    env = make_empty_start_env()
    observation, info = env.reset(seed=123)

    result = env.step("1")

    assert info["scenario"] == "random_position"
    assert info["applied_start_moves"] == 0
    assert "Connect 4 board:" in (observation.text or "")
    assert result.reward == 0.0
    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].accepted is True
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.label == "1"
    assert env.trajectory.steps[0].info["player"] == "x"


def test_factory_forwards_agent_context_projector() -> None:
    env = make_connect4_env(
        scenario=RandomPositionScenario(
            rows=6,
            columns=7,
            connect_length=4,
            min_start_moves=0,
            max_start_moves=0,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
        agent_context_projector=Connect4OpeningProjector(),
    )
    env.reset(seed=3)

    context = build_action_context(env=env)

    assert context.opening_events == (
        AgentVisibleEvent(
            kind="opening_event",
            source="setup",
            text="factory wired projector",
        ),
    )


def test_solver_auto_advance_returns_to_agent_turn_and_records_reply() -> None:
    env = make_connect4_env(
        scenario=RandomPositionScenario(
            rows=6,
            columns=7,
            connect_length=4,
            min_start_moves=0,
            max_start_moves=0,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
        auto_advance_policy=Connect4SolverAutoAdvancePolicy(
            move_selector=BitBullySolver(),
        ),
    )
    env.reset(seed=31)

    result = env.step("4")

    assert result.accepted is True
    assert result.terminated is False
    assert result.info["transition_count_delta"] == 2
    assert result.info["auto_advanced"] is True
    assert env.state.current_player == "x"
    assert env.state.board[5][3] == "x"
    assert env.state.board[4][3] == "o"
    assert result.observation.metadata["current_player"] == "x"
    assert len(env.trajectory.steps[0].transitions) == 2
    assert env.trajectory.steps[0].transitions[0].source == "agent"
    assert env.trajectory.steps[0].transitions[1].source == "opponent"
    assert env.trajectory.steps[0].transitions[1].raw_action == "4"


def test_solver_auto_advance_can_play_multiple_turns_inside_the_env() -> None:
    env = make_connect4_env(
        scenario=RandomPositionScenario(
            rows=6,
            columns=7,
            connect_length=4,
            min_start_moves=0,
            max_start_moves=0,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
        auto_advance_policy=Connect4SolverAutoAdvancePolicy(
            move_selector=BitBullySolver(),
        ),
    )
    env.reset(seed=31)

    first_result = env.step("4")
    second_result = env.step("3")
    third_result = env.step("5")

    assert first_result.info["transition_count_delta"] == 2
    assert second_result.info["transition_count_delta"] == 2
    assert third_result.info["transition_count_delta"] == 2
    assert first_result.info["auto_advanced"] is True
    assert second_result.info["auto_advanced"] is True
    assert third_result.info["auto_advanced"] is True
    assert env.state.current_player == "x"
    assert env.state.board == (
        (".", ".", ".", ".", ".", ".", "."),
        (".", ".", ".", ".", ".", ".", "."),
        (".", ".", ".", ".", ".", ".", "."),
        (".", ".", ".", "o", ".", ".", "."),
        (".", ".", ".", "o", "x", ".", "."),
        (".", ".", "x", "x", "o", ".", "."),
    )
    assert len(env.trajectory.steps) == 3
    assert [
        tuple(transition.raw_action for transition in step.transitions)
        for step in env.trajectory.steps
    ] == [
        ("4", "4"),
        ("3", "5"),
        ("5", "4"),
    ]
