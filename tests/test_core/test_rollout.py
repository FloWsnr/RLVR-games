"""Rollout runner tests."""

from rlvr_games.core import Observation
from rlvr_games.core.rollout import ActionContext, run_episode
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN


class ScriptedAgent:
    """Deterministic agent used to test rollout execution."""

    def __init__(self, actions: list[str]) -> None:
        """Initialize the agent with a finite move script.

        Parameters
        ----------
        actions : list[str]
            Ordered raw actions to emit during the rollout.
        """
        self._actions = list(actions)
        self.contexts: list[ActionContext] = []

    def act(self, observation: Observation, context: ActionContext) -> str:
        """Return the next scripted action and record the action context.

        Parameters
        ----------
        observation : Observation
            Unused observation parameter required by the rollout protocol.
        context : ActionContext
            Context captured for later assertions.

        Returns
        -------
        str
            Next scripted action string.
        """
        del observation
        self.contexts.append(context)
        if not self._actions:
            raise AssertionError("ScriptedAgent ran out of actions.")
        return self._actions.pop(0)


def test_run_episode_records_scripted_checkmate_trajectory() -> None:
    env = make_chess_env(
        initial_fen=STANDARD_START_FEN,
        max_turns=None,
        text_renderer_kind=ChessTextRendererKind.ASCII,
        image_output_dir=None,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
    agent = ScriptedAgent(["f2f3", "e7e5", "g2g4", "d8h4"])

    result = run_episode(env=env, agent=agent, seed=11)

    assert result.terminated is True
    assert result.truncated is False
    assert result.turn_count == 4
    assert result.trajectory.reset_info["seed"] == 11
    assert len(result.trajectory.steps) == 4
    assert result.trajectory.steps[-1].info["termination"] == "checkmate"
    assert result.trajectory.steps[-1].info["winner"] == "black"
    assert agent.contexts[0].turn_index == 0
    assert "f2f3" in agent.contexts[0].legal_actions


def test_run_episode_passes_sorted_legal_actions_in_context() -> None:
    env = make_chess_env(
        initial_fen=STANDARD_START_FEN,
        max_turns=1,
        text_renderer_kind=ChessTextRendererKind.ASCII,
        image_output_dir=None,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
    agent = ScriptedAgent(["e2e4"])

    result = run_episode(env=env, agent=agent, seed=3)

    assert result.terminated is False
    assert result.truncated is True
    assert result.turn_count == 1
    assert agent.contexts[0].legal_actions == tuple(
        sorted(agent.contexts[0].legal_actions)
    )
    assert len(agent.contexts[0].legal_actions) == 20
    assert result.trajectory.steps[0].accepted is True
    assert result.trajectory.steps[0].action is not None
    assert result.trajectory.steps[0].action.uci == "e2e4"
