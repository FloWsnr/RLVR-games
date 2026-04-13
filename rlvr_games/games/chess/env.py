"""Minimal chess environment wiring."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.protocol import (
    GameBackend,
    Renderer,
    RewardFn,
    Scenario,
    StateInspector,
)
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.state import ChessState


class ChessEnv(TurnBasedEnv[ChessState, ChessAction]):
    """Concrete turn-based environment for chess."""

    def __init__(
        self,
        *,
        backend: GameBackend[ChessState, ChessAction],
        scenario: Scenario[ChessState],
        renderer: Renderer[ChessState],
        state_inspector: StateInspector[ChessState],
        reward_fn: RewardFn[ChessState, ChessAction],
        config: EpisodeConfig,
    ) -> None:
        """Initialize a chess environment with explicit collaborators.

        Parameters
        ----------
        backend : GameBackend[ChessState, ChessAction]
            Rules backend used for legality checks and state transitions.
        scenario : Scenario[ChessState]
            Scenario used to generate initial positions.
        renderer : Renderer[ChessState]
            Renderer that converts chess state into observations.
        state_inspector : StateInspector[ChessState]
            Inspector that converts canonical state into a debug view.
        reward_fn : RewardFn[ChessState, ChessAction]
            Reward function used for verified transitions.
        config : EpisodeConfig
            Episode configuration such as turn limits and metadata.
        """
        super().__init__(
            backend=backend,
            scenario=scenario,
            renderer=renderer,
            state_inspector=state_inspector,
            reward_fn=reward_fn,
            config=config,
        )
