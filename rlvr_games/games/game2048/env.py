"""Minimal 2048 environment wiring."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.protocol import GameBackend, Renderer, RewardFn, Scenario
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.game2048.actions import Game2048Action
from rlvr_games.games.game2048.state import Game2048State


class Game2048Env(TurnBasedEnv[Game2048State, Game2048Action]):
    """Concrete turn-based environment for 2048."""

    def __init__(
        self,
        *,
        backend: GameBackend[Game2048State, Game2048Action],
        scenario: Scenario[Game2048State],
        renderer: Renderer[Game2048State],
        reward_fn: RewardFn[Game2048State, Game2048Action],
        config: EpisodeConfig,
    ) -> None:
        """Initialize a 2048 environment with explicit collaborators.

        Parameters
        ----------
        backend : GameBackend[Game2048State, Game2048Action]
            Rules backend used for legality checks, spawns, and transitions.
        scenario : Scenario[Game2048State]
            Scenario used to generate initial states.
        renderer : Renderer[Game2048State]
            Renderer that converts canonical state into observations.
        reward_fn : RewardFn[Game2048State, Game2048Action]
            Reward function used for verified transitions.
        config : EpisodeConfig
            Episode configuration such as invalid-action handling and optional
            attempt or transition limits.
        """
        super().__init__(
            backend=backend,
            scenario=scenario,
            renderer=renderer,
            reward_fn=reward_fn,
            config=config,
        )
