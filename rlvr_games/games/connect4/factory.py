"""Factory helpers for constructing Connect 4 environments."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.protocol import AutoAdvancePolicy, RewardFn, Scenario
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.backend import Connect4Backend
from rlvr_games.games.connect4.render import (
    Connect4AsciiBoardFormatter,
    Connect4ImageRenderer,
    Connect4ObservationRenderer,
)
from rlvr_games.games.connect4.state import Connect4State, inspect_connect4_state


def make_connect4_env(
    *,
    scenario: Scenario[Connect4State],
    reward_fn: RewardFn[Connect4State, Connect4Action],
    config: EpisodeConfig,
    include_images: bool,
    image_size: int,
    auto_advance_policy: AutoAdvancePolicy[Connect4State, Connect4Action] | None = None,
) -> TurnBasedEnv[Connect4State, Connect4Action]:
    """Construct a fully wired Connect 4 environment.

    Parameters
    ----------
    scenario : Scenario[Connect4State]
        Scenario used to create the initial Connect 4 position.
    reward_fn : RewardFn[Connect4State, Connect4Action]
        Reward function used to score verified transitions.
    config : EpisodeConfig
        Episode execution configuration controlling invalid-action handling and
        optional attempt or transition limits.
    include_images : bool
        Whether observations should include rendered board images.
    image_size : int
        Raster image size in pixels. Ignored when `include_images` is `False`.
    auto_advance_policy : AutoAdvancePolicy[Connect4State, Connect4Action] | None
        Optional policy that auto-applies internal Connect 4 transitions
        between agent turns.

    Returns
    -------
    TurnBasedEnv[Connect4State, Connect4Action]
        Connect 4 environment wired with the standard backend, scenario,
        renderer, and supplied reward function.
    """
    image_renderer: Connect4ImageRenderer | None = None
    if include_images:
        image_renderer = Connect4ImageRenderer(size=image_size)

    return TurnBasedEnv(
        backend=Connect4Backend(),
        scenario=scenario,
        renderer=Connect4ObservationRenderer(
            board_formatter=Connect4AsciiBoardFormatter(),
            image_renderer=image_renderer,
        ),
        inspect_canonical_state_fn=inspect_connect4_state,
        reward_fn=reward_fn,
        config=config,
        auto_advance_policy=auto_advance_policy,
    )
