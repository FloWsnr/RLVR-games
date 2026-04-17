"""Factory helpers for constructing 2048 environments."""

from typing import Sequence

from rlvr_games.core.action_context import AgentContextProjector
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.messages import (
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
)
from rlvr_games.core.protocol import RewardFn
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.game2048.actions import Game2048Action
from rlvr_games.games.game2048.backend import Game2048Backend
from rlvr_games.games.game2048.chance import Game2048ChanceModel
from rlvr_games.games.game2048.reset_events import Game2048StartTilePolicy
from rlvr_games.games.game2048.render import (
    Game2048AsciiBoardFormatter,
    Game2048ImageRenderer,
    Game2048ObservationRenderer,
)
from rlvr_games.games.game2048.scenarios import (
    FixedBoardScenario,
    RandomStartScenario,
    normalize_initial_board,
)
from rlvr_games.games.game2048.state import Game2048State, inspect_game2048_state


def _default_game2048_message_adapter() -> DefaultObservationMessageAdapter:
    """Return the default 2048 observation message adapter."""
    return DefaultObservationMessageAdapter(
        policy=DefaultObservationMessagePolicy(
            action_reminder_text=(
                "Respond with one move: `up`, `right`, `down`, or `left`."
            ),
        )
    )


def make_game2048_env(
    *,
    size: int,
    target_value: int,
    initial_board: Sequence[Sequence[int]] | None,
    initial_score: int,
    initial_move_count: int,
    start_tile_count: int = 2,
    reward_fn: RewardFn[Game2048State, Game2048Action],
    config: EpisodeConfig,
    include_images: bool,
    image_size: int,
    agent_context_projector: AgentContextProjector[Game2048State] | None = None,
) -> TurnBasedEnv[Game2048State, Game2048Action]:
    """Construct a fully wired 2048 environment.

    Parameters
    ----------
    size : int
        Board size used when `initial_board` is not supplied.
    target_value : int
        Tile value that counts as a win.
    initial_board : Sequence[Sequence[int]] | None
        Explicit initial board to use. When `None`, the environment starts from
        a seeded random board with two spawned tiles.
    initial_score : int
        Starting score used only when `initial_board` is supplied.
    initial_move_count : int
        Starting move count used only when `initial_board` is supplied.
    start_tile_count : int
        Number of tiles to spawn at reset when `initial_board` is not
        supplied. Ignored for fixed-board starts.
    reward_fn : RewardFn[Game2048State, Game2048Action]
        Reward function used to score verified transitions.
    config : EpisodeConfig
        Episode execution configuration controlling invalid-action handling and
        optional attempt or transition limits.
    include_images : bool
        Whether observations should include rendered board images.
    image_size : int
        Raster image size in pixels. Ignored when `include_images` is `False`.
    agent_context_projector : AgentContextProjector[Game2048State] | None
        Optional projector that adds structured agent-visible context such as
        opening events.

    Returns
    -------
    TurnBasedEnv[Game2048State, Game2048Action]
        2048 environment wired with the standard backend, scenario, renderer,
        and supplied reward function.
    """
    chance_model = Game2048ChanceModel()
    backend = Game2048Backend(chance_model=chance_model)
    reset_event_policy: Game2048StartTilePolicy | None = None
    if initial_board is None:
        scenario = RandomStartScenario(
            size=size,
            target_value=target_value,
            start_tile_count=start_tile_count,
            chance_model=chance_model,
        )
        reset_event_policy = Game2048StartTilePolicy(
            backend=backend,
            start_tile_count=start_tile_count,
        )
    else:
        normalized_board = normalize_initial_board(board=initial_board)
        if len(normalized_board) != size:
            raise ValueError(
                f"Configured 2048 board size {size} does not match initial board size "
                f"{len(normalized_board)}."
            )
        scenario = FixedBoardScenario(
            initial_board=normalized_board,
            initial_score=initial_score,
            initial_move_count=initial_move_count,
            target_value=target_value,
            chance_model=chance_model,
        )

    image_renderer: Game2048ImageRenderer | None = None
    if include_images:
        image_renderer = Game2048ImageRenderer(size=image_size)

    return TurnBasedEnv(
        backend=backend,
        scenario=scenario,
        renderer=Game2048ObservationRenderer(
            board_formatter=Game2048AsciiBoardFormatter(),
            image_renderer=image_renderer,
        ),
        inspect_canonical_state_fn=inspect_game2048_state,
        reward_fn=reward_fn,
        config=config,
        agent_context_projector=agent_context_projector,
        observation_message_adapter=_default_game2048_message_adapter(),
        reset_event_policy=reset_event_policy,
    )
