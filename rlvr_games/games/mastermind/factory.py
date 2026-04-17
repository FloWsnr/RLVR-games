"""Factory helpers for constructing Mastermind environments."""

from rlvr_games.core.action_context import AgentContextProjector
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.messages import (
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
)
from rlvr_games.core.protocol import RewardFn, Scenario
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.mastermind.actions import MastermindAction
from rlvr_games.games.mastermind.backend import MastermindBackend
from rlvr_games.games.mastermind.render import (
    MastermindAsciiBoardFormatter,
    MastermindImageRenderer,
    MastermindObservationRenderer,
)
from rlvr_games.games.mastermind.state import MastermindState, inspect_mastermind_state


def _default_mastermind_message_adapter() -> DefaultObservationMessageAdapter:
    """Return the default Mastermind observation message adapter."""
    return DefaultObservationMessageAdapter(
        policy=DefaultObservationMessagePolicy(
            action_reminder_text=(
                "Respond with one guess in the form `guess 1 1 2 2`. Bare `1122` "
                "is also accepted. Each digit must be from `1` to `6`."
            ),
        )
    )


def make_mastermind_env(
    *,
    scenario: Scenario[MastermindState],
    reward_fn: RewardFn[MastermindState, MastermindAction],
    config: EpisodeConfig,
    include_images: bool,
    image_size: int,
    agent_context_projector: AgentContextProjector[MastermindState] | None = None,
) -> TurnBasedEnv[MastermindState, MastermindAction]:
    """Construct a fully wired Mastermind environment."""
    image_renderer: MastermindImageRenderer | None = None
    if include_images:
        image_renderer = MastermindImageRenderer(size=image_size)

    return TurnBasedEnv(
        backend=MastermindBackend(),
        scenario=scenario,
        renderer=MastermindObservationRenderer(
            board_formatter=MastermindAsciiBoardFormatter(),
            image_renderer=image_renderer,
        ),
        inspect_canonical_state_fn=inspect_mastermind_state,
        reward_fn=reward_fn,
        config=config,
        agent_context_projector=agent_context_projector,
        observation_message_adapter=_default_mastermind_message_adapter(),
    )
