"""Factory helpers for constructing Yahtzee environments."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.messages import (
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
)
from rlvr_games.core.protocol import RewardFn
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.yahtzee.actions import YahtzeeAction
from rlvr_games.games.yahtzee.backend import YahtzeeBackend
from rlvr_games.games.yahtzee.chance import YahtzeeChanceModel
from rlvr_games.games.yahtzee.render import (
    YahtzeeDiceFormatter,
    YahtzeeImageRenderer,
    YahtzeeObservationRenderer,
    YahtzeeScorecardFormatter,
)
from rlvr_games.games.yahtzee.reset_events import YahtzeeOpeningRollPolicy
from rlvr_games.games.yahtzee.scenarios import FixedStateScenario, StandardGameScenario
from rlvr_games.games.yahtzee.state import YahtzeeState, inspect_yahtzee_state
from rlvr_games.games.yahtzee.turns import YahtzeeOpeningRollAutoAdvancePolicy


def _default_yahtzee_message_adapter() -> DefaultObservationMessageAdapter:
    """Return the default Yahtzee observation message adapter."""
    return DefaultObservationMessageAdapter(
        policy=DefaultObservationMessagePolicy(
            action_reminder_text=(
                "Respond with one Yahtzee action such as `reroll 1 3 5` "
                "or `score full-house`."
            ),
        )
    )


def make_yahtzee_env(
    *,
    initial_state: YahtzeeState | None,
    reward_fn: RewardFn[YahtzeeState, YahtzeeAction],
    config: EpisodeConfig,
    include_images: bool,
    image_size: int,
) -> TurnBasedEnv[YahtzeeState, YahtzeeAction]:
    """Construct a fully wired Yahtzee environment.

    Parameters
    ----------
    initial_state : YahtzeeState | None
        Explicit starting state to use. When `None`, the environment starts a
        standard new game and applies an opening roll at reset time.
    reward_fn : RewardFn[YahtzeeState, YahtzeeAction]
        Reward function used to score verified transitions.
    config : EpisodeConfig
        Episode execution configuration controlling invalid-action handling
        and optional attempt or transition limits.
    include_images : bool
        Whether observations should include rendered dice images.
    image_size : int
        Raster image size in pixels. Ignored when `include_images` is
        `False`.

    Returns
    -------
    TurnBasedEnv[YahtzeeState, YahtzeeAction]
        Yahtzee environment wired with the standard backend, renderer, reset
        policy, and supplied reward function.
    """
    chance_model = YahtzeeChanceModel()
    backend = YahtzeeBackend(chance_model=chance_model)

    if initial_state is None:
        scenario = StandardGameScenario(chance_model=chance_model)
        reset_event_policy: YahtzeeOpeningRollPolicy | None = YahtzeeOpeningRollPolicy(
            backend=backend
        )
    else:
        scenario = FixedStateScenario(initial_state=initial_state)
        reset_event_policy = None
        if initial_state.awaiting_roll:
            reset_event_policy = YahtzeeOpeningRollPolicy(backend=backend)

    image_renderer: YahtzeeImageRenderer | None = None
    if include_images:
        image_renderer = YahtzeeImageRenderer(size=image_size)

    return TurnBasedEnv(
        backend=backend,
        scenario=scenario,
        renderer=YahtzeeObservationRenderer(
            dice_formatter=YahtzeeDiceFormatter(),
            scorecard_formatter=YahtzeeScorecardFormatter(),
            image_renderer=image_renderer,
        ),
        inspect_canonical_state_fn=inspect_yahtzee_state,
        reward_fn=reward_fn,
        config=config,
        observation_message_adapter=_default_yahtzee_message_adapter(),
        reset_event_policy=reset_event_policy,
        auto_advance_policy=YahtzeeOpeningRollAutoAdvancePolicy(),
    )
