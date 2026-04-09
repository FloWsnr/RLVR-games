"""Minimal chess environment wiring."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.backend import ChessBackend
from rlvr_games.games.chess.render import ChessTextRenderer
from rlvr_games.games.chess.scenarios import StartingPositionScenario
from rlvr_games.games.chess.state import ChessState


class ChessEnv(TurnBasedEnv[ChessState, ChessAction]):
    """Minimal chess env.

    Reset works today. `step()` is intentionally blocked until the backend is
    connected to an executable chess rule engine.
    """

    def __init__(
        self,
        *,
        backend: ChessBackend | None = None,
        scenario: StartingPositionScenario | None = None,
        renderer: ChessTextRenderer | None = None,
        config: EpisodeConfig | None = None,
    ) -> None:
        super().__init__(
            backend=backend or ChessBackend(),
            scenario=scenario or StartingPositionScenario(),
            renderer=renderer or ChessTextRenderer(),
            reward_fn=ZeroReward(),
            config=config or EpisodeConfig(),
        )
