"""Registered per-game CLI specifications."""

from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.games.chess.cli import CHESS_CLI_SPEC
from rlvr_games.games.game2048.cli import GAME2048_CLI_SPEC

PLAY_GAME_SPECS: tuple[GameCliSpec, ...] = (
    CHESS_CLI_SPEC,
    GAME2048_CLI_SPEC,
)
