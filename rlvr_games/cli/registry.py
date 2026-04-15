"""Registered CLI specifications for play sessions."""

from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.games.chess.cli import CHESS_CLI_SPEC
from rlvr_games.games.connect4.cli import CONNECT4_CLI_SPEC
from rlvr_games.games.game2048.cli import GAME2048_CLI_SPEC
from rlvr_games.games.minesweeper.cli import MINESWEEPER_CLI_SPEC

PLAY_GAME_SPECS: tuple[GameCliSpec, ...] = (
    CHESS_CLI_SPEC,
    CONNECT4_CLI_SPEC,
    GAME2048_CLI_SPEC,
    MINESWEEPER_CLI_SPEC,
)
