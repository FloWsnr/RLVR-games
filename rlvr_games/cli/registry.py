"""Registered CLI specifications for play sessions and dataset builders."""

from rlvr_games.cli.specs import DatasetCliSpec, GameCliSpec
from rlvr_games.games.chess.dataset_cli import CHESS_LICHESS_PUZZLES_DATASET_SPEC
from rlvr_games.games.chess.cli import CHESS_CLI_SPEC
from rlvr_games.games.game2048.cli import GAME2048_CLI_SPEC

PLAY_GAME_SPECS: tuple[GameCliSpec, ...] = (
    CHESS_CLI_SPEC,
    GAME2048_CLI_SPEC,
)

DATASET_SPECS: tuple[DatasetCliSpec, ...] = (CHESS_LICHESS_PUZZLES_DATASET_SPEC,)
