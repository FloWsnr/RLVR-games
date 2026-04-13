# RLVR-games
Teach LLMs to reason by playing games

## Install

Sync the Python environment and install a local Stockfish binary for chess
engine-backed rewards:

```bash
uv sync
uv run rlvr-games-install-stockfish
```

The installer downloads the latest official Stockfish release from
https://stockfishchess.org/download/ and places the active binary under
`rlvr_games/games/chess/.stockfish/current/`. If you want to use a different
binary, set `RLVR_GAMES_STOCKFISH_PATH` or pass `--stockfish-path` to the chess
CLI.

## Interactive Play

Run an interactive chess or 2048 session with:

```bash
uv run rlvr-games play chess --seed 0 --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000
uv run rlvr-games play 2048 --seed 0
```

You can also start from a custom FEN, switch the text renderer, or flip the
board orientation for both text and image observations. When
`--image-output-dir` is set, the CLI saves the in-memory rendered images to
disk for inspection:

```bash
uv run rlvr-games play chess --seed 0 --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000 --fen "<fen>"
uv run rlvr-games play chess --seed 0 --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000 --renderer unicode
uv run rlvr-games play chess --seed 0 --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000 --orientation black
uv run rlvr-games play chess --seed 0 --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000 --image-output-dir ./renders
```

You can also play from processed chess puzzle datasets:

```bash
uv run rlvr-games datasets download chess lichess-puzzles
uv run rlvr-games datasets preprocess chess lichess-puzzles
uv run rlvr-games play chess --scenario lichess-puzzles --dataset-manifest data/processed/chess/lichess-puzzles/<version>/manifest.json --reward puzzle-sparse --seed 0
```

For 2048, you can also start from an explicit board:

```bash
uv run rlvr-games play 2048 --seed 0 --board "2,0,0,0/0,2,0,0/0,0,0,0/0,0,0,0"
```

Inside chess sessions, enter raw UCI moves such as `e2e4`. Inside 2048
sessions, enter directions such as `up`, `right`, `down`, or `left`. The
commands `help`, `legal`, `state`, `show fen`, `trajectory`, and `quit` are
also available; `show fen` is only meaningful for chess.

By default, invalid actions raise verifier errors without changing state. You
can make them explicit rollout events instead:

```bash
uv run rlvr-games play chess --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000 --invalid-action-policy penalize-continue --invalid-action-penalty -1
uv run rlvr-games play chess --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000 --invalid-action-policy penalize-truncate --invalid-action-penalty -1
```

You can also expose Stockfish evaluations directly as rewards:

```bash
uv run rlvr-games play chess \
  --seed 0 \
  --reward engine-eval-dense \
  --engine-depth 12 \
  --engine-mate-score 100000
```

Use `--reward engine-eval-sparse` when you only want evaluator reward on
terminal states.

## Programmatic Rollouts

Use the library in-process for training and evaluation loops:

```python
from rlvr_games.core.rollout import run_episode
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    TerminalOutcomeReward,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN, StartingPositionScenario

env = make_chess_env(
    scenario=StartingPositionScenario(initial_fen=STANDARD_START_FEN),
    reward_fn=TerminalOutcomeReward(
        perspective="white",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    ),
    config=EpisodeConfig(),
    text_renderer_kind=ChessTextRendererKind.ASCII,
    include_images=True,
    image_size=360,
    image_coordinates=True,
    orientation=ChessBoardOrientation.WHITE,
)
```

Rendered observations now carry in-memory images in `observation.images`, which
is suitable for multimodal training loops without forcing a filesystem round
trip.

To shape chess play directly with Stockfish evaluations, construct a
Stockfish-backed evaluator once and reuse it across episodes:

```python
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    EngineEvalDenseReward,
    StockfishEvaluator,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN, StartingPositionScenario

with StockfishEvaluator.from_installed_binary(
    depth=12,
    mate_score=100000,
) as evaluator:
    env = make_chess_env(
        scenario=StartingPositionScenario(initial_fen=STANDARD_START_FEN),
        reward_fn=EngineEvalDenseReward(
            evaluator=evaluator,
            perspective="mover",
        ),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
```

If you need a non-default binary location, use
`StockfishEvaluator.from_engine_path(...)`.

For 2048, a sparse reward policy can be passed explicitly: `0.0` until the
target tile is reached, then `float(target_value)` on the winning move:

```python
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.game2048 import TargetTileReward, make_game2048_env

env = make_game2048_env(
    size=4,
    target_value=2048,
    initial_board=None,
    initial_score=0,
    initial_move_count=0,
    reward_fn=TargetTileReward(),
    config=EpisodeConfig(),
    include_images=True,
    image_size=360,
)
```

The dense score-delta reward remains available as
`rlvr_games.games.game2048.ScoreDeltaReward` when you want shaping from merge
values instead.

To make invalid actions explicit in training or evaluation loops, configure the
base environment directly:

```python
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    TerminalOutcomeReward,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN, StartingPositionScenario

env = make_chess_env(
    scenario=StartingPositionScenario(initial_fen=STANDARD_START_FEN),
    reward_fn=TerminalOutcomeReward(
        perspective="white",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    ),
    config=EpisodeConfig(
        invalid_action_policy=InvalidActionPolicy(
            mode=InvalidActionMode.PENALIZE_CONTINUE,
            penalty=-1.0,
        ),
    ),
    text_renderer_kind=ChessTextRendererKind.ASCII,
    include_images=False,
    image_size=360,
    image_coordinates=True,
    orientation=ChessBoardOrientation.WHITE,
)
```

## Development

Run the test suite with:

```bash
uv run pytest
```
