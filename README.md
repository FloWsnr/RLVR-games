# RLVR-games
Teach LLMs to reason by playing games

## Interactive Play

Run an interactive chess session with:

```bash
uv run rlvr-games play chess --seed 0
```

You can also start from a custom FEN, switch the text renderer, or flip the
board orientation for both text and image observations:

```bash
uv run rlvr-games play chess --seed 0 --fen "<fen>"
uv run rlvr-games play chess --seed 0 --renderer unicode
uv run rlvr-games play chess --seed 0 --orientation black
```

Inside the session, enter raw UCI moves such as `e2e4`. The commands `help`,
`legal`, `fen`, `trajectory`, and `quit` are also available.

By default, invalid actions raise verifier errors without changing state. You
can make them explicit rollout events instead:

```bash
uv run rlvr-games play chess --invalid-action-policy penalize-continue --invalid-action-penalty -1
uv run rlvr-games play chess --invalid-action-policy penalize-truncate --invalid-action-penalty -1
```

## Programmatic Rollouts

Use the library in-process for training and evaluation loops:

```python
from rlvr_games.core.rollout import run_episode
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN

env = make_chess_env(
    initial_fen=STANDARD_START_FEN,
    config=EpisodeConfig(),
    text_renderer_kind=ChessTextRendererKind.ASCII,
    image_output_dir=None,
    image_size=360,
    image_coordinates=True,
    orientation=ChessBoardOrientation.WHITE,
)
```

To make invalid actions explicit in training or evaluation loops, configure the
base environment directly:

```python
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)

env = make_chess_env(
    initial_fen=STANDARD_START_FEN,
    config=EpisodeConfig(
        invalid_action_policy=InvalidActionPolicy(
            mode=InvalidActionMode.PENALIZE_CONTINUE,
            penalty=-1.0,
        ),
    ),
    text_renderer_kind=ChessTextRendererKind.ASCII,
    image_output_dir=None,
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
