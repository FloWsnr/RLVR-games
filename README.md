# RLVR-games
Teach LLMs to reason by playing games

## Interactive Play

Run an interactive chess session with:

```bash
uv run rlvr-games play chess --seed 0
```

You can also start from a custom FEN or switch the text renderer:

```bash
uv run rlvr-games play chess --seed 0 --fen "<fen>"
uv run rlvr-games play chess --seed 0 --renderer unicode
```

Inside the session, enter raw UCI moves such as `e2e4`. The commands `help`,
`legal`, `fen`, `trajectory`, and `quit` are also available.

## Programmatic Rollouts

Use the library in-process for training and evaluation loops:

```python
from rlvr_games.core.rollout import run_episode
from rlvr_games.games.chess import (
    ChessImageOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN

env = make_chess_env(
    initial_fen=STANDARD_START_FEN,
    max_turns=None,
    text_renderer_kind=ChessTextRendererKind.ASCII,
    image_output_dir=None,
    image_size=360,
    image_coordinates=True,
    image_orientation=ChessImageOrientation.WHITE,
)
```

## Development

Run the test suite with:

```bash
uv run pytest
```
