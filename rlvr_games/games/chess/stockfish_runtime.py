"""Stockfish runtime resolution and evaluator helpers."""

import os
from pathlib import Path
import shutil

from rlvr_games.games.chess.rewards import UciEngineEvaluator

STOCKFISH_PATH_ENV_VAR = "RLVR_GAMES_STOCKFISH_PATH"


class StockfishEvaluator(UciEngineEvaluator):
    """Concrete Stockfish-backed evaluator with binary resolution helpers."""

    @classmethod
    def from_engine_path(
        cls,
        *,
        engine_path: Path,
        depth: int,
        mate_score: int,
    ) -> "StockfishEvaluator":
        """Construct an evaluator from one explicit Stockfish binary path.

        Parameters
        ----------
        engine_path : Path
            Filesystem path to the Stockfish binary.
        depth : int
            Fixed analysis depth passed to the engine for each evaluation.
        mate_score : int
            Centipawn-equivalent scalar used to map mating lines to finite
            values.

        Returns
        -------
        StockfishEvaluator
            Evaluator configured to talk to the supplied Stockfish binary.
        """
        return cls(
            engine_path=validate_stockfish_binary_path(engine_path=engine_path),
            depth=depth,
            mate_score=mate_score,
        )

    @classmethod
    def from_installed_binary(
        cls,
        *,
        depth: int,
        mate_score: int,
    ) -> "StockfishEvaluator":
        """Construct an evaluator from the configured local Stockfish install.

        Parameters
        ----------
        depth : int
            Fixed analysis depth passed to the engine for each evaluation.
        mate_score : int
            Centipawn-equivalent scalar used to map mating lines to finite
            values.

        Returns
        -------
        StockfishEvaluator
            Evaluator configured to use the resolved Stockfish binary.
        """
        return cls(
            engine_path=resolve_stockfish_binary_path(),
            depth=depth,
            mate_score=mate_score,
        )


def repo_local_stockfish_root_dir() -> Path:
    """Return the repo-local directory reserved for Stockfish installs.

    Returns
    -------
    Path
        Directory stored next to the chess game implementation.
    """
    return Path(__file__).resolve().parent / ".stockfish"


def repo_local_stockfish_install_dir() -> Path:
    """Return the active repo-local Stockfish installation directory.

    Returns
    -------
    Path
        Directory expected to contain the active Stockfish binary and manifest.
    """
    return repo_local_stockfish_root_dir() / "current"


def stockfish_binary_filename() -> str:
    """Return the Stockfish executable filename for the current host.

    Returns
    -------
    str
        Expected filename of the Stockfish binary on the current platform.
    """
    if os.name == "nt":
        return "stockfish.exe"
    return "stockfish"


def repo_local_stockfish_binary_path() -> Path:
    """Return the expected repo-local Stockfish binary path.

    Returns
    -------
    Path
        Full path to the active repo-local Stockfish binary.
    """
    return repo_local_stockfish_install_dir() / stockfish_binary_filename()


def validate_stockfish_binary_path(*, engine_path: Path) -> Path:
    """Validate that an explicit Stockfish binary path exists.

    Parameters
    ----------
    engine_path : Path
        Candidate Stockfish binary path to validate.

    Returns
    -------
    Path
        Normalized absolute path to the existing binary.

    Raises
    ------
    FileNotFoundError
        If ``engine_path`` does not exist.
    """
    normalized_path = engine_path.expanduser().resolve()
    if not normalized_path.exists():
        raise FileNotFoundError(f"Stockfish binary does not exist: {normalized_path}")
    return normalized_path


def resolve_stockfish_binary_path() -> Path:
    """Resolve a usable Stockfish binary path for the current host.

    Resolution order is:

    1. ``RLVR_GAMES_STOCKFISH_PATH``
    2. the repo-local install managed by ``rlvr-games-install-stockfish``
    3. ``stockfish`` on ``PATH``

    Returns
    -------
    Path
        Normalized absolute path to a usable Stockfish binary.

    Raises
    ------
    FileNotFoundError
        If no Stockfish binary can be resolved.
    """
    stockfish_path_from_env = os.environ.get(STOCKFISH_PATH_ENV_VAR)
    if stockfish_path_from_env is not None:
        return validate_stockfish_binary_path(engine_path=Path(stockfish_path_from_env))

    repo_local_binary_path = repo_local_stockfish_binary_path()
    if repo_local_binary_path.exists():
        return repo_local_binary_path.resolve()

    stockfish_path_from_path = shutil.which(stockfish_binary_filename())
    if stockfish_path_from_path is not None:
        return Path(stockfish_path_from_path).resolve()

    raise FileNotFoundError(
        "Stockfish binary not found. Run `uv run rlvr-games-install-stockfish`, "
        f"set {STOCKFISH_PATH_ENV_VAR}, or pass --stockfish-path."
    )


__all__ = [
    "STOCKFISH_PATH_ENV_VAR",
    "StockfishEvaluator",
    "repo_local_stockfish_binary_path",
    "repo_local_stockfish_install_dir",
    "repo_local_stockfish_root_dir",
    "resolve_stockfish_binary_path",
    "stockfish_binary_filename",
    "validate_stockfish_binary_path",
]
