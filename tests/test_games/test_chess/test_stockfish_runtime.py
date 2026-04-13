"""Stockfish runtime resolution tests."""

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rlvr_games.games.chess.stockfish_runtime import resolve_stockfish_binary_path


def test_resolve_stockfish_binary_path_uses_env_var(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "stockfish"
    binary_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("RLVR_GAMES_STOCKFISH_PATH", str(binary_path))

    resolved_path = resolve_stockfish_binary_path()

    assert resolved_path == binary_path.resolve()


def test_resolve_stockfish_binary_path_uses_repo_local_install(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "stockfish"
    binary_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("RLVR_GAMES_STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(
        "rlvr_games.games.chess.stockfish_runtime.repo_local_stockfish_binary_path",
        lambda: binary_path,
    )
    monkeypatch.setattr(
        "rlvr_games.games.chess.stockfish_runtime.shutil.which",
        lambda name: None,
    )

    resolved_path = resolve_stockfish_binary_path()

    assert resolved_path == binary_path.resolve()


def test_resolve_stockfish_binary_path_raises_when_no_binary_is_available(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RLVR_GAMES_STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(
        "rlvr_games.games.chess.stockfish_runtime.repo_local_stockfish_binary_path",
        lambda: tmp_path / "missing-stockfish",
    )
    monkeypatch.setattr(
        "rlvr_games.games.chess.stockfish_runtime.shutil.which",
        lambda name: None,
    )

    with pytest.raises(FileNotFoundError):
        resolve_stockfish_binary_path()
