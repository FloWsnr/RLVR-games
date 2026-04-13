"""Stockfish integration tests."""

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rlvr_games.games.chess.stockfish import (
    StockfishAsset,
    StockfishRelease,
    resolve_stockfish_binary_path,
    select_stockfish_asset_for_host,
)


def make_release(*, asset_names: tuple[str, ...]) -> StockfishRelease:
    """Construct synthetic Stockfish release metadata for tests.

    Parameters
    ----------
    asset_names : tuple[str, ...]
        Asset filenames that should appear in the synthetic release.

    Returns
    -------
    StockfishRelease
        Release metadata containing the requested assets.
    """
    return StockfishRelease(
        tag_name="sf_test",
        display_name="Stockfish Test",
        assets=tuple(
            StockfishAsset(
                name=asset_name,
                download_url=f"https://example.com/{asset_name}",
                sha256="0" * 64,
            )
            for asset_name in asset_names
        ),
    )


def test_select_stockfish_asset_for_linux_x86_64_prefers_avx2() -> None:
    release = make_release(
        asset_names=(
            "stockfish-ubuntu-x86-64.tar",
            "stockfish-ubuntu-x86-64-avx2.tar",
        )
    )

    asset = select_stockfish_asset_for_host(
        release=release,
        system="Linux",
        machine="x86_64",
        cpu_features=frozenset({"avx2"}),
    )

    assert asset.name == "stockfish-ubuntu-x86-64-avx2.tar"


def test_select_stockfish_asset_for_linux_x86_64_falls_back_to_generic() -> None:
    release = make_release(asset_names=("stockfish-ubuntu-x86-64.tar",))

    asset = select_stockfish_asset_for_host(
        release=release,
        system="Linux",
        machine="x86_64",
        cpu_features=frozenset(),
    )

    assert asset.name == "stockfish-ubuntu-x86-64.tar"


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
        "rlvr_games.games.chess.stockfish.repo_local_stockfish_binary_path",
        lambda: binary_path,
    )
    monkeypatch.setattr(
        "rlvr_games.games.chess.stockfish.shutil.which",
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
        "rlvr_games.games.chess.stockfish.repo_local_stockfish_binary_path",
        lambda: tmp_path / "missing-stockfish",
    )
    monkeypatch.setattr(
        "rlvr_games.games.chess.stockfish.shutil.which",
        lambda name: None,
    )

    with pytest.raises(FileNotFoundError):
        resolve_stockfish_binary_path()
