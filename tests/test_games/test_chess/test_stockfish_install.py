"""Stockfish installation helper tests."""

from rlvr_games.games.chess.stockfish_install import (
    StockfishAsset,
    StockfishRelease,
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
