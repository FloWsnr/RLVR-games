"""Stockfish installer CLI tests."""

from io import StringIO
from pathlib import Path
import sys

from _pytest.monkeypatch import MonkeyPatch

from rlvr_games.games.chess.stockfish_install import run_install_cli


def test_run_install_cli_prints_installed_binary_path(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_stream = StringIO()
    binary_path = tmp_path / "stockfish"
    observed_arguments: dict[str, object] = {}
    monkeypatch.setattr(sys, "stdout", output_stream)

    def stub_install_latest_stockfish(*, install_dir: Path, force: bool) -> Path:
        """Capture installer arguments and return a synthetic binary path."""
        observed_arguments["install_dir"] = install_dir
        observed_arguments["force"] = force
        return binary_path

    monkeypatch.setattr(
        "rlvr_games.games.chess.stockfish_install.install_latest_stockfish",
        stub_install_latest_stockfish,
    )

    exit_code = run_install_cli(["--install-dir", str(tmp_path), "--force"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert observed_arguments == {"install_dir": tmp_path, "force": True}
    assert f"Installed Stockfish binary: {binary_path}" in output
