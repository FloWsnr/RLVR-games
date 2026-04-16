"""Integration tests for the checked-in Lichess puzzle fixture subset."""

from pathlib import Path

from rlvr_games.datasets import DatasetSplit
from rlvr_games.games.chess.scenarios import ChessPuzzleDatasetScenario


def fixture_manifest_path() -> Path:
    """Return the checked-in processed Lichess puzzle subset manifest.

    Returns
    -------
    Path
        Path to the fixture manifest.
    """
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "lichess_puzzles_subset"
        / "manifest.json"
    )


def test_checked_in_lichess_fixture_loads_via_puzzle_scenario() -> None:
    """Scenario loading should work against the checked-in processed subset."""
    scenario = ChessPuzzleDatasetScenario(
        manifest_path=fixture_manifest_path(),
        split=DatasetSplit.TRAIN,
    )

    reset = scenario.reset(seed=0)
    state = reset.initial_state
    info = reset.reset_info

    assert info == {
        "scenario": "dataset_puzzle",
        "dataset": "lichess-puzzles",
        "split": "train",
        "record_id": "0000D",
        "initial_fen": "5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 3 27",
        "rating": 1414,
        "themes": ("advantage", "endgame", "short"),
        "seed": 0,
    }
    assert state.fen == "5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 3 27"
    assert state.metadata["record_id"] == "0000D"
    assert state.metadata["task_type"] == "puzzle"
    assert state.metadata["solution_moves_uci"] == ("f8d8", "d6d8", "f6d8")
