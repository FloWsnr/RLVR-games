"""Task-spec loading and config-smoke tests."""

from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
import sys

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rlvr_games.cli.common import add_common_play_arguments
from rlvr_games.cli.main import build_parser, run_cli
from rlvr_games.games.connect4.cli import (
    build_connect4_environment,
    register_connect4_arguments,
)
from rlvr_games.task_specs import (
    TASK_SPEC_SCHEMA_VERSION,
    build_environment_from_task_spec,
    load_environment_from_task_spec_path,
    load_task_spec,
)
from rlvr_games.core.types import Observation


def example_task_spec_paths() -> tuple[Path, ...]:
    """Return every checked-in example task-spec YAML path."""
    config_root = Path(__file__).resolve().parents[2] / "config" / "games"
    return tuple(sorted(config_root.rglob("*.yaml")))


@pytest.mark.parametrize("task_spec_path", example_task_spec_paths())
def test_example_task_specs_load(task_spec_path: Path) -> None:
    task_spec = load_task_spec(path=task_spec_path)

    assert task_spec.schema_version == TASK_SPEC_SCHEMA_VERSION
    assert task_spec.task_id
    assert task_spec.game in {"chess", "connect4", "game2048", "minesweeper", "yahtzee"}


@pytest.mark.parametrize("task_spec_path", example_task_spec_paths())
def test_example_task_specs_build_env_and_reset(task_spec_path: Path) -> None:
    env = load_environment_from_task_spec_path(path=task_spec_path)

    try:
        observation, info = env.reset(seed=0)
        assert isinstance(observation, Observation)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_build_environment_from_task_spec_matches_loader_path_build() -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "random_start_terminal.yaml"
    )
    task_spec = load_task_spec(path=task_spec_path)
    env = build_environment_from_task_spec(task_spec=task_spec)

    try:
        observation, info = env.reset(seed=3)
        assert isinstance(observation, Observation)
        assert info["scenario"] == "random_position"
    finally:
        env.close()


def test_run_cli_accepts_task_spec(monkeypatch: MonkeyPatch) -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "solver_opponent.yaml"
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "connect4",
            "--task-spec",
            str(task_spec_path),
            "--seed",
            "11",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Connect 4 board:" in output
    assert "Session ended." in output


def test_run_cli_accepts_chess_task_spec_without_reward_flag(
    monkeypatch: MonkeyPatch,
) -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "chess"
        / "standard_start_terminal.yaml"
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "chess",
            "--task-spec",
            str(task_spec_path),
            "--seed",
            "11",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Session ended." in output


def test_run_cli_rejects_task_spec_for_different_game(
    monkeypatch: MonkeyPatch,
) -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "random_start_terminal.yaml"
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    with pytest.raises(SystemExit):
        run_cli(
            [
                "play",
                "chess",
                "--task-spec",
                str(task_spec_path),
                "--seed",
                "11",
            ]
        )


def test_run_cli_rejects_task_spec_cli_overrides(
    monkeypatch: MonkeyPatch,
) -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "solver_opponent.yaml"
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    with pytest.raises(SystemExit):
        run_cli(
            [
                "play",
                "connect4",
                "--task-spec",
                str(task_spec_path),
                "--max-transitions",
                "999",
            ]
        )


def test_run_cli_allows_task_spec_image_output_dir_when_images_enabled(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_spec_path = tmp_path / "connect4_images.yaml"
    task_spec_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "id: connect4_images",
                "game: connect4",
                "scenario:",
                "  kind: random_position",
                "  rows: 6",
                "  columns: 7",
                "  connect_length: 4",
                "  min_start_moves: 0",
                "  max_start_moves: 0",
                "reward:",
                "  kind: terminal_outcome",
                "  perspective: mover",
                "  win_reward: 1.0",
                "  draw_reward: 0.0",
                "  loss_reward: -1.0",
                "observation:",
                "  include_images: true",
                "  image_size: 180",
                "",
            )
        ),
        encoding="utf-8",
    )
    image_output_dir = tmp_path / "images"
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "connect4",
            "--task-spec",
            str(task_spec_path),
            "--image-output-dir",
            str(image_output_dir),
        ]
    )

    assert exit_code == 0
    assert "Image paths:" in output_stream.getvalue()
    assert tuple(image_output_dir.glob("*.png"))


def test_run_cli_rejects_task_spec_image_output_dir_when_images_disabled(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "solver_opponent.yaml"
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    error_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)
    monkeypatch.setattr(sys, "stderr", error_stream)

    with pytest.raises(SystemExit):
        run_cli(
            [
                "play",
                "connect4",
                "--task-spec",
                str(task_spec_path),
                "--image-output-dir",
                str(tmp_path / "images"),
            ]
        )

    assert "observation.include_images: true" in error_stream.getvalue()


def test_run_cli_rejects_explicit_default_task_spec_cli_overrides(
    monkeypatch: MonkeyPatch,
) -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "solver_opponent.yaml"
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    with pytest.raises(SystemExit):
        run_cli(
            [
                "play",
                "connect4",
                "--task-spec",
                str(task_spec_path),
                "--reward",
                "terminal",
            ]
        )


def test_run_cli_rejects_abbreviated_task_spec_cli_overrides(
    monkeypatch: MonkeyPatch,
) -> None:
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "solver_opponent.yaml"
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    with pytest.raises(SystemExit):
        run_cli(
            [
                "play",
                "connect4",
                "--task-spec",
                str(task_spec_path),
                "--max-tr",
                "999",
            ]
        )


def test_build_connect4_environment_rejects_explicit_default_task_spec_cli_overrides() -> (
    None
):
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "solver_opponent.yaml"
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "play",
            "connect4",
            "--task-spec",
            str(task_spec_path),
            "--reward",
            "terminal",
        ]
    )

    with pytest.raises(SystemExit):
        build_connect4_environment(args, parser)


def test_build_connect4_environment_rejects_explicit_default_task_spec_cli_overrides_with_plain_parser() -> (
    None
):
    task_spec_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "games"
        / "connect4"
        / "solver_opponent.yaml"
    )
    parser = ArgumentParser(prog="connect4")
    add_common_play_arguments(parser)
    register_connect4_arguments(parser)
    args = parser.parse_args(
        [
            "--task-spec",
            str(task_spec_path),
            "--reward",
            "terminal",
        ]
    )

    with pytest.raises(SystemExit):
        build_connect4_environment(args, parser)


def test_run_cli_reports_task_spec_build_errors(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_spec_path = tmp_path / "bad_chess_build.yaml"
    task_spec_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "id: bad_chess_build",
                "game: chess",
                "scenario:",
                "  kind: starting_position",
                "reward:",
                "  kind: engine_eval_dense",
                "  perspective: mover",
                "  engine:",
                f"    path: {tmp_path / 'missing-stockfish'}",
                "    depth: 1",
                "    mate_score: 100",
                "",
            )
        ),
        encoding="utf-8",
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    error_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)
    monkeypatch.setattr(sys, "stderr", error_stream)

    with pytest.raises(SystemExit):
        run_cli(
            [
                "play",
                "chess",
                "--task-spec",
                str(task_spec_path),
            ]
        )

    assert "--task-spec could not be built:" in error_stream.getvalue()


def test_load_task_spec_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    task_spec_path = tmp_path / "bad.yaml"
    task_spec_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "id: bad_spec",
                "game: connect4",
                "scenario:",
                "  kind: random_position",
                "reward:",
                "  kind: terminal_outcome",
                "  perspective: mover",
                "  win_reward: 1.0",
                "  draw_reward: 0.0",
                "  loss_reward: -1.0",
                "surprise: true",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        load_task_spec(path=task_spec_path)


def test_load_task_spec_rejects_incompatible_chess_configuration(
    tmp_path: Path,
) -> None:
    task_spec_path = tmp_path / "bad_chess.yaml"
    task_spec_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "id: bad_chess",
                "game: chess",
                "scenario:",
                "  kind: starting_position",
                "reward:",
                "  kind: puzzle_dense",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="puzzle rewards require"):
        load_task_spec(path=task_spec_path)


def test_load_task_spec_rejects_top_level_control_for_game_without_control(
    tmp_path: Path,
) -> None:
    task_spec_path = tmp_path / "bad_2048_control.yaml"
    task_spec_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "id: bad_2048_control",
                "game: game2048",
                "scenario:",
                "  kind: random_start",
                "reward:",
                "  kind: target_tile",
                "control:",
                "  auto_advance:",
                "    kind: nonsense",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        load_task_spec(path=task_spec_path)


def test_load_task_spec_rejects_variant_specific_connect4_fields(
    tmp_path: Path,
) -> None:
    task_spec_path = tmp_path / "bad_connect4_variant.yaml"
    task_spec_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "id: bad_connect4_variant",
                "game: connect4",
                "scenario:",
                "  kind: fixed_board",
                "  board:",
                "    - '.......'",
                "    - '.......'",
                "    - '.......'",
                "    - '.......'",
                "    - '.......'",
                "    - '.......'",
                "  min_start_moves: 2",
                "reward:",
                "  kind: terminal_outcome",
                "  perspective: mover",
                "  win_reward: 1.0",
                "  draw_reward: 0.0",
                "  loss_reward: -1.0",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        load_task_spec(path=task_spec_path)


def test_load_task_spec_rejects_variant_specific_chess_fields(
    tmp_path: Path,
) -> None:
    task_spec_path = tmp_path / "bad_chess_variant.yaml"
    task_spec_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "id: bad_chess_variant",
                "game: chess",
                "scenario:",
                "  kind: starting_position",
                "reward:",
                "  kind: terminal_outcome",
                "  perspective: white",
                "  win_reward: 1.0",
                "  draw_reward: 0.0",
                "  loss_reward: -1.0",
                "  engine:",
                "    path: /tmp/missing",
                "    depth: 1",
                "    mate_score: 100",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        load_task_spec(path=task_spec_path)
