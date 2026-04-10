"""Chess backend and environment tests."""

from pathlib import Path

import chess
import pytest

from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess import (
    AsciiBoardFormatter,
    ChessBackend,
    ChessEnv,
    ChessObservationRenderer,
    ChessRasterBoardImageRenderer,
    ChessState,
    StartingPositionScenario,
    UnicodeBoardFormatter,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN

PROMOTION_FEN = "k7/4P3/8/8/8/8/8/7K w - - 0 1"


def make_renderer() -> ChessObservationRenderer:
    return ChessObservationRenderer(
        board_formatter=AsciiBoardFormatter(orientation=chess.WHITE),
        image_renderer=None,
    )


class StubChessImageRenderer:
    def render_images(self, board: chess.Board) -> tuple[Path, ...]:
        del board
        return (Path("/tmp/chess-board.png"),)


def test_legal_actions_from_start_position_are_sorted_uci() -> None:
    backend = ChessBackend()
    state = ChessState(fen=STANDARD_START_FEN)

    legal_actions = backend.legal_actions(state)

    assert len(legal_actions) == 20
    assert legal_actions == sorted(legal_actions)
    assert "e2e4" in legal_actions
    assert "g1f3" in legal_actions


def test_apply_action_updates_state_and_transition_info() -> None:
    backend = ChessBackend()
    state = ChessState(fen=STANDARD_START_FEN)

    action = backend.parse_action(state, "e2e4").require_action()
    next_state, info = backend.apply_action(state, action)

    assert action.uci == "e2e4"
    assert (
        next_state.fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    assert info == {
        "move_uci": "e2e4",
        "move_san": "e4",
        "fen": next_state.fen,
        "side_to_move": "black",
        "legal_action_count": 20,
        "repetition_count": 1,
        "is_check": False,
        "is_terminal": False,
    }


@pytest.mark.parametrize("raw_action", ["", "bad", "e2e5"])
def test_parse_action_rejects_invalid_or_illegal_uci(raw_action: str) -> None:
    backend = ChessBackend()
    state = ChessState(fen=STANDARD_START_FEN)

    parse_result = backend.parse_action(state, raw_action)

    assert parse_result.action is None
    assert parse_result.error is not None


def test_promotion_requires_suffix_and_applies_correctly() -> None:
    backend = ChessBackend()
    state = ChessState(fen=PROMOTION_FEN)

    rejected = backend.parse_action(state, "e7e8")
    assert rejected.action is None
    assert rejected.error is not None

    action = backend.parse_action(state, "e7e8q").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.fen == "k3Q3/8/8/8/8/8/8/7K b - - 0 1"
    assert info["move_san"] == "e8=Q+"
    assert info["is_terminal"] is False


def test_checkmate_sequence_terminates_with_winner_metadata() -> None:
    env = ChessEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(),
        renderer=make_renderer(),
        reward_fn=ZeroReward(),
        config=EpisodeConfig(),
    )
    env.reset(seed=11)

    for raw_action in ["f2f3", "e7e5", "g2g4"]:
        result = env.step(raw_action)
        assert result.terminated is False

    final_result = env.step("d8h4")

    assert final_result.terminated is True
    assert final_result.truncated is False
    assert final_result.accepted is True
    assert final_result.info["move_san"] == "Qh4#"
    assert final_result.info["winner"] == "black"
    assert final_result.info["result"] == "0-1"
    assert final_result.info["termination"] == "checkmate"
    assert final_result.observation.metadata["is_terminal"] is True
    assert final_result.observation.metadata["winner"] == "black"


def test_threefold_repetition_terminates_on_the_third_occurrence() -> None:
    env = ChessEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(),
        renderer=make_renderer(),
        reward_fn=ZeroReward(),
        config=EpisodeConfig(),
    )
    env.reset(seed=5)

    repeated_knight_moves = [
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
    ]
    for raw_action in repeated_knight_moves:
        result = env.step(raw_action)
        assert result.terminated is False

    final_result = env.step("f6g8")

    assert final_result.terminated is True
    assert final_result.info["winner"] is None
    assert final_result.info["result"] == "1/2-1/2"
    assert final_result.info["termination"] == "threefold_repetition"
    assert final_result.info["repetition_count"] == 3


def test_custom_valid_fen_reset_is_normalized() -> None:
    env = ChessEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen=PROMOTION_FEN),
        renderer=make_renderer(),
        reward_fn=ZeroReward(),
        config=EpisodeConfig(),
    )

    observation, info = env.reset(seed=17)

    assert info["scenario"] == "fen_position"
    assert info["initial_fen"] == PROMOTION_FEN
    assert observation.metadata["fen"] == PROMOTION_FEN
    assert observation.metadata["side_to_move"] == "white"


def test_invalid_fen_reset_fails_fast() -> None:
    env = ChessEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen="not-a-fen"),
        renderer=make_renderer(),
        reward_fn=ZeroReward(),
        config=EpisodeConfig(),
    )

    with pytest.raises(ValueError):
        env.reset(seed=3)


def test_chess_env_records_trajectory_with_real_backend() -> None:
    env = ChessEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(),
        renderer=make_renderer(),
        reward_fn=ZeroReward(),
        config=EpisodeConfig(),
    )
    observation, info = env.reset(seed=123)

    result = env.step("e2e4")

    assert info["seed"] == 123
    assert STANDARD_START_FEN in (observation.text or "")
    assert result.reward == 0.0
    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].accepted is True
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.uci == "e2e4"
    assert env.trajectory.steps[0].info["move_san"] == "e4"


def test_observation_renderer_can_emit_text_and_image_paths() -> None:
    renderer = ChessObservationRenderer(
        board_formatter=AsciiBoardFormatter(orientation=chess.WHITE),
        image_renderer=StubChessImageRenderer(),
    )

    observation = renderer.render(ChessState(fen=STANDARD_START_FEN))

    assert observation.text is not None
    assert "Chess board:" in observation.text
    assert observation.image_paths == (Path("/tmp/chess-board.png"),)


def test_raster_board_image_renderer_writes_png_image_path(tmp_path: Path) -> None:
    renderer = ChessObservationRenderer(
        board_formatter=AsciiBoardFormatter(orientation=chess.WHITE),
        image_renderer=ChessRasterBoardImageRenderer(
            output_dir=tmp_path,
            size=360,
            coordinates=True,
            orientation=chess.WHITE,
        ),
    )

    observation = renderer.render(ChessState(fen=STANDARD_START_FEN))

    assert len(observation.image_paths) == 1
    image_path = observation.image_paths[0]
    assert image_path.parent == tmp_path
    assert image_path.suffix == ".png"
    assert image_path.exists()
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_raster_board_image_renderer_reuses_existing_image(tmp_path: Path) -> None:
    image_renderer = ChessRasterBoardImageRenderer(
        output_dir=tmp_path,
        size=360,
        coordinates=True,
        orientation=chess.WHITE,
    )
    board = chess.Board(STANDARD_START_FEN)

    first_image_path = image_renderer.render_images(board)[0]
    first_image_path.write_bytes(b"cached")
    second_image_path = image_renderer.render_images(board)[0]

    assert second_image_path == first_image_path
    assert second_image_path.read_bytes() == b"cached"


def test_unicode_board_formatter_can_be_used_in_observation_renderer() -> None:
    renderer = ChessObservationRenderer(
        board_formatter=UnicodeBoardFormatter(orientation=chess.WHITE),
        image_renderer=None,
    )

    observation = renderer.render(ChessState(fen=STANDARD_START_FEN))

    assert observation.text is not None
    assert "♜" in observation.text
    assert "a b c d e f g h" in observation.text


def test_ascii_board_formatter_supports_black_orientation() -> None:
    formatter = AsciiBoardFormatter(orientation=chess.BLACK)

    board_text = formatter.render_text(chess.Board(STANDARD_START_FEN))

    lines = board_text.splitlines()
    assert lines[0] == "1 R N B K Q B N R"
    assert lines[-1] == "  h g f e d c b a"


def test_unicode_board_formatter_supports_black_orientation() -> None:
    formatter = UnicodeBoardFormatter(orientation=chess.BLACK)

    board_text = formatter.render_text(chess.Board(STANDARD_START_FEN))

    assert "1 |♖|♘|♗|♔|♕|♗|♘|♖|" in board_text
    assert "   h g f e d c b a" in board_text
