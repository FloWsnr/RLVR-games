"""Tests for chess tactic behavior."""

import chess
import chess.svg

from rlvr_physics.core.rendering import ImageContent
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.games.chess_tactics import (
    ChessTacticsSession,
    chess_tactics_task_spec,
    make_chess_tactics_instance,
    render_chess_tactics_image,
    render_chess_tactics_text,
    verify_chess_tactic_submission,
)


def test_chess_instance_spec_and_solution_are_deterministic() -> None:
    first = make_chess_tactics_instance(seed=0, puzzle_id="fools_mate_mate_in_one")
    second = make_chess_tactics_instance(seed=0, puzzle_id="fools_mate_mate_in_one")
    spec = chess_tactics_task_spec(seed=0)

    assert first.task_id == second.task_id
    assert first.privileged_payload["solution_moves_uci"] == ("d8h4",)
    assert first.public_payload["side_to_move"] == "black"
    assert spec.kind == first.kind
    assert [renderer.renderer_type for renderer in spec.renderers] == ["text", "image"]


def test_chess_text_and_image_renderers() -> None:
    instance = make_chess_tactics_instance(seed=0, puzzle_id="fools_mate_mate_in_one")

    text = render_chess_tactics_text(instance)
    image = render_chess_tactics_image(instance)
    fen = instance.public_payload["fen"]
    assert isinstance(fen, str)

    assert "mate in 1" in text.text()
    assert "FEN:" in text.text()
    assert isinstance(image.contents[0], ImageContent)
    assert image.contents[0].mime_type == "image/svg+xml"
    assert image.contents[0].data == chess.svg.board(
        board=chess.Board(fen),
        orientation=chess.WHITE,
        size=640,
    ).encode("utf-8")
    assert image.text() == text.text()


def test_chess_verifier_accepts_san_and_uci_mates() -> None:
    instance = make_chess_tactics_instance(seed=0, puzzle_id="fools_mate_mate_in_one")

    san = verify_chess_tactic_submission(instance, TaskSubmission.final_text("Qh4#"))
    uci = verify_chess_tactic_submission(instance, TaskSubmission.final_text("d8h4"))

    assert san.accepted
    assert san.correct
    assert san.reason == "checkmate"
    assert uci.accepted
    assert uci.correct
    assert uci.reason == "checkmate"


def test_chess_verifier_distinguishes_parse_illegal_and_legal_non_solution() -> None:
    instance = make_chess_tactics_instance(seed=0, puzzle_id="fools_mate_mate_in_one")

    parse_failure = verify_chess_tactic_submission(
        instance, TaskSubmission.final_text("not-a-move")
    )
    illegal = verify_chess_tactic_submission(
        instance, TaskSubmission.final_text("e2e4")
    )
    legal_non_solution = verify_chess_tactic_submission(
        instance, TaskSubmission.final_text("a7a6")
    )

    assert not parse_failure.accepted
    assert parse_failure.reason == "parse_failure"
    assert not illegal.accepted
    assert illegal.reason == "illegal_move"
    assert legal_non_solution.accepted
    assert legal_non_solution.reason == "legal_non_solution"


def test_chess_session_records_reward_and_finishes() -> None:
    instance = make_chess_tactics_instance(seed=0, puzzle_id="fools_mate_mate_in_one")
    session = ChessTacticsSession(instance, "text")

    session.reset(seed=1)
    result = session.submit(TaskSubmission.final_text("Qh4#"))

    assert result.done
    assert result.reward == 1.0
    assert session.turn is None
    assert [event.event_type for event in session.trajectory.snapshot()] == [
        "reset",
        "observation",
        "submission",
        "reward",
    ]
