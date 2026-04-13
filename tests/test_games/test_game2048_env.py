"""2048 backend and environment tests."""

from random import Random

import pytest
from PIL import Image

from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig, RenderedImage
from rlvr_games.games.game2048 import (
    FixedBoardScenario,
    Game2048AsciiBoardFormatter,
    Game2048Backend,
    Game2048Env,
    Game2048ImageRenderer,
    Game2048ObservationRenderer,
    Game2048State,
    RandomStartScenario,
    ScoreDeltaReward,
    TargetTileReward,
    make_game2048_env,
    spawn_outcomes,
)
from rlvr_games.games.game2048.actions import MoveDirection
from rlvr_games.games.game2048.engine import apply_move

MERGE_BOARD = (
    (2, 2, 2, 2),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)
WIN_BOARD = (
    (1024, 1024, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)
NO_MOVES_BOARD = (
    (2, 4, 2, 4),
    (4, 2, 4, 2),
    (2, 4, 2, 4),
    (4, 2, 4, 8),
)
NOOP_UP_BOARD = (
    (2, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)


def make_renderer() -> Game2048ObservationRenderer:
    """Construct a text-only 2048 observation renderer.

    Returns
    -------
    Game2048ObservationRenderer
        Observation renderer with the standard ASCII board formatter.
    """
    return Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=None,
    )


class Stub2048ImageRenderer:
    """Small stub image renderer for observation tests."""

    def render_images(
        self,
        board: tuple[tuple[int, ...], ...],
    ) -> tuple[RenderedImage, ...]:
        """Return a single fixed image payload.

        Parameters
        ----------
        board : tuple[tuple[int, ...], ...]
            Board being rendered. It is ignored by the stub.

        Returns
        -------
        tuple[RenderedImage, ...]
            Single fixed RGBA image payload.
        """
        del board
        return (
            RenderedImage(
                key="stub-2048-board",
                image=Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
            ),
        )


def test_random_start_scenario_is_seeded_and_spawns_two_tiles() -> None:
    scenario = RandomStartScenario(
        size=4,
        target_value=2048,
        start_tile_count=2,
    )

    state, info = scenario.reset(seed=0)

    assert state.board == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (2, 0, 0, 0),
        (0, 2, 0, 0),
    )
    assert state.score == 0
    assert state.move_count == 0
    assert info["scenario"] == "random_start"
    assert info["spawned_tiles"] == (
        {"row": 3, "col": 1, "value": 2},
        {"row": 2, "col": 0, "value": 2},
    )


def test_legal_actions_exclude_noop_directions() -> None:
    backend = Game2048Backend()
    state = Game2048State(
        board=NOOP_UP_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(0).getstate(),
    )

    legal_actions = backend.legal_actions(state)

    assert legal_actions == ["right", "down"]


def test_target_tile_terminal_state_has_no_legal_actions() -> None:
    state = Game2048State(
        board=((2048, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        score=2048,
        move_count=1,
        target_value=2048,
        rng_state=Random(0).getstate(),
    )

    assert state.is_terminal is True
    assert state.legal_actions == ()
    assert state.legal_action_count == 0


@pytest.mark.parametrize("raw_action", ["", "north", "up"])
def test_parse_action_rejects_invalid_or_illegal_directions(raw_action: str) -> None:
    backend = Game2048Backend()
    state = Game2048State(
        board=NOOP_UP_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(0).getstate(),
    )

    parse_result = backend.parse_action(state, raw_action)

    assert parse_result.action is None
    assert parse_result.error is not None


def test_apply_action_merges_pairs_once_and_records_spawn_metadata() -> None:
    backend = Game2048Backend()
    state = Game2048State(
        board=MERGE_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(5).getstate(),
    )

    action = backend.parse_action(state, "left").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.board == (
        (4, 4, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 2, 0, 0),
    )
    assert next_state.score == 8
    assert next_state.move_count == 1
    assert info["score_gain"] == 8
    assert info["merge_count"] == 2
    assert info["spawned_tile"] == {"row": 3, "col": 1, "value": 2}
    assert info["merges"] == (
        {"row": 0, "col": 0, "value": 4, "sources": ((0, 0), (0, 1))},
        {"row": 0, "col": 1, "value": 4, "sources": ((0, 2), (0, 3))},
    )


def test_apply_move_matches_open_spiel_three_tiles_downward_case() -> None:
    board = (
        (0, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 0, 0, 0),
    )

    move_summary = apply_move(board=board, direction=MoveDirection.DOWN)

    assert move_summary.board == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (2, 0, 0, 0),
        (4, 0, 0, 0),
    )


def test_apply_move_matches_open_spiel_one_merge_per_turn_case() -> None:
    board = (
        (2, 4, 0, 4),
        (0, 2, 0, 2),
        (0, 0, 0, 0),
        (0, 2, 0, 0),
    )

    move_summary = apply_move(board=board, direction=MoveDirection.DOWN)

    assert move_summary.board == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 4, 0, 4),
        (2, 4, 0, 2),
    )


def test_score_delta_reward_returns_merge_score_gain_for_non_terminal_move() -> None:
    backend = Game2048Backend()
    state = Game2048State(
        board=MERGE_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(5).getstate(),
    )
    action = backend.parse_action(state, "left").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = ScoreDeltaReward().evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 8.0


def test_target_tile_reward_ignores_non_terminal_score_gain() -> None:
    env = make_game2048_env(
        size=4,
        target_value=2048,
        initial_board=MERGE_BOARD,
        initial_score=0,
        initial_move_count=0,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )
    env.reset(seed=5)

    result = env.step("left")

    assert result.accepted is True
    assert result.reward == 0.0
    assert result.info["score_gain"] == 8
    assert result.terminated is False


def test_reaching_target_tile_terminates_with_reward_and_metadata() -> None:
    env = Game2048Env(
        backend=Game2048Backend(),
        scenario=FixedBoardScenario(
            initial_board=WIN_BOARD,
            initial_score=0,
            initial_move_count=0,
            target_value=2048,
        ),
        renderer=make_renderer(),
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
    )
    env.reset(seed=1)

    result = env.step("left")

    assert result.accepted is True
    assert result.reward == 2048.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["spawned_tile"] is None
    assert result.info["termination"] == "target_tile"
    assert result.info["won"] is True
    assert env.state.board == (
        (2048, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert env.state.legal_actions == ()
    assert result.observation.metadata["is_terminal"] is True
    assert result.observation.metadata["legal_action_count"] == 0
    assert result.observation.metadata["won"] is True


def test_terminal_reset_marks_episode_finished_and_rejects_steps() -> None:
    env = Game2048Env(
        backend=Game2048Backend(),
        scenario=FixedBoardScenario(
            initial_board=NO_MOVES_BOARD,
            initial_score=0,
            initial_move_count=0,
            target_value=2048,
        ),
        renderer=make_renderer(),
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
    )

    observation, _ = env.reset(seed=23)

    assert env.episode_finished is True
    assert observation.metadata["is_terminal"] is True
    assert observation.metadata["termination"] == "no_moves"

    with pytest.raises(EpisodeFinishedError):
        env.step("left")


def test_game2048_env_records_trajectory_with_real_backend() -> None:
    env = Game2048Env(
        backend=Game2048Backend(),
        scenario=RandomStartScenario(
            size=4,
            target_value=2048,
            start_tile_count=2,
        ),
        renderer=make_renderer(),
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
    )
    observation, info = env.reset(seed=0)

    result = env.step("left")

    assert info["seed"] == 0
    assert "2048 board:" in (observation.text or "")
    assert result.reward == 0.0
    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].accepted is True
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.label == "left"
    assert env.trajectory.steps[0].info["direction"] == "left"


def test_observation_renderer_can_emit_text_and_images() -> None:
    renderer = Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=Stub2048ImageRenderer(),
    )

    observation = renderer.render(
        Game2048State(
            board=MERGE_BOARD,
            score=0,
            move_count=0,
            target_value=2048,
            rng_state=Random(0).getstate(),
        )
    )

    assert observation.text is not None
    assert "2048 board:" in observation.text
    assert len(observation.images) == 1
    assert observation.images[0].key == "stub-2048-board"
    assert observation.images[0].image.size == (32, 32)


def test_image_renderer_emits_single_raster_image() -> None:
    renderer = Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=Game2048ImageRenderer(size=360),
    )

    observation = renderer.render(
        Game2048State(
            board=MERGE_BOARD,
            score=8,
            move_count=1,
            target_value=2048,
            rng_state=Random(0).getstate(),
        )
    )

    assert len(observation.images) == 1
    rendered_image = observation.images[0]
    assert rendered_image.key.startswith("2048-board-")
    assert rendered_image.image.size == (360, 360)
    assert rendered_image.image.mode == "RGBA"


def test_image_renderer_is_deterministic_for_the_same_position() -> None:
    image_renderer = Game2048ImageRenderer(size=360)
    board = MERGE_BOARD

    first_render = image_renderer.render_images(board)[0]
    second_render = image_renderer.render_images(board)[0]

    assert second_render.key == first_render.key
    assert second_render.image.tobytes() == first_render.image.tobytes()


def test_spawn_outcomes_cover_all_empty_cells_with_probabilities() -> None:
    board = (
        (2, 4, 8, 16),
        (32, 64, 128, 256),
        (512, 1024, 0, 4096),
        (8192, 16384, 0, 32768),
    )

    outcomes = spawn_outcomes(board=board)

    assert len(outcomes) == 4
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    assert {
        (outcome.spawned_tile.row, outcome.spawned_tile.col) for outcome in outcomes
    } == {
        (2, 2),
        (3, 2),
    }
    probability_by_tile = {
        (
            outcome.spawned_tile.row,
            outcome.spawned_tile.col,
            outcome.spawned_tile.value,
        ): outcome.probability
        for outcome in outcomes
    }
    assert probability_by_tile[(2, 2, 2)] == pytest.approx(0.45)
    assert probability_by_tile[(2, 2, 4)] == pytest.approx(0.05)
    assert probability_by_tile[(3, 2, 2)] == pytest.approx(0.45)
    assert probability_by_tile[(3, 2, 4)] == pytest.approx(0.05)


def test_image_renderer_supports_large_tile_labels() -> None:
    renderer = Game2048ObservationRenderer(
        board_formatter=Game2048AsciiBoardFormatter(),
        image_renderer=Game2048ImageRenderer(size=360),
    )

    observation = renderer.render(
        Game2048State(
            board=((131072, 0, 0, 0), (0, 65536, 0, 0), (0, 0, 32768, 0), (0, 0, 0, 0)),
            score=0,
            move_count=0,
            target_value=262144,
            rng_state=Random(0).getstate(),
        )
    )

    assert len(observation.images) == 1
    assert observation.images[0].image.size == (360, 360)
