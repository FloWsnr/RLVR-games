"""2048 scenario tests."""

from rlvr_games.games.game2048 import Game2048ChanceModel, RandomStartScenario


def test_random_start_scenario_returns_seeded_empty_board_setup() -> None:
    scenario = RandomStartScenario(
        size=4,
        target_value=2048,
        start_tile_count=2,
        chance_model=Game2048ChanceModel(),
    )

    reset = scenario.reset(seed=0)
    state = reset.initial_state
    info = reset.reset_info

    assert state.board == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert state.score == 0
    assert state.move_count == 0
    assert info["scenario"] == "random_start"
    assert "seed" not in info
    assert info["start_tile_count"] == 2
