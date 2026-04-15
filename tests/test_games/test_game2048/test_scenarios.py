"""2048 scenario tests."""

from rlvr_games.games.game2048 import Game2048ChanceModel, RandomStartScenario


def test_random_start_scenario_is_seeded_and_spawns_two_tiles() -> None:
    scenario = RandomStartScenario(
        size=4,
        target_value=2048,
        start_tile_count=2,
        chance_model=Game2048ChanceModel(),
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
    assert "seed" not in info
    assert info["spawned_tiles"] == (
        {"row": 3, "col": 1, "value": 2},
        {"row": 2, "col": 0, "value": 2},
    )
