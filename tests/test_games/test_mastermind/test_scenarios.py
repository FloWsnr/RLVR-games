"""Mastermind scenario tests."""

from rlvr_games.games.mastermind import FixedCodeScenario, StandardGameScenario


def test_standard_game_scenario_is_deterministic_by_seed() -> None:
    scenario = StandardGameScenario()

    first_reset = scenario.reset(seed=7)
    second_reset = scenario.reset(seed=7)

    assert (
        first_reset.initial_state.secret_code == second_reset.initial_state.secret_code
    )
    assert first_reset.reset_info["scenario"] == "standard_game"
    assert "secret_code" not in first_reset.reset_info


def test_fixed_code_scenario_hides_the_secret_in_public_reset_info() -> None:
    scenario = FixedCodeScenario(secret_code=(1, 1, 2, 2))

    reset = scenario.reset(seed=3)

    assert reset.initial_state.secret_code == (1, 1, 2, 2)
    assert reset.reset_info["scenario"] == "fixed_code"
    assert "secret_code" not in reset.reset_info
