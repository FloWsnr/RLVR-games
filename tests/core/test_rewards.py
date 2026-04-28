"""Tests for shared reward result types."""

import pytest

from rlvr_physics.core.rewards import RewardResult


def test_reward_result_metadata_is_frozen() -> None:
    result = RewardResult(
        reward=0.75,
        score=0.5,
        public_info={"reason": "partial"},
        debug_info={"exact": 1.0},
    )

    assert result.reward == 0.75
    assert result.score == 0.5
    assert result.public_info["reason"] == "partial"
    assert result.debug_info["exact"] == 1.0
    with pytest.raises(TypeError):
        result.public_info["extra"] = "blocked"  # type: ignore[index]
