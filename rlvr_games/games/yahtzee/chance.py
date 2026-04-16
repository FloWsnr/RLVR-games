"""Chance-model helpers for Yahtzee dice rolls."""

from dataclasses import dataclass
from random import Random
from typing import Any

from rlvr_games.games.yahtzee.engine import Dice, ZERO_DICE

type RandomState = tuple[Any, ...]


@dataclass(slots=True, frozen=True)
class DiceRollTransition:
    """One sampled Yahtzee dice-roll transition.

    Attributes
    ----------
    dice : Dice
        Dice after the sampled roll has been applied.
    rng_state : RandomState
        Updated Python RNG state after the roll.
    rolled_positions : tuple[int, ...]
        Zero-based positions rerolled during this transition.
    """

    dice: Dice
    rng_state: RandomState
    rolled_positions: tuple[int, ...]


class YahtzeeChanceModel:
    """Encapsulate deterministic RNG-backed Yahtzee dice rolls."""

    def initial_rng_state(self, *, seed: int) -> RandomState:
        """Return the initial RNG state for a seeded episode."""
        rng = Random(seed)
        return rng.getstate()

    def roll_all(self, *, rng_state: RandomState) -> DiceRollTransition:
        """Roll all five dice from the supplied RNG state."""
        return self.reroll(
            dice=ZERO_DICE,
            reroll_positions=(0, 1, 2, 3, 4),
            rng_state=rng_state,
        )

    def reroll(
        self,
        *,
        dice: Dice,
        reroll_positions: tuple[int, ...],
        rng_state: RandomState,
    ) -> DiceRollTransition:
        """Reroll one subset of dice from the supplied RNG state."""
        rng = Random()
        rng.setstate(rng_state)

        next_dice = list(dice)
        for position in reroll_positions:
            next_dice[position] = rng.randint(1, 6)

        return DiceRollTransition(
            dice=tuple(next_dice),  # type: ignore[arg-type]
            rng_state=rng.getstate(),
            rolled_positions=reroll_positions,
        )
