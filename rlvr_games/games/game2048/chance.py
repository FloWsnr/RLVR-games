"""Chance-model helpers for 2048 random tile spawns."""

from dataclasses import dataclass
from random import Random
from typing import Any

from rlvr_games.games.game2048.engine import (
    Board,
    SpawnOutcome,
    SpawnSummary,
    spawn_outcomes,
    spawn_random_tile,
)

RandomState = tuple[Any, ...]


@dataclass(slots=True, frozen=True)
class SpawnTransition:
    """One sampled post-move random-tile transition.

    Attributes
    ----------
    board : Board
        Board after inserting the sampled tile.
    rng_state : RandomState
        Updated Python RNG state after the spawn has been sampled.
    spawned_tile : SpawnSummary
        Structured metadata describing the sampled tile.
    """

    board: Board
    rng_state: RandomState
    spawned_tile: SpawnSummary


class Game2048ChanceModel:
    """Encapsulate deterministic RNG-backed chance events for 2048."""

    def initial_rng_state(self, *, seed: int) -> RandomState:
        """Return the initial RNG state for a seeded episode.

        Parameters
        ----------
        seed : int
            Explicit seed used to initialize the episode RNG.

        Returns
        -------
        RandomState
            Python RNG internal state suitable for future tile spawns.
        """
        rng = Random(seed)
        return rng.getstate()

    def spawn_tile(
        self,
        *,
        board: Board,
        rng_state: RandomState,
    ) -> SpawnTransition:
        """Sample one random tile insertion from the supplied RNG state.

        Parameters
        ----------
        board : Board
            Board on which a random tile should be inserted.
        rng_state : RandomState
            Python RNG internal state describing the next chance outcome.

        Returns
        -------
        SpawnTransition
            Sampled board, updated RNG state, and spawned tile metadata.
        """
        rng = Random()
        rng.setstate(rng_state)
        next_board, spawned_tile = spawn_random_tile(board=board, rng=rng)
        return SpawnTransition(
            board=next_board,
            rng_state=rng.getstate(),
            spawned_tile=spawned_tile,
        )

    def spawn_outcomes(self, *, board: Board) -> tuple[SpawnOutcome, ...]:
        """Enumerate all legal random-tile outcomes for a board.

        Parameters
        ----------
        board : Board
            Board to analyze for the next random tile spawn.

        Returns
        -------
        tuple[SpawnOutcome, ...]
            Explicit spawn outcomes with their exact probabilities.
        """
        return spawn_outcomes(board=board)
