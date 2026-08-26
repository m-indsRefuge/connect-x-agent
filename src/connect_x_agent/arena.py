from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeatResult:
    games: int
    wins: int
    draws: int
    losses: int
    failures: int


@dataclass(frozen=True)
class ArenaReport:
    player_one: SeatResult
    player_two: SeatResult

    @classmethod
    def from_rewards(
        cls,
        player_one_rewards: list[list[float | None]],
        player_two_rewards: list[list[float | None]],
    ) -> "ArenaReport":
        return cls(
            player_one=summarize_rewards(
                player_one_rewards,
                candidate_index=0,
            ),
            player_two=summarize_rewards(
                player_two_rewards,
                candidate_index=1,
            ),
        )

    @property
    def games(self) -> int:
        return self.player_one.games + self.player_two.games

    @property
    def wins(self) -> int:
        return self.player_one.wins + self.player_two.wins

    @property
    def draws(self) -> int:
        return self.player_one.draws + self.player_two.draws

    @property
    def losses(self) -> int:
        return self.player_one.losses + self.player_two.losses

    @property
    def failures(self) -> int:
        return self.player_one.failures + self.player_two.failures

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0

        return self.wins / self.games


def summarize_rewards(
    rewards: list[list[float | None]],
    candidate_index: int,
) -> SeatResult:
    wins = 0
    draws = 0
    losses = 0
    failures = 0

    for episode in rewards:
        reward = episode[candidate_index]

        if reward == 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1
        else:
            failures += 1

    return SeatResult(
        games=len(rewards),
        wins=wins,
        draws=draws,
        losses=losses,
        failures=failures,
    )


def run_arena(
    candidate: Any,
    opponent: Any,
    games_per_seat: int,
) -> ArenaReport:
    # Import lazily so ordinary unit tests do not pay the heavy
    # kaggle-environments/OpenSpiel startup cost.
    from kaggle_environments import evaluate  # type: ignore[import-untyped]

    player_one_rewards = evaluate(
        "connectx",
        [candidate, opponent],
        num_episodes=games_per_seat,
    )

    player_two_rewards = evaluate(
        "connectx",
        [opponent, candidate],
        num_episodes=games_per_seat,
    )

    return ArenaReport.from_rewards(
        player_one_rewards=player_one_rewards,
        player_two_rewards=player_two_rewards,
    )
