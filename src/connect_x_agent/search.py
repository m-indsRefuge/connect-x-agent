from dataclasses import dataclass
from typing import Literal

GameValue = Literal["win", "draw", "loss", "unknown"]


@dataclass(frozen=True)
class MoveAnalysis:
    column: int
    value: GameValue


@dataclass(frozen=True)
class PositionSolution:
    value: GameValue
    moves: tuple[MoveAnalysis, ...]
    complete: bool


def _aggregate_values(values: tuple[GameValue, ...]) -> GameValue:
    if "win" in values:
        return "win"
    if "unknown" in values:
        return "unknown"
    if "draw" in values:
        return "draw"
    return "loss"
