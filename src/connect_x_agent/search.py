from dataclasses import dataclass
from typing import Literal

from connect_x_agent.tactical import legal_columns

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


def _aggregate_values(
    values: tuple[GameValue, ...],
) -> GameValue:
    if "win" in values:
        return "win"
    if "unknown" in values:
        return "unknown"
    if "draw" in values:
        return "draw"
    return "loss"


def _require_positive_int(
    name: str,
    value: int,
) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer"
        )


def _counts_match_turn(
    board: list[int],
    mark: int,
) -> bool:
    p1 = board.count(1)
    p2 = board.count(2)

    if mark == 1:
        return p1 == p2

    return p1 == p2 + 1


def _gravity_is_valid(
    board: list[int],
    columns: int,
    rows: int,
) -> bool:
    for column in range(columns):
        empty_seen_below = False

        for row in range(rows - 1, -1, -1):
            value = board[row * columns + column]

            if value == 0:
                empty_seen_below = True
            elif empty_seen_below:
                return False

    return True


def _validate_structure(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    max_depth: int | None,
) -> None:
    _require_positive_int("columns", columns)
    _require_positive_int("rows", rows)
    _require_positive_int("inarow", inarow)

    if type(mark) is not int or mark not in (1, 2):
        raise ValueError("mark must be 1 or 2")

    if max_depth is not None and (
        type(max_depth) is not int
        or max_depth < 0
    ):
        raise ValueError(
            "max_depth must be None "
            "or a non-negative integer"
        )

    if len(board) != columns * rows:
        raise ValueError(
            "board length must equal columns * rows"
        )

    if any(
        type(cell) is not int
        or cell not in (0, 1, 2)
        for cell in board
    ):
        raise ValueError(
            "board cell values must be 0, 1, or 2"
        )

    if not _gravity_is_valid(
        board,
        columns,
        rows,
    ):
        raise ValueError("board violates gravity")

    if not _counts_match_turn(board, mark):
        raise ValueError(
            "piece counts are inconsistent with turn"
        )


def _unknown_root_solution(
    board: list[int],
    columns: int,
) -> PositionSolution:
    moves = tuple(
        MoveAnalysis(
            column=column,
            value="unknown",
        )
        for column in legal_columns(
            board,
            columns,
        )
    )

    return PositionSolution(
        value="unknown",
        moves=moves,
        complete=False,
    )


def solve_position(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    max_depth: int | None = None,
) -> PositionSolution:
    _validate_structure(
        board,
        mark,
        columns,
        rows,
        inarow,
        max_depth,
    )

    return _unknown_root_solution(
        board,
        columns,
    )
