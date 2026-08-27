from dataclasses import dataclass
from typing import Literal

from connect_x_agent.tactical import drop_piece, is_win, legal_columns

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



def _has_consistent_last_move(
    board: list[int],
    winner: int,
    columns: int,
    rows: int,
    inarow: int,
) -> bool:
    for column in range(columns):
        topmost: int | None = None

        for row in range(rows):
            index = row * columns + column

            if board[index] != 0:
                topmost = index
                break

        if topmost is None or board[topmost] != winner:
            continue

        predecessor = board.copy()
        predecessor[topmost] = 0

        if not _gravity_is_valid(
            predecessor,
            columns,
            rows,
        ):
            continue

        if not _counts_match_turn(
            predecessor,
            winner,
        ):
            continue

        if is_win(
            predecessor,
            1,
            columns,
            rows,
            inarow,
        ):
            continue

        if is_win(
            predecessor,
            2,
            columns,
            rows,
            inarow,
        ):
            continue

        return True

    return False


def _winner(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> int | None:
    p1_wins = is_win(
        board,
        1,
        columns,
        rows,
        inarow,
    )
    p2_wins = is_win(
        board,
        2,
        columns,
        rows,
        inarow,
    )

    if p1_wins and p2_wins:
        raise ValueError(
            "both players cannot have winning lines"
        )

    if not p1_wins and not p2_wins:
        return None

    winner = 1 if p1_wins else 2
    previous_player = 3 - mark

    if winner != previous_player:
        raise ValueError(
            "winner must be the previous player"
        )

    if not _has_consistent_last_move(
        board,
        winner,
        columns,
        rows,
        inarow,
    ):
        raise ValueError(
            "winning board has no consistent last move"
        )

    return winner


def _invert_value(
    value: GameValue,
) -> GameValue:
    if value == "win":
        return "loss"

    if value == "loss":
        return "win"

    return value


def _solve_exact_value(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> GameValue:
    legal = legal_columns(
        board,
        columns,
    )

    if not legal:
        return "draw"

    move_values: list[GameValue] = []

    for column in legal:
        child = drop_piece(
            board,
            column,
            mark,
            columns,
            rows,
        )

        if is_win(
            child,
            mark,
            columns,
            rows,
            inarow,
        ):
            move_value: GameValue = "win"
        else:
            child_value = _solve_exact_value(
                child,
                3 - mark,
                columns,
                rows,
                inarow,
            )
            move_value = _invert_value(
                child_value
            )

        move_values.append(
            move_value
        )

    return _aggregate_values(
        tuple(move_values)
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

    if _winner(
        board,
        mark,
        columns,
        rows,
        inarow,
    ) is not None:
        return PositionSolution(
            value="loss",
            moves=(),
            complete=True,
        )

    legal = legal_columns(
        board,
        columns,
    )

    if not legal:
        return PositionSolution(
            value="draw",
            moves=(),
            complete=True,
        )

    if max_depth is not None:
        return _unknown_root_solution(
            board,
            columns,
        )

    moves: list[MoveAnalysis] = []

    for column in legal:
        child = drop_piece(
            board,
            column,
            mark,
            columns,
            rows,
        )

        if is_win(
            child,
            mark,
            columns,
            rows,
            inarow,
        ):
            move_value: GameValue = "win"
        else:
            child_value = _solve_exact_value(
                child,
                3 - mark,
                columns,
                rows,
                inarow,
            )
            move_value = _invert_value(
                child_value
            )

        moves.append(
            MoveAnalysis(
                column=column,
                value=move_value,
            )
        )

    move_tuple = tuple(moves)

    return PositionSolution(
        value=_aggregate_values(
            tuple(
                move.value
                for move in move_tuple
            )
        ),
        moves=move_tuple,
        complete=True,
    )
