from __future__ import annotations

from connect_x_agent.search import (
    GameValue,
    MoveAnalysis,
    PositionSolution,
)

COLUMNS = 7
ROWS = 6
INAROW = 4
BOARD_SIZE = COLUMNS * ROWS


def _legal_columns(board: list[int]) -> list[int]:
    return [
        column
        for column in range(COLUMNS)
        if board[column] == 0
    ]


def _drop_piece(
    board: list[int],
    column: int,
    mark: int,
) -> list[int]:
    child = board.copy()

    for row in range(ROWS - 1, -1, -1):
        index = row * COLUMNS + column
        if child[index] == 0:
            child[index] = mark
            return child

    raise ValueError(f"Column {column} is full")


def _is_win(
    board: list[int],
    mark: int,
) -> bool:
    directions = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    )

    for row in range(ROWS):
        for column in range(COLUMNS):
            if board[row * COLUMNS + column] != mark:
                continue

            for row_step, column_step in directions:
                if all(
                    0 <= row + row_step * offset < ROWS
                    and 0 <= column + column_step * offset < COLUMNS
                    and board[
                        (row + row_step * offset) * COLUMNS
                        + column
                        + column_step * offset
                    ]
                    == mark
                    for offset in range(1, INAROW)
                ):
                    return True

    return False


def _invert_value(value: GameValue) -> GameValue:
    if value == "win":
        return "loss"
    if value == "loss":
        return "win"
    return value


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


def _validate_max_depth(
    max_depth: int | None,
) -> None:
    if max_depth is None:
        return
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise ValueError("max_depth must be None or a non-negative integer")
    if max_depth < 0:
        raise ValueError("max_depth must be None or a non-negative integer")


def _terminal_value(
    board: list[int],
    mark: int,
) -> GameValue | None:
    previous_mark = 3 - mark

    if _is_win(board, previous_mark):
        return "loss"

    if not _legal_columns(board):
        return "draw"

    return None


def solve_optimized_position(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> PositionSolution:
    _validate_max_depth(max_depth)

    terminal = _terminal_value(board, mark)
    if terminal is not None:
        return PositionSolution(
            value=terminal,
            moves=(),
            complete=True,
        )

    legal = _legal_columns(board)

    if max_depth == 0:
        moves = tuple(
            MoveAnalysis(column, "unknown")
            for column in legal
        )
        return PositionSolution(
            value="unknown",
            moves=moves,
            complete=False,
        )

    raise NotImplementedError(
        "Recursive optimized search is introduced in Task 3"
    )
