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


def _validate_gravity(board: list[int]) -> None:
    for column in range(COLUMNS):
        seen_empty = False

        for row in range(ROWS - 1, -1, -1):
            value = board[row * COLUMNS + column]

            if value == 0:
                seen_empty = True
            elif seen_empty:
                raise ValueError("Board violates gravity")


def _topmost_occupied_index(
    board: list[int],
    column: int,
) -> int | None:
    for row in range(ROWS):
        index = row * COLUMNS + column
        if board[index] != 0:
            return index
    return None


def _winner_has_consistent_last_move(
    board: list[int],
    winner: int,
) -> bool:
    for column in range(COLUMNS):
        index = _topmost_occupied_index(
            board,
            column,
        )

        if index is None or board[index] != winner:
            continue

        predecessor = board.copy()
        predecessor[index] = 0

        if not _is_win(predecessor, winner):
            return True

    return False


def _validate_position(
    board: list[int],
    mark: int,
) -> None:
    if len(board) != BOARD_SIZE:
        raise ValueError(
            f"Board must contain exactly {BOARD_SIZE} cells"
        )

    if any(cell not in (0, 1, 2) for cell in board):
        raise ValueError("Board cells must be 0, 1, or 2")

    if mark not in (1, 2):
        raise ValueError("mark must be 1 or 2")

    _validate_gravity(board)

    player_one = board.count(1)
    player_two = board.count(2)

    if mark == 1 and player_one != player_two:
        raise ValueError(
            "Piece counts are inconsistent with mark 1 to move"
        )

    if mark == 2 and player_one != player_two + 1:
        raise ValueError(
            "Piece counts are inconsistent with mark 2 to move"
        )

    player_one_wins = _is_win(board, 1)
    player_two_wins = _is_win(board, 2)

    if player_one_wins and player_two_wins:
        raise ValueError("Both players cannot be winners")

    winner: int | None = None

    if player_one_wins:
        winner = 1
    elif player_two_wins:
        winner = 2

    if winner is None:
        return

    previous_mark = 3 - mark

    if winner != previous_mark:
        raise ValueError(
            "Winner must be the previous mover"
        )

    if not _winner_has_consistent_last_move(
        board,
        winner,
    ):
        raise ValueError(
            "Winner is inconsistent with a legal last move"
        )


def _validate_max_depth(
    max_depth: int | None,
) -> None:
    if max_depth is None:
        return
    if type(max_depth) is not int:
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
    _validate_position(board, mark)
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
