from typing import Any


def legal_columns(board: list[int], columns: int) -> list[int]:
    return [
        column
        for column in range(columns)
        if board[column] == 0
    ]


def drop_piece(
    board: list[int],
    column: int,
    mark: int,
    columns: int,
    rows: int,
) -> list[int]:
    result = board.copy()

    for row in range(rows - 1, -1, -1):
        index = row * columns + column

        if result[index] == 0:
            result[index] = mark
            return result

    raise ValueError(f"Column {column} is full")


def is_win(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> bool:
    directions = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    )

    for row in range(rows):
        for column in range(columns):
            if board[row * columns + column] != mark:
                continue

            for row_step, column_step in directions:
                end_row = row + (inarow - 1) * row_step
                end_column = column + (inarow - 1) * column_step

                if not (
                    0 <= end_row < rows
                    and 0 <= end_column < columns
                ):
                    continue

                if all(
                    board[
                        (row + offset * row_step) * columns
                        + column
                        + offset * column_step
                    ]
                    == mark
                    for offset in range(inarow)
                ):
                    return True

    return False


def tactical_agent(
    observation: Any,
    configuration: Any,
) -> int:
    board = list(observation.board)
    mark = int(observation.mark)

    columns = int(configuration.columns)
    rows = int(configuration.rows)
    inarow = int(configuration.inarow)

    legal = legal_columns(board, columns)

    if not legal:
        raise RuntimeError("No legal moves available")

    # Win immediately if possible.
    for column in legal:
        candidate = drop_piece(
            board,
            column,
            mark,
            columns,
            rows,
        )

        if is_win(
            candidate,
            mark,
            columns,
            rows,
            inarow,
        ):
            return column

    # Otherwise block an immediate opponent win.
    opponent = 2 if mark == 1 else 1

    for column in legal:
        candidate = drop_piece(
            board,
            column,
            opponent,
            columns,
            rows,
        )

        if is_win(
            candidate,
            opponent,
            columns,
            rows,
            inarow,
        ):
            return column

    # Preserve CX-01 behavior when no tactic exists.
    return legal[0]
