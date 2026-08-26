from dataclasses import dataclass


@dataclass(frozen=True)
class Threat:
    window: tuple[int, ...]
    target: int
    playable: bool


def winning_windows(
    columns: int,
    rows: int,
    inarow: int,
) -> list[tuple[int, ...]]:
    directions = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    )
    windows: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for row in range(rows):
        for column in range(columns):
            for row_step, column_step in directions:
                end_row = row + (inarow - 1) * row_step
                end_column = column + (inarow - 1) * column_step

                if not (0 <= end_row < rows and 0 <= end_column < columns):
                    continue

                window = tuple(
                    (row + offset * row_step) * columns
                    + column
                    + offset * column_step
                    for offset in range(inarow)
                )

                if window not in seen:
                    seen.add(window)
                    windows.append(window)

    return windows


def cell_influence(
    columns: int,
    rows: int,
    inarow: int,
) -> list[int]:
    influence = [0] * (columns * rows)

    for window in winning_windows(
        columns=columns,
        rows=rows,
        inarow=inarow,
    ):
        for cell in window:
            influence[cell] += 1

    return influence


def viable_windows(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> list[tuple[int, ...]]:
    return [
        window
        for window in winning_windows(
            columns=columns,
            rows=rows,
            inarow=inarow,
        )
        if all(board[cell] in (0, mark) for cell in window)
    ]


def landing_cells(
    board: list[int],
    columns: int,
    rows: int,
) -> dict[int, int]:
    landing: dict[int, int] = {}

    for column in range(columns):
        for row in range(rows - 1, -1, -1):
            cell = row * columns + column

            if board[cell] == 0:
                landing[column] = cell
                break

    return landing


def threats(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> list[Threat]:
    landing = landing_cells(
        board,
        columns=columns,
        rows=rows,
    )
    found_threats: list[Threat] = []

    for window in winning_windows(
        columns=columns,
        rows=rows,
        inarow=inarow,
    ):
        if not all(board[cell] in (0, mark) for cell in window):
            continue

        if sum(board[cell] == mark for cell in window) != inarow - 1:
            continue

        target = next(cell for cell in window if board[cell] == 0)
        found_threats.append(
            Threat(
                window=window,
                target=target,
                playable=landing.get(target % columns) == target,
            )
        )

    return found_threats


def _board_after_drop(
    board: list[int],
    mark: int,
    column: int,
    columns: int,
    rows: int,
) -> list[int]:
    landing = landing_cells(board, columns, rows)
    next_board = board.copy()
    next_board[landing[column]] = mark
    return next_board


def winning_columns(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> tuple[int, ...]:
    landing = landing_cells(board, columns, rows)
    playable_targets = {
        threat.target
        for threat in threats(board, mark, columns, rows, inarow)
        if threat.playable
    }

    return tuple(
        column for column, target in landing.items() if target in playable_targets
    )


def fork_moves(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> dict[int, tuple[Threat, ...]]:
    immediate_wins = set(winning_columns(board, mark, columns, rows, inarow))
    found_forks: dict[int, tuple[Threat, ...]] = {}

    for column in landing_cells(board, columns, rows):
        board_after_drop = _board_after_drop(board, mark, column, columns, rows)

        if column in immediate_wins:
            continue

        playable_threats = tuple(
            threat
            for threat in threats(board_after_drop, mark, columns, rows, inarow)
            if threat.playable
        )

        if len({threat.target for threat in playable_threats}) >= 2:
            found_forks[column] = playable_threats

    return found_forks


def surviving_replies(
    board: list[int],
    defender_mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> tuple[int, ...]:
    defender_wins = set(
        winning_columns(board, defender_mark, columns, rows, inarow)
    )
    opponent_mark = 3 - defender_mark
    replies: list[int] = []

    for column in landing_cells(board, columns, rows):
        board_after_drop = _board_after_drop(
            board,
            defender_mark,
            column,
            columns,
            rows,
        )

        if column in defender_wins or not winning_columns(
            board_after_drop,
            opponent_mark,
            columns,
            rows,
            inarow,
        ):
            replies.append(column)

    return tuple(replies)
