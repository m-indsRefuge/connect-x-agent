from typing import Any

from connect_x_agent.tactical import (
    drop_piece,
    is_win,
    legal_columns,
)


def lookahead_agent(
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

    # Preserve CX-02: take an immediate win.
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

    opponent = 2 if mark == 1 else 1

    # CX-03:
    # Reject moves that allow an immediate opponent win.
    safe_moves: list[int] = []

    for column in legal:
        candidate = drop_piece(
            board,
            column,
            mark,
            columns,
            rows,
        )

        opponent_can_win = False

        for reply in legal_columns(candidate, columns):
            response = drop_piece(
                candidate,
                reply,
                opponent,
                columns,
                rows,
            )

            if is_win(
                response,
                opponent,
                columns,
                rows,
                inarow,
            ):
                opponent_can_win = True
                break

        if not opponent_can_win:
            safe_moves.append(column)

    if safe_moves:
        return safe_moves[0]

    # Every move loses immediately; remain deterministic.
    return legal[0]
