from connect_x_agent.search import MoveAnalysis, PositionSolution
from connect_x_agent.optimized_search import (
    BOARD_SIZE,
    COLUMNS,
    INAROW,
    ROWS,
    solve_optimized_position,
)


FULL_DRAW_BOARD = [
    1, 1, 2, 2, 2, 1, 1,
    2, 2, 1, 1, 2, 2, 2,
    1, 1, 2, 2, 1, 2, 1,
    2, 2, 2, 1, 1, 2, 1,
    2, 1, 2, 1, 2, 1, 1,
    1, 2, 1, 1, 2, 1, 2,
]


def test_optimized_search_is_fixed_to_standard_connect_four() -> None:
    assert (COLUMNS, ROWS, INAROW, BOARD_SIZE) == (7, 6, 4, 42)


def test_zero_depth_returns_unknown_for_every_legal_root_move() -> None:
    solution = solve_optimized_position(
        [0] * 42,
        mark=1,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="unknown",
        moves=tuple(
            MoveAnalysis(column, "unknown")
            for column in range(7)
        ),
        complete=False,
    )


def test_terminal_previous_player_win_precedes_zero_depth() -> None:
    board = [0] * 35 + [1, 1, 1, 1, 2, 2, 2]

    solution = solve_optimized_position(
        board,
        mark=2,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="loss",
        moves=(),
        complete=True,
    )


def test_full_no_win_board_is_terminal_draw() -> None:
    solution = solve_optimized_position(
        FULL_DRAW_BOARD,
        mark=1,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="draw",
        moves=(),
        complete=True,
    )


def test_solver_does_not_mutate_input_board() -> None:
    board = [0] * 42
    before = board.copy()

    solve_optimized_position(
        board,
        mark=1,
        max_depth=0,
    )

    assert board == before
