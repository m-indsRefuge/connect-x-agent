from dataclasses import FrozenInstanceError

import pytest

from connect_x_agent.search import (
    MoveAnalysis,
    PositionSolution,
    _aggregate_values,
    solve_position,
)


def test_solver_result_contracts_are_immutable_and_structural() -> None:
    move = MoveAnalysis(column=3, value="win")
    solution = PositionSolution(
        value="win",
        moves=(move,),
        complete=True,
    )

    assert move == MoveAnalysis(column=3, value="win")
    assert solution == PositionSolution(
        value="win",
        moves=(MoveAnalysis(column=3, value="win"),),
        complete=True,
    )

    with pytest.raises(FrozenInstanceError):
        move.column = 4  # type: ignore[misc]


def test_value_aggregation_preserves_partial_proof_semantics() -> None:
    assert _aggregate_values(("loss", "win", "unknown")) == "win"
    assert _aggregate_values(("loss", "draw", "unknown")) == "unknown"
    assert _aggregate_values(("loss", "loss", "draw")) == "draw"
    assert _aggregate_values(("loss", "loss", "loss")) == "loss"


def test_zero_depth_classifies_every_legal_root_move_as_unknown() -> None:
    solution = solve_position(
        board=[0] * 9,
        mark=1,
        columns=3,
        rows=3,
        inarow=3,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="unknown",
        moves=(
            MoveAnalysis(0, "unknown"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "unknown"),
        ),
        complete=False,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"columns": 0}, "columns"),
        ({"rows": 0}, "rows"),
        ({"inarow": 0}, "inarow"),
        ({"mark": 3}, "mark"),
        ({"max_depth": -1}, "max_depth"),
    ],
)
def test_solver_rejects_invalid_scalar_inputs(
    kwargs: dict[str, int],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "board": [0] * 9,
        "mark": 1,
        "columns": 3,
        "rows": 3,
        "inarow": 3,
        "max_depth": 0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        solve_position(**arguments)  # type: ignore[arg-type]


def test_solver_rejects_wrong_board_length() -> None:
    with pytest.raises(ValueError, match="board length"):
        solve_position(
            [0] * 8,
            1,
            3,
            3,
            3,
            max_depth=0,
        )


def test_solver_rejects_invalid_cell_value() -> None:
    board = [0] * 9
    board[-1] = 7

    with pytest.raises(ValueError, match="cell"):
        solve_position(
            board,
            1,
            3,
            3,
            3,
            max_depth=0,
        )


def test_solver_rejects_floating_checker() -> None:
    board = [0] * 9
    board[0] = 1

    with pytest.raises(ValueError, match="gravity"):
        solve_position(
            board,
            2,
            3,
            3,
            3,
            max_depth=0,
        )


def test_solver_rejects_turn_count_mismatch() -> None:
    board = [0] * 9
    board[6:9] = [1, 1, 0]

    with pytest.raises(ValueError, match="turn"):
        solve_position(
            board,
            1,
            3,
            3,
            3,
            max_depth=0,
        )


def test_terminal_previous_player_win_is_exact_loss() -> None:
    board = [
        0, 0, 2, 0,
        1, 1, 1, 2,
    ]

    solution = solve_position(
        board,
        mark=2,
        columns=4,
        rows=2,
        inarow=3,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="loss",
        moves=(),
        complete=True,
    )


def test_terminal_full_board_without_winner_is_exact_draw() -> None:
    solution = solve_position(
        board=[1, 2],
        mark=1,
        columns=2,
        rows=1,
        inarow=2,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="draw",
        moves=(),
        complete=True,
    )


def test_solver_rejects_simultaneous_winners() -> None:
    board = [
        1, 2, 0, 0,
        1, 2, 0, 0,
    ]

    with pytest.raises(
        ValueError,
        match="both players",
    ):
        solve_position(
            board,
            mark=1,
            columns=4,
            rows=2,
            inarow=2,
            max_depth=0,
        )


def test_solver_rejects_winner_that_is_not_previous_player() -> None:
    board = [
        0, 2, 2, 0,
        1, 1, 1, 2,
    ]

    with pytest.raises(
        ValueError,
        match="previous player",
    ):
        solve_position(
            board,
            mark=1,
            columns=4,
            rows=2,
            inarow=3,
            max_depth=0,
        )


def test_solver_rejects_board_requiring_play_after_an_earlier_win() -> None:
    board = [
        1, 2, 2,
        1, 2, 2,
        1, 1, 1,
    ]

    with pytest.raises(
        ValueError,
        match="last move",
    ):
        solve_position(
            board,
            mark=2,
            columns=3,
            rows=3,
            inarow=3,
            max_depth=0,
        )


def test_exhaustive_small_board_is_a_complete_draw() -> None:
    solution = solve_position(
        board=[0, 0],
        mark=1,
        columns=2,
        rows=1,
        inarow=2,
    )

    assert solution == PositionSolution(
        value="draw",
        moves=(
            MoveAnalysis(0, "draw"),
            MoveAnalysis(1, "draw"),
        ),
        complete=True,
    )


def test_exhaustive_solver_classifies_mixed_root_values_independently() -> None:
    board = [
        0, 0, 0,
        0, 2, 2,
        0, 1, 1,
    ]

    solution = solve_position(
        board,
        mark=1,
        columns=3,
        rows=3,
        inarow=3,
    )

    assert solution == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(0, "win"),
            MoveAnalysis(1, "loss"),
            MoveAnalysis(2, "draw"),
        ),
        complete=True,
    )


def test_exhaustive_solver_preserves_multiple_winning_moves() -> None:
    board = [
        0, 0, 0,
        0, 0, 2,
        0, 0, 1,
    ]

    solution = solve_position(
        board,
        mark=1,
        columns=3,
        rows=3,
        inarow=3,
    )

    assert solution == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(0, "win"),
            MoveAnalysis(1, "win"),
            MoveAnalysis(2, "draw"),
        ),
        complete=True,
    )


def test_exhaustive_solver_proves_forced_loss() -> None:
    board = [
        0, 0, 1,
        0, 1, 2,
        0, 2, 1,
    ]

    solution = solve_position(
        board,
        mark=2,
        columns=3,
        rows=3,
        inarow=3,
    )

    assert solution == PositionSolution(
        value="loss",
        moves=(
            MoveAnalysis(0, "loss"),
            MoveAnalysis(1, "loss"),
        ),
        complete=True,
    )


def test_exhaustive_solver_solves_late_standard_board_exactly() -> None:
    board = [
        2, 1, 0, 1, 0, 1, 0,
        2, 1, 0, 2, 0, 2, 2,
        1, 1, 0, 1, 1, 1, 2,
        2, 2, 2, 1, 2, 1, 1,
        2, 1, 2, 2, 1, 1, 2,
        2, 1, 1, 1, 2, 2, 2,
    ]

    solution = solve_position(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )

    assert solution == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(2, "win"),
            MoveAnalysis(4, "win"),
            MoveAnalysis(6, "loss"),
        ),
        complete=True,
    )


def test_bounded_search_preserves_proven_win_with_unknown_alternatives() -> None:
    board = [0] * 42
    board[35:42] = [1, 1, 1, 0, 2, 2, 2]

    solution = solve_position(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
        max_depth=1,
    )

    assert solution == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(0, "unknown"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "unknown"),
            MoveAnalysis(3, "win"),
            MoveAnalysis(4, "unknown"),
            MoveAnalysis(5, "unknown"),
            MoveAnalysis(6, "unknown"),
        ),
        complete=False,
    )


def test_solver_is_deterministic_for_identical_inputs() -> None:
    board = [
        0, 0, 0,
        0, 2, 2,
        0, 1, 1,
    ]

    first = solve_position(
        board,
        1,
        3,
        3,
        3,
        max_depth=4,
    )
    second = solve_position(
        board,
        1,
        3,
        3,
        3,
        max_depth=4,
    )

    assert first == second
    assert tuple(move.column for move in first.moves) == (
        0,
        1,
        2,
    )
