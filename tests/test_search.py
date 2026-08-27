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
