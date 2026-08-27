from dataclasses import FrozenInstanceError

import pytest

from connect_x_agent.search import (
    MoveAnalysis,
    PositionSolution,
    _aggregate_values,
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
