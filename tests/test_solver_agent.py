from types import SimpleNamespace

from connect_x_agent.search import MoveAnalysis, PositionSolution
from connect_x_agent.solver_agent import (
    SOLVER_DEPTH,
    _choose_hybrid_move,
    _choose_solver_move,
    hybrid_solver_agent,
    solver_agent,
)


def test_solver_depth_is_fixed_at_four() -> None:
    assert SOLVER_DEPTH == 4


def test_pure_policy_prefers_win_over_all_other_values() -> None:
    solution = PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(0, "loss"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "draw"),
            MoveAnalysis(3, "win"),
            MoveAnalysis(4, "win"),
        ),
        complete=False,
    )

    assert _choose_solver_move(solution) == 3


def test_pure_policy_uses_leftmost_move_within_same_value() -> None:
    solution = PositionSolution(
        value="unknown",
        moves=(
            MoveAnalysis(0, "loss"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "unknown"),
        ),
        complete=False,
    )

    assert _choose_solver_move(solution) == 1


def test_pure_policy_prefers_unknown_over_proven_loss() -> None:
    solution = PositionSolution(
        value="unknown",
        moves=(
            MoveAnalysis(0, "loss"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "loss"),
        ),
        complete=False,
    )

    assert _choose_solver_move(solution) == 1


def test_hybrid_policy_prefers_proven_draw_before_tactical_unknown() -> None:
    solution = PositionSolution(
        value="draw",
        moves=(
            MoveAnalysis(0, "unknown"),
            MoveAnalysis(1, "draw"),
            MoveAnalysis(2, "loss"),
        ),
        complete=False,
    )

    assert _choose_hybrid_move(
        solution,
        tactical_column=0,
    ) == 1


def test_hybrid_policy_uses_tactical_move_when_it_is_unresolved() -> None:
    solution = PositionSolution(
        value="unknown",
        moves=(
            MoveAnalysis(0, "unknown"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "loss"),
        ),
        complete=False,
    )

    assert _choose_hybrid_move(
        solution,
        tactical_column=1,
    ) == 1


def test_hybrid_policy_refuses_tactical_move_proven_losing() -> None:
    solution = PositionSolution(
        value="unknown",
        moves=(
            MoveAnalysis(0, "loss"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "unknown"),
        ),
        complete=False,
    )

    assert _choose_hybrid_move(
        solution,
        tactical_column=0,
    ) == 1


def test_hybrid_policy_uses_leftmost_loss_when_every_move_is_loss() -> None:
    solution = PositionSolution(
        value="loss",
        moves=(
            MoveAnalysis(0, "loss"),
            MoveAnalysis(2, "loss"),
        ),
        complete=True,
    )

    assert _choose_hybrid_move(
        solution,
        tactical_column=2,
    ) == 0


def test_playable_agents_take_immediate_proven_win() -> None:
    board = [0] * 42
    board[35:42] = [1, 1, 1, 0, 2, 2, 2]

    observation = SimpleNamespace(
        board=board,
        mark=1,
    )
    configuration = SimpleNamespace(
        columns=7,
        rows=6,
        inarow=4,
    )

    assert solver_agent(
        observation,
        configuration,
    ) == 3

    assert hybrid_solver_agent(
        observation,
        configuration,
    ) == 3
