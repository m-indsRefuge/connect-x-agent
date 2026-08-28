import pytest

from connect_x_agent.optimized_search import (
    solve_optimized_position,
)
from connect_x_agent.search import (
    MoveAnalysis,
    PositionSolution,
    solve_position,
)

INVALID_CASES = (
    ([0] * 41, 1),
    ([0] * 41 + [3], 1),
    ([1] + [0] * 41, 2),
    ([0] * 35 + [1, 0, 0, 0, 0, 0, 0], 1),
    (
        [0] * 28
        + [2, 2, 2, 2, 1, 1, 1]
        + [1, 1, 1, 1, 2, 2, 2],
        1,
    ),
    (
        [0] * 28
        + [0, 0, 0, 0, 2, 0, 0]
        + [1, 1, 1, 1, 2, 2, 2],
        1,
    ),
    (
        [
            0, 0, 2, 0, 0, 0, 0,
            0, 0, 1, 1, 2, 1, 2,
            0, 0, 2, 2, 1, 2, 1,
            0, 1, 1, 1, 2, 1, 1,
            2, 1, 1, 1, 2, 1, 1,
            2, 2, 2, 1, 2, 2, 2,
        ],
        2,
    ),
)

FRONTIER_CASES = (
    (
        "empty",
        [0] * 42,
        1,
    ),
    (
        "game1-danger-ply20",
        [
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 2, 0, 0, 0,
            1, 1, 1, 2, 0, 0, 0,
        ],
        1,
    ),
    (
        "negamax-danger-ply17",
        [
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 1,
            2, 1, 2, 0, 1, 0, 1,
            2, 1, 2, 2, 1, 1, 1,
        ],
        2,
    ),
    (
        "hybrid-pre-loss-ply15",
        [
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0,
            2, 2, 0, 0, 0, 0, 0,
            2, 1, 0, 0, 1, 0, 1,
            2, 2, 1, 0, 1, 0, 1,
        ],
        2,
    ),
)


def _reference_rejects(
    board: list[int],
    mark: int,
) -> bool:
    try:
        solve_position(
            board,
            mark=mark,
            columns=7,
            rows=6,
            inarow=4,
            max_depth=0,
        )
    except ValueError:
        return True
    return False


@pytest.mark.parametrize(("board", "mark"), INVALID_CASES)
def test_validation_accept_reject_parity(
    board: list[int],
    mark: int,
) -> None:
    reference_rejects = _reference_rejects(
        board,
        mark,
    )

    try:
        solve_optimized_position(
            board,
            mark=mark,
            max_depth=0,
        )
    except ValueError:
        optimized_rejects = True
    else:
        optimized_rejects = False

    assert optimized_rejects == reference_rejects
    assert reference_rejects is True


@pytest.mark.parametrize(
    ("name", "board", "mark"),
    FRONTIER_CASES,
    ids=[case[0] for case in FRONTIER_CASES],
)
@pytest.mark.parametrize("max_depth", [0, 1, 4, 6])
def test_bounded_search_matches_cx04(
    name: str,
    board: list[int],
    mark: int,
    max_depth: int,
) -> None:
    del name

    reference = solve_position(
        board,
        mark=mark,
        columns=7,
        rows=6,
        inarow=4,
        max_depth=max_depth,
    )
    optimized = solve_optimized_position(
        board,
        mark=mark,
        max_depth=max_depth,
    )

    assert optimized == reference


def test_depth_one_preserves_proven_win_with_unknown_alternatives() -> None:
    board = [0] * 42
    board[35:42] = [1, 1, 1, 0, 2, 2, 2]

    optimized = solve_optimized_position(
        board,
        mark=1,
        max_depth=1,
    )

    assert optimized == PositionSolution(
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
