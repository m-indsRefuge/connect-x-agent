import pytest

from connect_x_agent.optimized_search import (
    solve_optimized_position,
)
from connect_x_agent.search import solve_position

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
