import runpy
from pathlib import Path

import pytest

from connect_x_agent.optimized_search import (
    BOARD_SIZE,
    COLUMNS,
    INAROW,
    ROWS,
    SearchStats,
    _cache_key,
    solve_optimized_position,
    solve_optimized_position_with_stats,
)
from connect_x_agent.search import MoveAnalysis, PositionSolution

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


@pytest.mark.parametrize(
    ("board", "mark"),
    [
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
    ],
    ids=[
        "wrong-board-length",
        "invalid-cell",
        "floating-piece",
        "turn-count-mismatch",
        "simultaneous-winners",
        "winner-not-previous-mover",
        "winner-not-created-by-legal-last-move",
    ],
)
def test_invalid_positions_raise_value_error(
    board: list[int],
    mark: int,
) -> None:
    with pytest.raises(ValueError):
        solve_optimized_position(
            board,
            mark=mark,
            max_depth=0,
        )


@pytest.mark.parametrize("mark", [0, 3])
def test_invalid_mark_raises_value_error(mark: int) -> None:
    with pytest.raises(ValueError):
        solve_optimized_position(
            [0] * 42,
            mark=mark,
            max_depth=0,
        )


@pytest.mark.parametrize("max_depth", [-1, -2])
def test_negative_depth_raises_value_error(max_depth: int) -> None:
    with pytest.raises(ValueError):
        solve_optimized_position(
            [0] * 42,
            mark=1,
            max_depth=max_depth,
        )


def test_cache_key_isolates_depth_and_player() -> None:
    board = [0] * 42

    depth_four = _cache_key(
        board,
        mark=1,
        remaining_depth=4,
    )
    depth_six = _cache_key(
        board,
        mark=1,
        remaining_depth=6,
    )
    other_player = _cache_key(
        board,
        mark=2,
        remaining_depth=4,
    )
    exhaustive = _cache_key(
        board,
        mark=1,
        remaining_depth=None,
    )

    assert depth_four != depth_six
    assert depth_four != other_player
    assert depth_four != exhaustive


def test_transposition_rich_search_records_cache_hits() -> None:
    solution, stats = solve_optimized_position_with_stats(
        [0] * 42,
        mark=1,
        max_depth=4,
    )

    assert solution.value == "unknown"
    assert isinstance(stats, SearchStats)
    assert stats.nodes_visited > 0
    assert stats.cache_hits > 0
    assert stats.cache_misses > 0
    assert stats.cache_entries > 0
    assert stats.cache_entries <= stats.cache_misses


def test_transposition_table_is_fresh_per_solver_call() -> None:
    first_solution, first_stats = solve_optimized_position_with_stats(
        [0] * 42,
        mark=1,
        max_depth=4,
    )
    second_solution, second_stats = solve_optimized_position_with_stats(
        [0] * 42,
        mark=1,
        max_depth=4,
    )

    assert first_solution == second_solution
    assert first_stats == second_stats


def test_production_api_matches_diagnostic_api() -> None:
    board = [0] * 42

    normal = solve_optimized_position(
        board,
        mark=1,
        max_depth=4,
    )
    diagnostic, _ = solve_optimized_position_with_stats(
        board,
        mark=1,
        max_depth=4,
    )

    assert normal == diagnostic


def test_cx06a_cache_probe_declares_fixed_frontier_positions() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cx06a_cache_probe.py"
    )

    namespace = runpy.run_path(
        str(script),
        run_name="cx06a_probe_test",
    )

    positions = namespace["FRONTIER_POSITIONS"]

    assert tuple(
        position.name
        for position in positions
    ) == (
        "empty-opening",
        "game1-danger-ply20",
        "negamax-danger-ply17",
        "hybrid-pre-loss-ply15",
    )
    assert namespace["DEFAULT_TRIALS"] == 3
