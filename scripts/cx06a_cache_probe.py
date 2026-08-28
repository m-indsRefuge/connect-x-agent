from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable

from connect_x_agent.optimized_search import (
    SearchStats,
    solve_optimized_position_with_stats,
)
from connect_x_agent.search import (
    PositionSolution,
    solve_position,
)


DEFAULT_TRIALS = 3


@dataclass(frozen=True)
class ProbePosition:
    name: str
    board: tuple[int, ...]
    mark: int


FRONTIER_POSITIONS = (
    ProbePosition(
        name="empty-opening",
        board=tuple([0] * 42),
        mark=1,
    ),
    ProbePosition(
        name="game1-danger-ply20",
        board=(
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 2, 0, 0, 0,
            1, 1, 1, 2, 0, 0, 0,
        ),
        mark=1,
    ),
    ProbePosition(
        name="negamax-danger-ply17",
        board=(
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 1,
            2, 1, 2, 0, 1, 0, 1,
            2, 1, 2, 2, 1, 1, 1,
        ),
        mark=2,
    ),
    ProbePosition(
        name="hybrid-pre-loss-ply15",
        board=(
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0,
            2, 2, 0, 0, 0, 0, 0,
            2, 1, 0, 0, 1, 0, 1,
            2, 2, 1, 0, 1, 0, 1,
        ),
        mark=2,
    ),
)


def unknown_count(
    solution: PositionSolution,
) -> int:
    return sum(
        move.value == "unknown"
        for move in solution.moves
    )


def time_call(
    function: Callable[[], PositionSolution],
    trials: int,
) -> tuple[PositionSolution, tuple[float, ...]]:
    elapsed_ms: list[float] = []
    result: PositionSolution | None = None

    for _ in range(trials):
        started = perf_counter()
        result = function()
        elapsed_ms.append(
            (perf_counter() - started) * 1000.0
        )

    if result is None:
        raise RuntimeError("Benchmark executed no trials")

    return result, tuple(elapsed_ms)


def reference_solution(
    position: ProbePosition,
    depth: int,
) -> PositionSolution:
    return solve_position(
        list(position.board),
        mark=position.mark,
        columns=7,
        rows=6,
        inarow=4,
        max_depth=depth,
    )


def optimized_solution(
    position: ProbePosition,
    depth: int,
) -> tuple[PositionSolution, SearchStats]:
    return solve_optimized_position_with_stats(
        list(position.board),
        mark=position.mark,
        max_depth=depth,
    )


def run_reference(
    position: ProbePosition,
    depth: int,
    trials: int,
) -> None:
    solution, elapsed = time_call(
        lambda: reference_solution(
            position,
            depth,
        ),
        trials,
    )

    print("solver:      CX-04 reference")
    print(f"position:    {position.name}")
    print(f"depth:       {depth}")
    print(f"root:        {solution.value.upper()}")
    print(f"unknowns:    {unknown_count(solution)}")
    print(
        "trials ms:   "
        + ", ".join(
            f"{value:.3f}"
            for value in elapsed
        )
    )
    print(f"median ms:   {median(elapsed):.3f}")


def run_optimized(
    position: ProbePosition,
    depth: int,
    trials: int,
) -> None:
    elapsed_ms: list[float] = []
    result: PositionSolution | None = None
    stats: SearchStats | None = None

    for _ in range(trials):
        started = perf_counter()
        result, stats = optimized_solution(
            position,
            depth,
        )
        elapsed_ms.append(
            (perf_counter() - started) * 1000.0
        )

    if result is None or stats is None:
        raise RuntimeError("Benchmark executed no trials")

    print("solver:      CX-06A cached")
    print(f"position:    {position.name}")
    print(f"depth:       {depth}")
    print(f"root:        {result.value.upper()}")
    print(f"unknowns:    {unknown_count(result)}")
    print(
        "trials ms:   "
        + ", ".join(
            f"{value:.3f}"
            for value in elapsed_ms
        )
    )
    print(
        f"median ms:   "
        f"{median(elapsed_ms):.3f}"
    )
    print(f"nodes:       {stats.nodes_visited}")
    print(f"cache hits:  {stats.cache_hits}")
    print(f"cache miss:  {stats.cache_misses}")
    print(f"entries:     {stats.cache_entries}")
    print(
        f"max recurse: "
        f"{stats.max_recursion_depth}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CX-06A transposition-cache benchmark",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=(6, 7),
        default=6,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
    )
    parser.add_argument(
        "--optimized-only",
        action="store_true",
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")

    return args


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("CX-06A TRANSPOSITION CACHE PROBE")
    print(
        f"depth={args.depth} "
        f"trials={args.trials} "
        f"optimized_only={args.optimized_only}"
    )
    print("=" * 72)

    for position in FRONTIER_POSITIONS:
        print()
        print("-" * 72)

        if not args.optimized_only:
            run_reference(
                position,
                args.depth,
                args.trials,
            )
            print()

        run_optimized(
            position,
            args.depth,
            args.trials,
        )

    print()
    print("=" * 72)
    print("CX-06A CACHE PROBE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
