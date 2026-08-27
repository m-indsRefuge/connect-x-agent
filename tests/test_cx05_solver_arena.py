import runpy
from pathlib import Path

from connect_x_agent.solver_agent import (
    hybrid_solver_agent,
    solver_agent,
)


def test_cx05_smoke_arena_wires_both_approved_variants() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cx05_solver_arena.py"
    )

    namespace = runpy.run_path(
        str(script_path)
    )

    assert namespace["GAMES_PER_SEAT"] == 10

    assert tuple(
        agent
        for _, agent in namespace["ARENA_VARIANTS"]
    ) == (
        solver_agent,
        hybrid_solver_agent,
    )
