from connect_x_agent.arena import ArenaReport, run_arena
from connect_x_agent.solver_agent import (
    hybrid_solver_agent,
    solver_agent,
)

GAMES_PER_SEAT = 10

ARENA_VARIANTS = (
    ("PURE CX-04 DEPTH-4", solver_agent),
    ("HYBRID CX-04 + CX-02", hybrid_solver_agent),
)


def _percentage(
    value: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return value / total * 100


def _print_report(
    label: str,
    report: ArenaReport,
) -> None:
    print()
    print(f"=== {label} ===")

    print()
    print("PLAYER 1")
    print(f"Games:    {report.player_one.games}")
    print(f"Wins:     {report.player_one.wins}")
    print(f"Draws:    {report.player_one.draws}")
    print(f"Losses:   {report.player_one.losses}")
    print(f"Failures: {report.player_one.failures}")
    print(
        "Win rate:",
        f"{_percentage(report.player_one.wins, report.player_one.games):.1f}%",
    )

    print()
    print("PLAYER 2")
    print(f"Games:    {report.player_two.games}")
    print(f"Wins:     {report.player_two.wins}")
    print(f"Draws:    {report.player_two.draws}")
    print(f"Losses:   {report.player_two.losses}")
    print(f"Failures: {report.player_two.failures}")
    print(
        "Win rate:",
        f"{_percentage(report.player_two.wins, report.player_two.games):.1f}%",
    )

    print()
    print("COMBINED")
    print(f"Games:    {report.games}")
    print(f"Wins:     {report.wins}")
    print(f"Draws:    {report.draws}")
    print(f"Losses:   {report.losses}")
    print(f"Failures: {report.failures}")
    print(f"Win rate: {report.win_rate * 100:.1f}%")


def main() -> None:
    print("=== CX-05 SOLVER ARENA SMOKE ===")
    print("Opponent: Kaggle random")
    print("Solver depth: 4")
    print(f"Games per seat: {GAMES_PER_SEAT}")

    for label, agent in ARENA_VARIANTS:
        report = run_arena(
            candidate=agent,
            opponent="random",
            games_per_seat=GAMES_PER_SEAT,
        )

        _print_report(
            label,
            report,
        )

        if report.failures:
            raise RuntimeError(
                f"{label} recorded "
                f"{report.failures} failed games"
            )


if __name__ == "__main__":
    main()
