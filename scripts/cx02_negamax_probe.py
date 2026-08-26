from connect_x_agent.arena import run_arena
from connect_x_agent.tactical import tactical_agent

GAMES_PER_SEAT = 50


def percentage(value: int, total: int) -> float:
    if total == 0:
        return 0.0

    return value / total * 100


def main() -> None:
    print("=== CX-02 NEGAMAX PROBE ===")
    print("Candidate: immediate-win + immediate-block")
    print("Opponent: Kaggle negamax")
    print(f"Games per seat: {GAMES_PER_SEAT}")
    print()

    report = run_arena(
        candidate=tactical_agent,
        opponent="negamax",
        games_per_seat=GAMES_PER_SEAT,
    )

    print("=== PLAYER 1 ===")
    print(f"Games:    {report.player_one.games}")
    print(f"Wins:     {report.player_one.wins}")
    print(f"Draws:    {report.player_one.draws}")
    print(f"Losses:   {report.player_one.losses}")
    print(f"Failures: {report.player_one.failures}")
    print(
        "Win rate:",
        f"{percentage(report.player_one.wins, report.player_one.games):.1f}%",
    )

    print()
    print("=== PLAYER 2 ===")
    print(f"Games:    {report.player_two.games}")
    print(f"Wins:     {report.player_two.wins}")
    print(f"Draws:    {report.player_two.draws}")
    print(f"Losses:   {report.player_two.losses}")
    print(f"Failures: {report.player_two.failures}")
    print(
        "Win rate:",
        f"{percentage(report.player_two.wins, report.player_two.games):.1f}%",
    )

    print()
    print("=== COMBINED ===")
    print(f"Games:    {report.games}")
    print(f"Wins:     {report.wins}")
    print(f"Draws:    {report.draws}")
    print(f"Losses:   {report.losses}")
    print(f"Failures: {report.failures}")
    print(f"Win rate: {report.win_rate * 100:.1f}%")

    if report.failures:
        raise RuntimeError(
            f"CX-02 recorded {report.failures} failed games"
        )


if __name__ == "__main__":
    main()
