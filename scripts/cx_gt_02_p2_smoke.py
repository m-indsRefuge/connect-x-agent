from pathlib import Path

from connect_x_agent.study.records import EpisodeRecord, write_jsonl
from connect_x_agent.study.runner import run_p2_study
from connect_x_agent.tactical import tactical_agent


def summarize_episodes(
    episodes: tuple[EpisodeRecord, ...],
) -> dict[str, int | float]:
    games = len(episodes)
    wins = sum(episode.candidate_result == "win" for episode in episodes)
    draws = sum(episode.candidate_result == "draw" for episode in episodes)
    losses = sum(episode.candidate_result == "loss" for episode in episodes)
    failures = sum(episode.candidate_result == "failure" for episode in episodes)
    average_plies = (
        sum(len(episode.plies) for episode in episodes) / games if games else 0.0
    )

    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "failures": failures,
        "win_rate": wins / games if games else 0.0,
        "average_plies": average_plies,
    }


def main() -> None:
    episodes = run_p2_study(
        candidate=tactical_agent,
        opponent="random",
        games=100,
        opponent_name="random",
    )
    output_path = Path("evidence/cx_gt_02/raw/p2_random_smoke.jsonl")
    write_jsonl(episodes, output_path)
    summary = summarize_episodes(episodes)

    print(f"games: {summary['games']}")
    print(f"wins: {summary['wins']}")
    print(f"draws: {summary['draws']}")
    print(f"losses: {summary['losses']}")
    print(f"failures: {summary['failures']}")
    print(f"win rate: {summary['win_rate']:.3f}")
    print(f"average plies: {summary['average_plies']:.2f}")


if __name__ == "__main__":
    main()
