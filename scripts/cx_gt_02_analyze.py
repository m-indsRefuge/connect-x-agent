import json
from argparse import ArgumentParser
from pathlib import Path

from connect_x_agent.study.analysis import analyze_episodes, report_to_dict
from connect_x_agent.study.loading import load_jsonl


def main() -> None:
    parser = ArgumentParser(description="Analyze recorded CX-GT-02 Player-2 episodes")
    parser.add_argument("input", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    episodes = load_jsonl(args.input)
    report = analyze_episodes(episodes)
    payload = report_to_dict(report)

    print(f"episodes: {report.corpus.episodes}")
    print(f"wins: {report.corpus.wins}")
    print(f"draws: {report.corpus.draws}")
    print(f"losses: {report.corpus.losses}")
    print(f"failures: {report.corpus.failures}")
    print(f"unique trajectories: {report.corpus.unique_trajectories}")
    print(f"duplicate trajectories: {report.corpus.duplicate_trajectories}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
