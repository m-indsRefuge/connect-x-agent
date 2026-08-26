import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from connect_x_agent.study.records import EpisodeRecord
from connect_x_agent.study.runner import run_p2_study


def first_legal_agent(observation: Any, configuration: Any) -> int:
    return next(
        column
        for column in range(int(configuration.columns))
        if observation.board[column] == 0
    )


def test_p2_runner_records_ordered_complete_games_with_coherent_plies() -> None:
    candidate_marks: list[int] = []

    def candidate(observation: Any, configuration: Any) -> int:
        candidate_marks.append(int(observation.mark))
        return first_legal_agent(observation, configuration)

    episodes = run_p2_study(
        candidate=candidate,
        opponent=first_legal_agent,
        games=2,
        columns=4,
        rows=3,
        inarow=3,
        opponent_name="first-legal",
    )

    assert tuple(episode.episode_id for episode in episodes) == (0, 1)
    assert candidate_marks and set(candidate_marks) == {2}

    for episode in episodes:
        expected_outcomes = {
            1: ("win", 2),
            0: ("draw", None),
            -1: ("loss", 1),
        }

        assert episode.candidate_mark == 2
        assert isinstance(episode, EpisodeRecord)
        assert episode.candidate_reward in expected_outcomes
        assert (episode.candidate_result, episode.winner) == expected_outcomes[
            episode.candidate_reward
        ]
        assert getattr(episode, "columns", None) == 4
        assert getattr(episode, "rows", None) == 3
        assert getattr(episode, "inarow", None) == 3
        assert getattr(episode, "failure_kind", None) is None
        assert getattr(episode, "failure_reason", None) is None
        assert episode.opening_column == episode.plies[0].action
        assert episode.plies[0].mark == 1
        assert {ply.mark for ply in episode.plies} == {1, 2}

        for previous, current in zip(episode.plies, episode.plies[1:]):
            assert previous.board_after == current.board_before

        for ply in episode.plies:
            changed_cells = [
                cell
                for cell, (before, after) in enumerate(
                    zip(ply.board_before, ply.board_after, strict=True)
                )
                if before != after
            ]

            assert 0 <= ply.action < 4
            assert ply.board_before[ply.action] == 0
            assert len(changed_cells) == 1
            assert changed_cells[0] % 4 == ply.action
            assert ply.board_before[changed_cells[0]] == 0
            assert ply.board_after[changed_cells[0]] == ply.mark
            assert ply.features_before.legal_columns == tuple(
                column for column in range(4) if ply.board_before[column] == 0
            )


def test_p2_runner_records_invalid_candidate_runtime_provenance() -> None:
    def invalid_candidate(observation: Any, configuration: Any) -> int:
        return int(configuration.columns)

    episodes = run_p2_study(
        candidate=invalid_candidate,
        opponent=first_legal_agent,
        games=2,
        columns=4,
        rows=3,
        inarow=3,
        opponent_name="first-legal",
    )

    assert tuple(episode.episode_id for episode in episodes) == (0, 1)
    assert all(episode.candidate_mark == 2 for episode in episodes)
    assert all(episode.candidate_result == "failure" for episode in episodes)
    assert all(episode.candidate_reward is None for episode in episodes)
    assert all(episode.winner is None for episode in episodes)
    assert all(episode.opening_column == 0 for episode in episodes)
    assert all(
        getattr(episode, "failure_kind", None) == "candidate_runtime"
        for episode in episodes
    )
    assert all(
        "INVALID" in str(getattr(episode, "failure_reason", ""))
        for episode in episodes
    )


def test_p2_runner_does_not_hide_study_pipeline_decode_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedEnvironment:
        def run(self, agents: list[Any]) -> list[list[Any]]:
            del agents
            return []

    fake_kaggle = ModuleType("kaggle_environments")
    fake_kaggle.make = lambda *args, **kwargs: MalformedEnvironment()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_environments", fake_kaggle)

    with pytest.raises(ValueError, match="history is empty"):
        run_p2_study(
            candidate=first_legal_agent,
            opponent=first_legal_agent,
            games=1,
            columns=4,
            rows=3,
            inarow=3,
            opponent_name="malformed",
        )


def test_smoke_script_summarizes_study_outcomes() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "cx_gt_02_p2_smoke.py"
    module = runpy.run_path(script_path)
    summarize_episodes = module["summarize_episodes"]
    episodes = (
        EpisodeRecord(0, "random", 2, None, (), 2, 1, "win"),
        EpisodeRecord(1, "random", 2, None, (), None, 0, "draw"),
        EpisodeRecord(2, "random", 2, None, (), 1, -1, "loss"),
        EpisodeRecord(3, "random", 2, None, (), None, None, "failure"),
    )

    assert summarize_episodes(episodes) == {
        "games": 4,
        "wins": 1,
        "draws": 1,
        "losses": 1,
        "failures": 1,
        "win_rate": 0.25,
        "average_plies": 0.0,
    }
