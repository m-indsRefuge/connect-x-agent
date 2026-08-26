from dataclasses import replace
from pathlib import Path

from connect_x_agent.study.loading import load_jsonl
from connect_x_agent.study.records import (
    EpisodeRecord,
    PlyRecord,
    PositionFeatures,
    write_jsonl,
)


def example_features() -> PositionFeatures:
    return PositionFeatures(
        legal_columns=(0, 1, 2, 3),
        p1_viable_windows=14,
        p2_viable_windows=14,
        p1_playable_threats=0,
        p2_playable_threats=0,
        p1_latent_threats=0,
        p2_latent_threats=0,
        p1_winning_columns=(),
        p2_winning_columns=(),
        p1_fork_moves=(),
        p2_fork_moves=(),
        p1_surviving_replies=(0, 1, 2, 3),
        p2_surviving_replies=(0, 1, 2, 3),
    )


def example_episode(episode_id: int = 1) -> EpisodeRecord:
    features = example_features()
    return EpisodeRecord(
        episode_id=episode_id,
        opponent="random",
        candidate_mark=2,
        opening_column=1,
        plies=(
            PlyRecord(
                ply=1,
                mark=1,
                board_before=(0,) * 12,
                action=1,
                board_after=(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                features_before=features,
            ),
        ),
        winner=2,
        candidate_reward=1,
        candidate_result="win",
        columns=4,
        rows=3,
        inarow=3,
    )


def test_load_jsonl_round_trips_typed_episode(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    expected = example_episode()
    write_jsonl((expected,), path)

    assert load_jsonl(path) == (expected,)


def test_load_jsonl_preserves_episode_order(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    first = example_episode(1)
    second = replace(first, episode_id=2)
    write_jsonl((first, second), path)

    assert load_jsonl(path) == (first, second)
