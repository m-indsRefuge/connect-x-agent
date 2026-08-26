import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from connect_x_agent.study.records import (
    EpisodeRecord,
    PlyRecord,
    PositionFeatures,
    episode_to_dict,
    write_jsonl,
)


def example_features() -> PositionFeatures:
    return PositionFeatures(
        legal_columns=(0, 1),
        p1_viable_windows=3,
        p2_viable_windows=2,
        p1_playable_threats=1,
        p2_playable_threats=0,
        p1_latent_threats=2,
        p2_latent_threats=1,
        p1_winning_columns=(1,),
        p2_winning_columns=(),
        p1_fork_moves=(0,),
        p2_fork_moves=(),
        p1_surviving_replies=(0, 1),
        p2_surviving_replies=(1,),
    )


def example_episode() -> EpisodeRecord:
    features = example_features()
    return EpisodeRecord(
        episode_id=7,
        opponent="random",
        candidate_mark=2,
        opening_column=1,
        plies=(
            PlyRecord(
                ply=1,
                mark=1,
                board_before=(0, 0, 0, 0),
                action=1,
                board_after=(0, 0, 0, 1),
                features_before=features,
            ),
        ),
        winner=2,
        candidate_reward=1,
        candidate_result="win",
    )


def test_study_records_are_immutable() -> None:
    record = example_episode()

    with pytest.raises(FrozenInstanceError):
        record.candidate_result = "loss"

    with pytest.raises(FrozenInstanceError):
        record.plies[0].action = 0

    with pytest.raises(FrozenInstanceError):
        record.plies[0].features_before.legal_columns = ()


def test_episode_serialization_preserves_all_fields_as_json_values() -> None:
    assert episode_to_dict(example_episode()) == {
        "episode_id": 7,
        "opponent": "random",
        "candidate_mark": 2,
        "opening_column": 1,
        "plies": [
            {
                "ply": 1,
                "mark": 1,
                "board_before": [0, 0, 0, 0],
                "action": 1,
                "board_after": [0, 0, 0, 1],
                "features_before": {
                    "legal_columns": [0, 1],
                    "p1_viable_windows": 3,
                    "p2_viable_windows": 2,
                    "p1_playable_threats": 1,
                    "p2_playable_threats": 0,
                    "p1_latent_threats": 2,
                    "p2_latent_threats": 1,
                    "p1_winning_columns": [1],
                    "p2_winning_columns": [],
                    "p1_fork_moves": [0],
                    "p2_fork_moves": [],
                    "p1_surviving_replies": [0, 1],
                    "p2_surviving_replies": [1],
                },
            },
        ],
        "winner": 2,
        "candidate_reward": 1,
        "candidate_result": "win",
    }


def test_episode_serialization_is_deterministic_for_tuple_board_states() -> None:
    episode = example_episode()

    assert episode_to_dict(episode) == episode_to_dict(episode)
    assert episode_to_dict(episode)["plies"][0]["board_before"] == [0, 0, 0, 0]
    assert episode_to_dict(episode)["plies"][0]["board_after"] == [0, 0, 0, 1]


def test_write_jsonl_creates_parent_and_writes_parseable_episode_lines(
    tmp_path: Path,
) -> None:
    first_episode = example_episode()
    second_episode = replace(
        first_episode,
        episode_id=8,
        winner=None,
        candidate_reward=0,
        candidate_result="draw",
    )
    path = tmp_path / "nested" / "episodes.jsonl"

    write_jsonl((first_episode, second_episode), path)

    lines = path.read_text(encoding="utf-8").splitlines()

    assert path.parent.is_dir()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [
        episode_to_dict(first_episode),
        episode_to_dict(second_episode),
    ]
