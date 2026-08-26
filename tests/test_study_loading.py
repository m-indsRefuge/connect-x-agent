import json
from dataclasses import replace
from pathlib import Path

import pytest

from connect_x_agent.study.loading import load_jsonl
from connect_x_agent.study.records import (
    EpisodeRecord,
    PlyRecord,
    PositionFeatures,
    episode_to_dict,
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


def write_payload(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
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


def test_load_jsonl_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON at line 1"):
        load_jsonl(path)


def test_load_jsonl_rejects_blank_record(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="Blank JSONL record at line 1"):
        load_jsonl(path)


@pytest.mark.parametrize(
    ("field", "bad_value", "error_type"),
    [
        ("candidate_result", "victory", ValueError),
        ("failure_kind", "study_decode", ValueError),
        ("columns", "4", TypeError),
        ("rows", 0, ValueError),
        ("inarow", 0, ValueError),
    ],
)
def test_load_jsonl_rejects_invalid_top_level_schema(
    tmp_path: Path,
    field: str,
    bad_value: object,
    error_type: type[Exception],
) -> None:
    path = tmp_path / "episodes.jsonl"
    payload = episode_to_dict(example_episode())
    payload[field] = bad_value
    write_payload(path, payload)

    with pytest.raises(error_type, match=field):
        load_jsonl(path)


def test_load_jsonl_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    payload = episode_to_dict(example_episode())
    del payload["candidate_result"]
    write_payload(path, payload)

    with pytest.raises(ValueError, match="candidate_result"):
        load_jsonl(path)


def test_load_jsonl_rejects_board_length_not_matching_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episodes.jsonl"
    payload = episode_to_dict(example_episode())
    plies = payload["plies"]
    assert isinstance(plies, list)
    first_ply = plies[0]
    assert isinstance(first_ply, dict)
    first_ply["board_before"] = [0, 0, 0, 0]
    write_payload(path, payload)

    with pytest.raises(ValueError, match="board_before"):
        load_jsonl(path)


def test_load_jsonl_rejects_action_outside_recorded_columns(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    payload = episode_to_dict(example_episode())
    plies = payload["plies"]
    assert isinstance(plies, list)
    first_ply = plies[0]
    assert isinstance(first_ply, dict)
    first_ply["action"] = 4
    write_payload(path, payload)

    with pytest.raises(ValueError, match="action"):
        load_jsonl(path)


def test_load_jsonl_preserves_runtime_failure_provenance(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    expected = replace(
        example_episode(),
        opening_column=None,
        plies=(),
        winner=None,
        candidate_reward=None,
        candidate_result="failure",
        failure_kind="candidate_runtime",
        failure_reason="INVALID",
    )
    write_jsonl((expected,), path)

    assert load_jsonl(path) == (expected,)
