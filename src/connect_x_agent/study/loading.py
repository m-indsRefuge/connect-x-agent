import json
from pathlib import Path
from typing import Any, cast

from .records import (
    CandidateResult,
    EpisodeRecord,
    FailureKind,
    PlyRecord,
    PositionFeatures,
)


def load_jsonl(path: Path) -> tuple[EpisodeRecord, ...]:
    episodes: list[EpisodeRecord] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"Blank JSONL record at line {line_number}")

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at line {line_number}: {error.msg}"
            ) from error

        episodes.append(_episode_from_json(raw))

    return tuple(episodes)


def _episode_from_json(raw: Any) -> EpisodeRecord:
    episode = cast(dict[str, Any], raw)

    return EpisodeRecord(
        episode_id=int(episode["episode_id"]),
        opponent=str(episode["opponent"]),
        candidate_mark=int(episode["candidate_mark"]),
        opening_column=_optional_int(episode["opening_column"]),
        plies=tuple(_ply_from_json(ply) for ply in episode["plies"]),
        winner=_optional_int(episode["winner"]),
        candidate_reward=_optional_int(episode["candidate_reward"]),
        candidate_result=cast(CandidateResult, episode["candidate_result"]),
        columns=int(episode["columns"]),
        rows=int(episode["rows"]),
        inarow=int(episode["inarow"]),
        failure_kind=cast(FailureKind | None, episode["failure_kind"]),
        failure_reason=_optional_str(episode["failure_reason"]),
    )


def _ply_from_json(raw: Any) -> PlyRecord:
    ply = cast(dict[str, Any], raw)

    return PlyRecord(
        ply=int(ply["ply"]),
        mark=int(ply["mark"]),
        board_before=tuple(int(cell) for cell in ply["board_before"]),
        action=int(ply["action"]),
        board_after=tuple(int(cell) for cell in ply["board_after"]),
        features_before=_position_features_from_json(ply["features_before"]),
    )


def _position_features_from_json(raw: Any) -> PositionFeatures:
    features = cast(dict[str, Any], raw)

    return PositionFeatures(
        legal_columns=_int_tuple(features["legal_columns"]),
        p1_viable_windows=int(features["p1_viable_windows"]),
        p2_viable_windows=int(features["p2_viable_windows"]),
        p1_playable_threats=int(features["p1_playable_threats"]),
        p2_playable_threats=int(features["p2_playable_threats"]),
        p1_latent_threats=int(features["p1_latent_threats"]),
        p2_latent_threats=int(features["p2_latent_threats"]),
        p1_winning_columns=_int_tuple(features["p1_winning_columns"]),
        p2_winning_columns=_int_tuple(features["p2_winning_columns"]),
        p1_fork_moves=_int_tuple(features["p1_fork_moves"]),
        p2_fork_moves=_int_tuple(features["p2_fork_moves"]),
        p1_surviving_replies=_int_tuple(features["p1_surviving_replies"]),
        p2_surviving_replies=_int_tuple(features["p2_surviving_replies"]),
    )


def _int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
