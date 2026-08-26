import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CandidateResult = Literal["win", "draw", "loss", "failure"]


@dataclass(frozen=True)
class PositionFeatures:
    legal_columns: tuple[int, ...]
    p1_viable_windows: int
    p2_viable_windows: int
    p1_playable_threats: int
    p2_playable_threats: int
    p1_latent_threats: int
    p2_latent_threats: int
    p1_winning_columns: tuple[int, ...]
    p2_winning_columns: tuple[int, ...]
    p1_fork_moves: tuple[int, ...]
    p2_fork_moves: tuple[int, ...]
    p1_surviving_replies: tuple[int, ...]
    p2_surviving_replies: tuple[int, ...]


@dataclass(frozen=True)
class PlyRecord:
    ply: int
    mark: int
    board_before: tuple[int, ...]
    action: int
    board_after: tuple[int, ...]
    features_before: PositionFeatures


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    opponent: str
    candidate_mark: int
    opening_column: int | None
    plies: tuple[PlyRecord, ...]
    winner: int | None
    candidate_reward: int | None
    candidate_result: CandidateResult


def position_features_to_dict(features: PositionFeatures) -> dict[str, object]:
    return {
        "legal_columns": list(features.legal_columns),
        "p1_viable_windows": features.p1_viable_windows,
        "p2_viable_windows": features.p2_viable_windows,
        "p1_playable_threats": features.p1_playable_threats,
        "p2_playable_threats": features.p2_playable_threats,
        "p1_latent_threats": features.p1_latent_threats,
        "p2_latent_threats": features.p2_latent_threats,
        "p1_winning_columns": list(features.p1_winning_columns),
        "p2_winning_columns": list(features.p2_winning_columns),
        "p1_fork_moves": list(features.p1_fork_moves),
        "p2_fork_moves": list(features.p2_fork_moves),
        "p1_surviving_replies": list(features.p1_surviving_replies),
        "p2_surviving_replies": list(features.p2_surviving_replies),
    }


def ply_to_dict(record: PlyRecord) -> dict[str, object]:
    return {
        "ply": record.ply,
        "mark": record.mark,
        "board_before": list(record.board_before),
        "action": record.action,
        "board_after": list(record.board_after),
        "features_before": position_features_to_dict(record.features_before),
    }


def episode_to_dict(record: EpisodeRecord) -> dict[str, object]:
    return {
        "episode_id": record.episode_id,
        "opponent": record.opponent,
        "candidate_mark": record.candidate_mark,
        "opening_column": record.opening_column,
        "plies": [ply_to_dict(ply) for ply in record.plies],
        "winner": record.winner,
        "candidate_reward": record.candidate_reward,
        "candidate_result": record.candidate_result,
    }


def write_jsonl(records: tuple[EpisodeRecord, ...], path: Path) -> None:
    """Write one deterministic JSON object per study episode to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(
                json.dumps(
                    episode_to_dict(record),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            output.write("\n")
