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
    """Load self-describing study JSONL into immutable typed records."""
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

        episodes.append(_episode_from_json(raw, line_number=line_number))

    return tuple(episodes)


def _episode_from_json(raw: Any, *, line_number: int) -> EpisodeRecord:
    episode = _mapping(raw, name="episode", line_number=line_number)
    columns = _positive_int(
        _required(episode, "columns", line_number=line_number),
        name="columns",
        line_number=line_number,
    )
    rows = _positive_int(
        _required(episode, "rows", line_number=line_number),
        name="rows",
        line_number=line_number,
    )
    inarow = _positive_int(
        _required(episode, "inarow", line_number=line_number),
        name="inarow",
        line_number=line_number,
    )
    plies = _list(
        _required(episode, "plies", line_number=line_number),
        name="plies",
        line_number=line_number,
    )

    return EpisodeRecord(
        episode_id=_int(
            _required(episode, "episode_id", line_number=line_number),
            name="episode_id",
            line_number=line_number,
        ),
        opponent=_str(
            _required(episode, "opponent", line_number=line_number),
            name="opponent",
            line_number=line_number,
        ),
        candidate_mark=_mark(
            _required(episode, "candidate_mark", line_number=line_number),
            name="candidate_mark",
            line_number=line_number,
        ),
        opening_column=_optional_column(
            _required(episode, "opening_column", line_number=line_number),
            columns=columns,
            name="opening_column",
            line_number=line_number,
        ),
        plies=tuple(
            _ply_from_json(
                ply,
                columns=columns,
                rows=rows,
                line_number=line_number,
            )
            for ply in plies
        ),
        winner=_optional_mark(
            _required(episode, "winner", line_number=line_number),
            name="winner",
            line_number=line_number,
        ),
        candidate_reward=_optional_int(
            _required(episode, "candidate_reward", line_number=line_number),
            name="candidate_reward",
            line_number=line_number,
        ),
        candidate_result=_candidate_result(
            _required(episode, "candidate_result", line_number=line_number),
            line_number=line_number,
        ),
        columns=columns,
        rows=rows,
        inarow=inarow,
        failure_kind=_failure_kind(
            _required(episode, "failure_kind", line_number=line_number),
            line_number=line_number,
        ),
        failure_reason=_optional_str(
            _required(episode, "failure_reason", line_number=line_number),
            name="failure_reason",
            line_number=line_number,
        ),
    )


def _ply_from_json(
    raw: Any,
    *,
    columns: int,
    rows: int,
    line_number: int,
) -> PlyRecord:
    ply = _mapping(raw, name="ply", line_number=line_number)

    return PlyRecord(
        ply=_int(
            _required(ply, "ply", line_number=line_number),
            name="ply",
            line_number=line_number,
        ),
        mark=_mark(
            _required(ply, "mark", line_number=line_number),
            name="mark",
            line_number=line_number,
        ),
        board_before=_board(
            _required(ply, "board_before", line_number=line_number),
            columns=columns,
            rows=rows,
            name="board_before",
            line_number=line_number,
        ),
        action=_column(
            _required(ply, "action", line_number=line_number),
            columns=columns,
            name="action",
            line_number=line_number,
        ),
        board_after=_board(
            _required(ply, "board_after", line_number=line_number),
            columns=columns,
            rows=rows,
            name="board_after",
            line_number=line_number,
        ),
        features_before=_position_features_from_json(
            _required(ply, "features_before", line_number=line_number),
            line_number=line_number,
        ),
    )


def _position_features_from_json(
    raw: Any,
    *,
    line_number: int,
) -> PositionFeatures:
    features = _mapping(raw, name="features_before", line_number=line_number)

    return PositionFeatures(
        legal_columns=_feature_int_tuple(
            features, "legal_columns", line_number=line_number
        ),
        p1_viable_windows=_feature_int(
            features, "p1_viable_windows", line_number=line_number
        ),
        p2_viable_windows=_feature_int(
            features, "p2_viable_windows", line_number=line_number
        ),
        p1_playable_threats=_feature_int(
            features, "p1_playable_threats", line_number=line_number
        ),
        p2_playable_threats=_feature_int(
            features, "p2_playable_threats", line_number=line_number
        ),
        p1_latent_threats=_feature_int(
            features, "p1_latent_threats", line_number=line_number
        ),
        p2_latent_threats=_feature_int(
            features, "p2_latent_threats", line_number=line_number
        ),
        p1_winning_columns=_feature_int_tuple(
            features, "p1_winning_columns", line_number=line_number
        ),
        p2_winning_columns=_feature_int_tuple(
            features, "p2_winning_columns", line_number=line_number
        ),
        p1_fork_moves=_feature_int_tuple(
            features, "p1_fork_moves", line_number=line_number
        ),
        p2_fork_moves=_feature_int_tuple(
            features, "p2_fork_moves", line_number=line_number
        ),
        p1_surviving_replies=_feature_int_tuple(
            features, "p1_surviving_replies", line_number=line_number
        ),
        p2_surviving_replies=_feature_int_tuple(
            features, "p2_surviving_replies", line_number=line_number
        ),
    )


def _feature_int(
    features: dict[str, Any],
    name: str,
    *,
    line_number: int,
) -> int:
    return _int(
        _required(features, name, line_number=line_number),
        name=name,
        line_number=line_number,
    )


def _feature_int_tuple(
    features: dict[str, Any],
    name: str,
    *,
    line_number: int,
) -> tuple[int, ...]:
    return _int_tuple(
        _required(features, name, line_number=line_number),
        name=name,
        line_number=line_number,
    )


def _required(
    mapping: dict[str, Any],
    key: str,
    *,
    line_number: int,
) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing {key!r} at line {line_number}")
    return mapping[key]


def _mapping(value: Any, *, name: str, line_number: int) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object at line {line_number}")
    return cast(dict[str, Any], value)


def _list(value: Any, *, name: str, line_number: int) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list at line {line_number}")
    return cast(list[Any], value)


def _str(value: Any, *, name: str, line_number: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be str at line {line_number}")
    return cast(str, value)


def _optional_str(value: Any, *, name: str, line_number: int) -> str | None:
    if value is None:
        return None
    return _str(value, name=name, line_number=line_number)


def _int(value: Any, *, name: str, line_number: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be int at line {line_number}")
    return cast(int, value)


def _positive_int(value: Any, *, name: str, line_number: int) -> int:
    parsed = _int(value, name=name, line_number=line_number)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive at line {line_number}")
    return parsed


def _optional_int(value: Any, *, name: str, line_number: int) -> int | None:
    if value is None:
        return None
    return _int(value, name=name, line_number=line_number)


def _mark(value: Any, *, name: str, line_number: int) -> int:
    parsed = _int(value, name=name, line_number=line_number)
    if parsed not in (1, 2):
        raise ValueError(f"{name} must be 1 or 2 at line {line_number}")
    return parsed


def _optional_mark(value: Any, *, name: str, line_number: int) -> int | None:
    if value is None:
        return None
    return _mark(value, name=name, line_number=line_number)


def _column(
    value: Any,
    *,
    columns: int,
    name: str,
    line_number: int,
) -> int:
    parsed = _int(value, name=name, line_number=line_number)
    if not 0 <= parsed < columns:
        raise ValueError(
            f"{name} must be in [0, {columns}) at line {line_number}"
        )
    return parsed


def _optional_column(
    value: Any,
    *,
    columns: int,
    name: str,
    line_number: int,
) -> int | None:
    if value is None:
        return None
    return _column(
        value,
        columns=columns,
        name=name,
        line_number=line_number,
    )


def _board(
    value: Any,
    *,
    columns: int,
    rows: int,
    name: str,
    line_number: int,
) -> tuple[int, ...]:
    cells = _list(value, name=name, line_number=line_number)
    expected_cells = columns * rows
    if len(cells) != expected_cells:
        raise ValueError(
            f"{name} must contain {expected_cells} cells at line {line_number}"
        )

    board: list[int] = []
    for index, cell in enumerate(cells):
        parsed = _int(
            cell,
            name=f"{name}[{index}]",
            line_number=line_number,
        )
        if parsed not in (0, 1, 2):
            raise ValueError(
                f"{name}[{index}] must be 0, 1, or 2 at line {line_number}"
            )
        board.append(parsed)

    return tuple(board)


def _int_tuple(value: Any, *, name: str, line_number: int) -> tuple[int, ...]:
    items = _list(value, name=name, line_number=line_number)
    return tuple(
        _int(item, name=f"{name}[{index}]", line_number=line_number)
        for index, item in enumerate(items)
    )


def _candidate_result(value: Any, *, line_number: int) -> CandidateResult:
    parsed = _str(value, name="candidate_result", line_number=line_number)
    if parsed not in ("win", "draw", "loss", "failure"):
        raise ValueError(f"Invalid candidate_result at line {line_number}: {parsed!r}")
    return cast(CandidateResult, parsed)


def _failure_kind(value: Any, *, line_number: int) -> FailureKind | None:
    if value is None:
        return None

    parsed = _str(value, name="failure_kind", line_number=line_number)
    if parsed not in ("candidate_runtime", "opponent_runtime"):
        raise ValueError(f"Invalid failure_kind at line {line_number}: {parsed!r}")
    return cast(FailureKind, parsed)
