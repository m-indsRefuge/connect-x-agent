from dataclasses import replace

import pytest

from connect_x_agent.study.analysis import (
    analyze_episodes,
    ConfigurationCount,
    MoveCount,
    OpeningResponseCount,
    OpeningResult,
    OpponentCount,
    trajectory_fingerprint,
)
from connect_x_agent.study.records import EpisodeRecord, PlyRecord, PositionFeatures


COLUMNS = 4
ROWS = 3
INAROW = 3


def neutral_features(columns: int = COLUMNS) -> PositionFeatures:
    legal = tuple(range(columns))
    return PositionFeatures(
        legal_columns=legal,
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
        p1_surviving_replies=legal,
        p2_surviving_replies=legal,
    )


def drop(
    board: tuple[int, ...],
    *,
    column: int,
    mark: int,
    columns: int,
    rows: int,
) -> tuple[int, ...]:
    updated = list(board)
    for row in range(rows - 1, -1, -1):
        cell = row * columns + column
        if updated[cell] == 0:
            updated[cell] = mark
            return tuple(updated)
    raise AssertionError(f"synthetic column {column} is full")


def episode_from_actions(
    *,
    episode_id: int,
    opponent: str,
    actions: tuple[tuple[int, int], ...],
    candidate_result: str = "win",
    columns: int = COLUMNS,
    rows: int = ROWS,
    inarow: int = INAROW,
) -> EpisodeRecord:
    board = (0,) * (columns * rows)
    plies: list[PlyRecord] = []

    for ply_number, (mark, action) in enumerate(actions, start=1):
        board_after = drop(
            board,
            column=action,
            mark=mark,
            columns=columns,
            rows=rows,
        )
        plies.append(
            PlyRecord(
                ply=ply_number,
                mark=mark,
                board_before=board,
                action=action,
                board_after=board_after,
                features_before=neutral_features(columns),
            )
        )
        board = board_after

    reward_by_result = {"win": 1, "draw": 0, "loss": -1, "failure": None}
    winner_by_result = {"win": 2, "draw": None, "loss": 1, "failure": None}
    opening_column = next((action for mark, action in actions if mark == 1), None)

    return EpisodeRecord(
        episode_id=episode_id,
        opponent=opponent,
        candidate_mark=2,
        opening_column=opening_column,
        plies=tuple(plies),
        winner=winner_by_result[candidate_result],
        candidate_reward=reward_by_result[candidate_result],
        candidate_result=candidate_result,  # type: ignore[arg-type]
        columns=columns,
        rows=rows,
        inarow=inarow,
        failure_kind="candidate_runtime" if candidate_result == "failure" else None,
        failure_reason="INVALID" if candidate_result == "failure" else None,
    )


def test_trajectory_identity_ignores_episode_metadata_and_outcome() -> None:
    actions = ((1, 1), (2, 2), (1, 0), (2, 3))
    first = episode_from_actions(
        episode_id=1,
        opponent="zeta",
        actions=actions,
        candidate_result="win",
    )
    duplicate = episode_from_actions(
        episode_id=999,
        opponent="alpha",
        actions=actions,
        candidate_result="loss",
    )
    different = episode_from_actions(
        episode_id=2,
        opponent="alpha",
        actions=((1, 1), (2, 3), (1, 0), (2, 2)),
        candidate_result="draw",
    )

    assert trajectory_fingerprint(first) == trajectory_fingerprint(duplicate)
    assert trajectory_fingerprint(first) != trajectory_fingerprint(different)

    report = analyze_episodes((first, duplicate, different))

    assert report.corpus.episodes == 3
    assert report.corpus.unique_trajectories == 2
    assert report.corpus.duplicate_trajectories == 1
    assert report.corpus.duplicate_rate == pytest.approx(1 / 3)


def test_corpus_summary_counts_results_and_sorts_distributions() -> None:
    standard = episode_from_actions(
        episode_id=1,
        opponent="zeta",
        actions=((1, 1), (2, 2)),
        candidate_result="win",
    )
    loss = episode_from_actions(
        episode_id=2,
        opponent="alpha",
        actions=((1, 2), (2, 1)),
        candidate_result="loss",
    )
    draw = episode_from_actions(
        episode_id=3,
        opponent="alpha",
        actions=((1, 3), (2, 0)),
        candidate_result="draw",
    )
    failure = replace(
        episode_from_actions(
            episode_id=4,
            opponent="beta",
            actions=(),
            candidate_result="failure",
            columns=5,
            rows=4,
            inarow=4,
        ),
        opening_column=None,
        plies=(),
    )

    report = analyze_episodes((standard, loss, draw, failure))

    assert report.corpus.episodes == 4
    assert report.corpus.wins == 1
    assert report.corpus.draws == 1
    assert report.corpus.losses == 1
    assert report.corpus.failures == 1
    assert report.corpus.opponents == (
        OpponentCount(opponent="alpha", episodes=2),
        OpponentCount(opponent="beta", episodes=1),
        OpponentCount(opponent="zeta", episodes=1),
    )
    assert report.corpus.configurations == (
        ConfigurationCount(columns=4, rows=3, inarow=3, episodes=3),
        ConfigurationCount(columns=5, rows=4, inarow=4, episodes=1),
    )


def test_opening_summary_counts_results_and_first_p2_responses() -> None:
    first = episode_from_actions(
        episode_id=1,
        opponent="random",
        actions=((1, 1), (2, 2), (1, 0)),
        candidate_result="win",
    )
    second = episode_from_actions(
        episode_id=2,
        opponent="random",
        actions=((1, 1), (2, 2), (1, 3)),
        candidate_result="loss",
    )
    third = episode_from_actions(
        episode_id=3,
        opponent="random",
        actions=((1, 3), (2, 1)),
        candidate_result="draw",
    )
    no_response = episode_from_actions(
        episode_id=4,
        opponent="random",
        actions=((1, 0),),
        candidate_result="failure",
    )

    report = analyze_episodes((first, second, third, no_response))

    assert report.opening.openings == (
        OpeningResult(
            opening_column=0,
            episodes=1,
            wins=0,
            draws=0,
            losses=0,
            failures=1,
        ),
        OpeningResult(
            opening_column=1,
            episodes=2,
            wins=1,
            draws=0,
            losses=1,
            failures=0,
        ),
        OpeningResult(
            opening_column=3,
            episodes=1,
            wins=0,
            draws=1,
            losses=0,
            failures=0,
        ),
    )
    assert report.opening.first_responses == (
        MoveCount(column=1, episodes=1),
        MoveCount(column=2, episodes=2),
    )
    assert report.opening.responses_by_opening == (
        OpeningResponseCount(opening_column=1, response_column=2, episodes=2),
        OpeningResponseCount(opening_column=3, response_column=1, episodes=1),
    )
