from dataclasses import replace

import pytest

import connect_x_agent.study.analysis as study_analysis
from connect_x_agent.study.analysis import (
    ConfigurationCount,
    MoveCount,
    OpeningResponseCount,
    OpeningResult,
    OpponentCount,
    analyze_episodes,
    trajectory_fingerprint,
)
from connect_x_agent.study.instrumentation import position_features
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


def test_forced_defense_counts_compliance_and_ignores_p1_positions() -> None:
    obeyed = episode_from_actions(
        episode_id=10,
        opponent="random",
        actions=((1, 1), (2, 2)),
        candidate_result="win",
    )
    obeyed = replace(
        obeyed,
        plies=(
            replace(
                obeyed.plies[0],
                features_before=replace(
                    obeyed.plies[0].features_before,
                    p2_surviving_replies=(1,),
                ),
            ),
            replace(
                obeyed.plies[1],
                features_before=replace(
                    obeyed.plies[1].features_before,
                    p2_surviving_replies=(2,),
                ),
            ),
        ),
    )
    missed = episode_from_actions(
        episode_id=11,
        opponent="random",
        actions=((1, 1), (2, 3)),
        candidate_result="loss",
    )
    missed = replace(
        missed,
        plies=(
            replace(
                missed.plies[0],
                features_before=replace(
                    missed.plies[0].features_before,
                    p2_surviving_replies=(1,),
                ),
            ),
            replace(
                missed.plies[1],
                features_before=replace(
                    missed.plies[1].features_before,
                    p2_surviving_replies=(2,),
                ),
            ),
        ),
    )

    report = analyze_episodes((obeyed, missed))

    assert report.forced_defense.episodes_with_forced_defense == 2
    assert report.forced_defense.total_positions == 2
    assert report.forced_defense.obeyed_positions == 1
    assert len(report.forced_defense.events) == 2

    first, second = report.forced_defense.events
    assert first.episode_id == 10
    assert first.first_ply == 2
    assert first.total_positions == 1
    assert first.obeyed_positions == 1
    assert first.first_obeyed is True
    assert first.candidate_result == "win"

    assert second.episode_id == 11
    assert second.first_ply == 2
    assert second.total_positions == 1
    assert second.obeyed_positions == 0
    assert second.first_obeyed is False
    assert second.candidate_result == "loss"


def test_counterattack_records_new_p2_fork_and_whether_candidate_uses_it() -> None:
    used = episode_from_actions(
        episode_id=20,
        opponent="random",
        actions=((1, 1), (2, 2), (1, 0), (2, 3)),
        candidate_result="win",
    )
    used = replace(
        used,
        plies=(
            used.plies[0],
            used.plies[1],
            used.plies[2],
            replace(
                used.plies[3],
                features_before=replace(
                    used.plies[3].features_before,
                    p2_fork_moves=(3,),
                ),
            ),
        ),
    )
    missed = episode_from_actions(
        episode_id=21,
        opponent="random",
        actions=((1, 1), (2, 2), (1, 0), (2, 2)),
        candidate_result="loss",
    )
    missed = replace(
        missed,
        plies=(
            missed.plies[0],
            missed.plies[1],
            missed.plies[2],
            replace(
                missed.plies[3],
                features_before=replace(
                    missed.plies[3].features_before,
                    p2_fork_moves=(3,),
                ),
            ),
        ),
    )

    report = analyze_episodes((used, missed))

    assert report.counterattack.episodes_with_counterattack == 2
    assert len(report.counterattack.events) == 2

    first, second = report.counterattack.events
    assert first.episode_id == 20
    assert first.creating_p1_ply == 3
    assert first.fork_moves == (3,)
    assert first.next_p2_used_fork is True
    assert first.candidate_result == "win"

    assert second.episode_id == 21
    assert second.creating_p1_ply == 3
    assert second.fork_moves == (3,)
    assert second.next_p2_used_fork is False
    assert second.candidate_result == "loss"


def test_forcing_records_first_p2_move_that_leaves_one_p1_reply() -> None:
    episode = episode_from_actions(
        episode_id=30,
        opponent="random",
        actions=((1, 1), (2, 2), (1, 0), (2, 3), (1, 1)),
        candidate_result="win",
    )
    episode = replace(
        episode,
        plies=(
            episode.plies[0],
            episode.plies[1],
            episode.plies[2],
            episode.plies[3],
            replace(
                episode.plies[4],
                features_before=replace(
                    episode.plies[4].features_before,
                    p1_surviving_replies=(0,),
                ),
            ),
        ),
    )

    report = analyze_episodes((episode,))

    assert report.forcing.episodes_with_forcing_move == 1
    assert len(report.forcing.events) == 1
    event = report.forcing.events[0]
    assert event.episode_id == 30
    assert event.creating_p2_ply == 4
    assert event.sole_p1_reply == 0
    assert event.candidate_result == "win"


def test_forcing_recomputes_features_after_terminal_last_move() -> None:
    episode = episode_from_actions(
        episode_id=31,
        opponent="random",
        actions=(
            (1, 0),
            (2, 0),
            (1, 0),
            (2, 1),
            (1, 2),
            (2, 1),
            (1, 1),
            (2, 2),
        ),
        candidate_result="win",
    )
    terminal = episode.plies[-1]
    expected = position_features(
        list(terminal.board_after),
        episode.columns,
        episode.rows,
        episode.inarow,
    ).p1_surviving_replies
    assert expected == (2,)

    report = analyze_episodes((episode,))

    assert report.forcing.episodes_with_forcing_move == 1
    event = report.forcing.events[0]
    assert event.creating_p2_ply == terminal.ply
    assert event.sole_p1_reply == expected[0]


def test_loss_anatomy_reports_mechanical_flags_and_excludes_non_losses() -> None:
    tactical_loss = episode_from_actions(
        episode_id=40,
        opponent="random",
        actions=((1, 1), (2, 2), (1, 0), (2, 3), (1, 1)),
        candidate_result="loss",
    )
    tactical_loss = replace(
        tactical_loss,
        plies=(
            tactical_loss.plies[0],
            replace(
                tactical_loss.plies[1],
                features_before=replace(
                    tactical_loss.plies[1].features_before,
                    p2_surviving_replies=(1,),
                ),
            ),
            tactical_loss.plies[2],
            replace(
                tactical_loss.plies[3],
                features_before=replace(
                    tactical_loss.plies[3].features_before,
                    p2_fork_moves=(3,),
                    p2_surviving_replies=(),
                ),
            ),
            replace(
                tactical_loss.plies[4],
                features_before=replace(
                    tactical_loss.plies[4].features_before,
                    p1_surviving_replies=(0,),
                ),
            ),
        ),
    )
    quiet_loss = episode_from_actions(
        episode_id=41,
        opponent="random",
        actions=((1, 1), (2, 2), (1, 0)),
        candidate_result="loss",
    )
    non_loss = episode_from_actions(
        episode_id=42,
        opponent="random",
        actions=((1, 1), (2, 2)),
        candidate_result="win",
    )

    report = analyze_episodes((tactical_loss, quiet_loss, non_loss))

    assert len(report.loss_anatomy) == 2

    first, second = report.loss_anatomy
    assert first.episode_id == 40
    assert first.first_zero_survival_ply == 4
    assert first.missed_forced_defense is True
    assert first.ever_fork_opportunity is True
    assert first.ever_forced_p1_reply is True

    assert second.episode_id == 41
    assert second.first_zero_survival_ply is None
    assert second.missed_forced_defense is False
    assert second.ever_fork_opportunity is False
    assert second.ever_forced_p1_reply is False
    assert all(item.episode_id != 42 for item in report.loss_anatomy)


def test_report_serialization_is_deterministic_and_json_ready() -> None:
    episode = episode_from_actions(
        episode_id=50,
        opponent="random",
        actions=((1, 1), (2, 2)),
        candidate_result="win",
    )
    report = analyze_episodes((episode,))

    serialized = study_analysis.report_to_dict(report)

    assert serialized == study_analysis.report_to_dict(report)
    assert serialized["corpus"]["episodes"] == 1
    assert isinstance(serialized["forced_defense"]["events"], list)
    assert isinstance(serialized["opening"]["openings"], list)
    assert isinstance(serialized["loss_anatomy"], list)
