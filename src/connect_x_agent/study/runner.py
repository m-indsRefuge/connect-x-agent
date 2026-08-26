from itertools import pairwise
from typing import Any

from .instrumentation import position_features
from .records import CandidateResult, EpisodeRecord, FailureKind, PlyRecord


def run_p2_study(
    *,
    candidate: Any,
    opponent: Any,
    games: int,
    columns: int = 7,
    rows: int = 6,
    inarow: int = 4,
    opponent_name: str,
) -> tuple[EpisodeRecord, ...]:
    """Run recorded games with ``candidate`` fixed to Kaggle seat two.

    Structured agent/runtime failures are recorded as episode provenance.
    Malformed Kaggle history or study-pipeline decoding failures are allowed
    to raise so they cannot be silently mixed into the empirical corpus.
    """
    # Import lazily so record and instrumentation unit tests do not pay the
    # kaggle-environments/OpenSpiel startup cost.
    from kaggle_environments import make  # type: ignore[import-untyped]

    episodes: list[EpisodeRecord] = []

    for episode_id in range(games):
        environment = make(
            "connectx",
            configuration={
                "columns": columns,
                "rows": rows,
                "inarow": inarow,
            },
        )
        history = environment.run([opponent, candidate])
        episodes.append(
            _episode_from_history(
                history=history,
                episode_id=episode_id,
                opponent_name=opponent_name,
                columns=columns,
                rows=rows,
                inarow=inarow,
            )
        )

    return tuple(episodes)


def _episode_from_history(
    *,
    history: list[list[Any]],
    episode_id: int,
    opponent_name: str,
    columns: int,
    rows: int,
    inarow: int,
) -> EpisodeRecord:
    if not history:
        raise ValueError("Kaggle Connect X history is empty")

    plies: list[PlyRecord] = []

    for previous_state, current_state in pairwise(history):
        board_before = _shared_board(previous_state)
        board_after = _shared_board(current_state)
        changed_cells = [
            cell
            for cell, (before, after) in enumerate(
                zip(board_before, board_after, strict=True)
            )
            if before != after
        ]

        if not changed_cells:
            continue

        if len(changed_cells) != 1:
            raise ValueError("Kaggle Connect X transition changed multiple cells")

        changed_cell = changed_cells[0]
        mark = board_after[changed_cell]

        if board_before[changed_cell] != 0 or mark not in (1, 2):
            raise ValueError("Kaggle Connect X transition is not a legal drop")

        action = int(current_state[mark - 1].action)

        if (
            not 0 <= action < columns
            or changed_cell % columns != action
            or board_before[action] != 0
        ):
            raise ValueError("Kaggle Connect X action does not match its board transition")

        plies.append(
            PlyRecord(
                ply=len(plies) + 1,
                mark=mark,
                board_before=board_before,
                action=action,
                board_after=board_after,
                features_before=position_features(
                    list(board_before),
                    columns,
                    rows,
                    inarow,
                ),
            )
        )

    (
        candidate_reward,
        winner,
        candidate_result,
        failure_kind,
        failure_reason,
    ) = _candidate_outcome(history[-1])
    opening_column = plies[0].action if plies and plies[0].mark == 1 else None

    return EpisodeRecord(
        episode_id=episode_id,
        opponent=opponent_name,
        candidate_mark=2,
        opening_column=opening_column,
        plies=tuple(plies),
        winner=winner,
        candidate_reward=candidate_reward,
        candidate_result=candidate_result,
        columns=columns,
        rows=rows,
        inarow=inarow,
        failure_kind=failure_kind,
        failure_reason=failure_reason,
    )


def _shared_board(state: list[Any]) -> tuple[int, ...]:
    return tuple(int(cell) for cell in state[0].observation["board"])


def _candidate_outcome(
    final_state: list[Any],
) -> tuple[
    int | None,
    int | None,
    CandidateResult,
    FailureKind | None,
    str | None,
]:
    opponent_state = final_state[0]
    candidate_state = final_state[1]
    opponent_status = str(opponent_state.status)
    candidate_status = str(candidate_state.status)

    if candidate_status != "DONE":
        return None, None, "failure", "candidate_runtime", candidate_status

    if opponent_status != "DONE":
        return None, None, "failure", "opponent_runtime", opponent_status

    reward = candidate_state.reward

    if not isinstance(reward, int):
        raise ValueError(f"Unexpected candidate reward: {reward!r}")

    if reward == 1:
        return reward, 2, "win", None, None

    if reward == 0:
        return reward, None, "draw", None, None

    if reward == -1:
        return reward, 1, "loss", None, None

    raise ValueError(f"Unexpected candidate reward: {reward!r}")
