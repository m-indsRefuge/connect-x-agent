import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from .instrumentation import position_features
from .records import CandidateResult, EpisodeRecord, PositionFeatures


@dataclass(frozen=True)
class OpponentCount:
    opponent: str
    episodes: int


@dataclass(frozen=True)
class ConfigurationCount:
    columns: int
    rows: int
    inarow: int
    episodes: int


@dataclass(frozen=True)
class CorpusSummary:
    episodes: int
    wins: int
    draws: int
    losses: int
    failures: int
    opponents: tuple[OpponentCount, ...]
    configurations: tuple[ConfigurationCount, ...]
    unique_trajectories: int
    duplicate_trajectories: int
    duplicate_rate: float


@dataclass(frozen=True)
class OpeningResult:
    opening_column: int
    episodes: int
    wins: int
    draws: int
    losses: int
    failures: int


@dataclass(frozen=True)
class MoveCount:
    column: int
    episodes: int


@dataclass(frozen=True)
class OpeningResponseCount:
    opening_column: int
    response_column: int
    episodes: int


@dataclass(frozen=True)
class OpeningSummary:
    openings: tuple[OpeningResult, ...]
    first_responses: tuple[MoveCount, ...]
    responses_by_opening: tuple[OpeningResponseCount, ...]


@dataclass(frozen=True)
class ForcedDefenseEpisode:
    episode_id: int
    first_ply: int
    total_positions: int
    obeyed_positions: int
    first_obeyed: bool
    candidate_result: CandidateResult


@dataclass(frozen=True)
class ForcedDefenseSummary:
    episodes_with_forced_defense: int
    total_positions: int
    obeyed_positions: int
    events: tuple[ForcedDefenseEpisode, ...]


@dataclass(frozen=True)
class CounterattackEpisode:
    episode_id: int
    creating_p1_ply: int
    fork_moves: tuple[int, ...]
    next_p2_used_fork: bool | None
    candidate_result: CandidateResult


@dataclass(frozen=True)
class CounterattackSummary:
    episodes_with_counterattack: int
    events: tuple[CounterattackEpisode, ...]


@dataclass(frozen=True)
class ForcingEpisode:
    episode_id: int
    creating_p2_ply: int
    sole_p1_reply: int
    candidate_result: CandidateResult


@dataclass(frozen=True)
class ForcingSummary:
    episodes_with_forcing_move: int
    events: tuple[ForcingEpisode, ...]


@dataclass(frozen=True)
class LossAnatomy:
    episode_id: int
    first_zero_survival_ply: int | None
    missed_forced_defense: bool
    ever_fork_opportunity: bool
    ever_forced_p1_reply: bool


@dataclass(frozen=True)
class StudyReport:
    corpus: CorpusSummary
    opening: OpeningSummary
    forced_defense: ForcedDefenseSummary
    counterattack: CounterattackSummary
    forcing: ForcingSummary
    loss_anatomy: tuple[LossAnatomy, ...]


def trajectory_fingerprint(episode: EpisodeRecord) -> str:
    payload = {
        "columns": episode.columns,
        "rows": episode.rows,
        "inarow": episode.inarow,
        "plies": [[ply.mark, ply.action] for ply in episode.plies],
    }
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def analyze_episodes(episodes: tuple[EpisodeRecord, ...]) -> StudyReport:
    forced_defense = _forced_defense_summary(episodes)
    counterattack = _counterattack_summary(episodes)
    forcing = _forcing_summary(episodes)

    return StudyReport(
        corpus=_corpus_summary(episodes),
        opening=_opening_summary(episodes),
        forced_defense=forced_defense,
        counterattack=counterattack,
        forcing=forcing,
        loss_anatomy=_loss_anatomy(episodes, counterattack, forcing),
    )


def _corpus_summary(episodes: tuple[EpisodeRecord, ...]) -> CorpusSummary:
    opponent_counts = Counter(episode.opponent for episode in episodes)
    configuration_counts = Counter(
        (episode.columns, episode.rows, episode.inarow) for episode in episodes
    )
    fingerprints = {trajectory_fingerprint(episode) for episode in episodes}
    total = len(episodes)
    unique = len(fingerprints)

    return CorpusSummary(
        episodes=total,
        wins=sum(episode.candidate_result == "win" for episode in episodes),
        draws=sum(episode.candidate_result == "draw" for episode in episodes),
        losses=sum(episode.candidate_result == "loss" for episode in episodes),
        failures=sum(episode.candidate_result == "failure" for episode in episodes),
        opponents=tuple(
            OpponentCount(opponent=name, episodes=count)
            for name, count in sorted(opponent_counts.items())
        ),
        configurations=tuple(
            ConfigurationCount(
                columns=columns,
                rows=rows,
                inarow=inarow,
                episodes=count,
            )
            for (columns, rows, inarow), count in sorted(configuration_counts.items())
        ),
        unique_trajectories=unique,
        duplicate_trajectories=total - unique,
        duplicate_rate=(total - unique) / total if total else 0.0,
    )


def _opening_summary(episodes: tuple[EpisodeRecord, ...]) -> OpeningSummary:
    opening_results: dict[int, Counter[str]] = {}
    response_counts: Counter[int] = Counter()
    responses_by_opening: Counter[tuple[int, int]] = Counter()

    for episode in episodes:
        opening = _first_action(episode, mark=1)
        if opening is None:
            continue

        result_counts = opening_results.setdefault(opening, Counter())
        result_counts[episode.candidate_result] += 1

        response = _first_action_after_opening(episode)
        if response is not None:
            response_counts[response] += 1
            responses_by_opening[(opening, response)] += 1

    openings = tuple(
        OpeningResult(
            opening_column=column,
            episodes=sum(counts.values()),
            wins=counts["win"],
            draws=counts["draw"],
            losses=counts["loss"],
            failures=counts["failure"],
        )
        for column, counts in sorted(opening_results.items())
    )
    first_responses = tuple(
        MoveCount(column=column, episodes=count)
        for column, count in sorted(response_counts.items())
    )
    conditional_responses = tuple(
        OpeningResponseCount(
            opening_column=opening,
            response_column=response,
            episodes=count,
        )
        for (opening, response), count in sorted(responses_by_opening.items())
    )

    return OpeningSummary(
        openings=openings,
        first_responses=first_responses,
        responses_by_opening=conditional_responses,
    )


def _forced_defense_summary(
    episodes: tuple[EpisodeRecord, ...],
) -> ForcedDefenseSummary:
    events: list[ForcedDefenseEpisode] = []

    for episode in episodes:
        forced_positions = [
            ply
            for ply in episode.plies
            if ply.mark == 2 and len(ply.features_before.p2_surviving_replies) == 1
        ]
        if not forced_positions:
            continue

        obeyed_positions = sum(
            ply.action == ply.features_before.p2_surviving_replies[0]
            for ply in forced_positions
        )
        first = forced_positions[0]
        events.append(
            ForcedDefenseEpisode(
                episode_id=episode.episode_id,
                first_ply=first.ply,
                total_positions=len(forced_positions),
                obeyed_positions=obeyed_positions,
                first_obeyed=(
                    first.action == first.features_before.p2_surviving_replies[0]
                ),
                candidate_result=episode.candidate_result,
            )
        )

    return ForcedDefenseSummary(
        episodes_with_forced_defense=len(events),
        total_positions=sum(event.total_positions for event in events),
        obeyed_positions=sum(event.obeyed_positions for event in events),
        events=tuple(events),
    )


def _counterattack_summary(
    episodes: tuple[EpisodeRecord, ...],
) -> CounterattackSummary:
    events: list[CounterattackEpisode] = []

    for episode in episodes:
        for ply_index, ply in enumerate(episode.plies):
            if ply.mark != 1 or ply.features_before.p2_fork_moves:
                continue

            features_after = _features_after(episode, ply_index)
            if not features_after.p2_fork_moves:
                continue

            next_p2 = next(
                (later for later in episode.plies[ply_index + 1 :] if later.mark == 2),
                None,
            )
            events.append(
                CounterattackEpisode(
                    episode_id=episode.episode_id,
                    creating_p1_ply=ply.ply,
                    fork_moves=features_after.p2_fork_moves,
                    next_p2_used_fork=(
                        None
                        if next_p2 is None
                        else next_p2.action in features_after.p2_fork_moves
                    ),
                    candidate_result=episode.candidate_result,
                )
            )
            break

    return CounterattackSummary(
        episodes_with_counterattack=len(events),
        events=tuple(events),
    )


def _forcing_summary(episodes: tuple[EpisodeRecord, ...]) -> ForcingSummary:
    events: list[ForcingEpisode] = []

    for episode in episodes:
        for ply_index, ply in enumerate(episode.plies):
            if ply.mark != 2:
                continue

            replies = _features_after(episode, ply_index).p1_surviving_replies
            if len(replies) != 1:
                continue

            events.append(
                ForcingEpisode(
                    episode_id=episode.episode_id,
                    creating_p2_ply=ply.ply,
                    sole_p1_reply=replies[0],
                    candidate_result=episode.candidate_result,
                )
            )
            break

    return ForcingSummary(
        episodes_with_forcing_move=len(events),
        events=tuple(events),
    )


def _loss_anatomy(
    episodes: tuple[EpisodeRecord, ...],
    counterattack: CounterattackSummary,
    forcing: ForcingSummary,
) -> tuple[LossAnatomy, ...]:
    counterattack_episode_ids = {event.episode_id for event in counterattack.events}
    forcing_episode_ids = {event.episode_id for event in forcing.events}
    losses: list[LossAnatomy] = []

    for episode in episodes:
        if episode.candidate_result != "loss":
            continue

        p2_plies = tuple(ply for ply in episode.plies if ply.mark == 2)
        first_zero = next(
            (
                ply.ply
                for ply in p2_plies
                if len(ply.features_before.p2_surviving_replies) == 0
            ),
            None,
        )
        missed_forced_defense = any(
            len(ply.features_before.p2_surviving_replies) == 1
            and ply.action != ply.features_before.p2_surviving_replies[0]
            for ply in p2_plies
        )
        pre_move_fork = any(ply.features_before.p2_fork_moves for ply in p2_plies)

        losses.append(
            LossAnatomy(
                episode_id=episode.episode_id,
                first_zero_survival_ply=first_zero,
                missed_forced_defense=missed_forced_defense,
                ever_fork_opportunity=(
                    pre_move_fork or episode.episode_id in counterattack_episode_ids
                ),
                ever_forced_p1_reply=episode.episode_id in forcing_episode_ids,
            )
        )

    return tuple(losses)


def _features_after(episode: EpisodeRecord, ply_index: int) -> PositionFeatures:
    if ply_index + 1 < len(episode.plies):
        return episode.plies[ply_index + 1].features_before

    ply = episode.plies[ply_index]
    return position_features(
        list(ply.board_after),
        episode.columns,
        episode.rows,
        episode.inarow,
    )


def _first_action(episode: EpisodeRecord, *, mark: int) -> int | None:
    return next((ply.action for ply in episode.plies if ply.mark == mark), None)


def _first_action_after_opening(episode: EpisodeRecord) -> int | None:
    opening_seen = False
    for ply in episode.plies:
        if not opening_seen:
            if ply.mark == 1:
                opening_seen = True
            continue
        if ply.mark == 2:
            return ply.action
    return None
