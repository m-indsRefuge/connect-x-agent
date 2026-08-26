# CX-GT-02B Descriptive Player-2 Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, typed, read-only analysis layer that reloads CX-GT-02 JSONL evidence and reports corpus integrity, opening structure, forced defence, newly exposed P2 forks, P2 forcing moves, and loss anatomy without influencing the live agent.

**Architecture:** Add a strict JSONL loader that reconstructs the existing immutable study record types, then add a pure `analyze_episodes()` layer that consumes only typed records and existing `position_features()` instrumentation. A thin script performs file I/O and report presentation; all metric definitions remain in the library layer.

**Tech Stack:** Python 3.12+, dataclasses, pathlib, json, hashlib, pytest 9.1+, Ruff, Mypy; no new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-26-cx-gt-02b-analysis-design.md`

## Global Constraints

- CX-GT-02B is observational only: no agent changes, move ranking, training, RL, minimax labels, or challenger promotion.
- Create only `src/connect_x_agent/study/loading.py`, `src/connect_x_agent/study/analysis.py`, `tests/test_study_loading.py`, `tests/test_study_analysis.py`, and `scripts/cx_gt_02_analyze.py` in addition to this plan/spec documentation.
- No pandas, NumPy, notebooks, plotting, machine learning, reinforcement learning, or new package dependencies.
- Loader errors fail loudly; malformed raw records are never silently skipped or repaired.
- Analysis performs no file I/O and never calls Kaggle.
- Post-move features come from the next ply's `features_before`; only a terminal last move may recompute features from `board_after` through the existing `position_features()` function.
- Failure episodes remain visible in corpus counts and are excluded from gameplay metrics when the relevant coherent ply data is unavailable.
- Implementation follows RED -> GREEN and each task ends in a separately testable state.

---

## File structure

- `src/connect_x_agent/study/loading.py` — strict JSONL parsing and reconstruction of `PositionFeatures`, `PlyRecord`, and `EpisodeRecord`.
- `src/connect_x_agent/study/analysis.py` — immutable report types, trajectory identity, pure descriptive metrics, deterministic report serialization.
- `tests/test_study_loading.py` — round-trip, order, schema, enum, board/action, configuration, and failure-provenance tests.
- `tests/test_study_analysis.py` — synthetic episode fixtures and observable metric tests.
- `scripts/cx_gt_02_analyze.py` — CLI-style script that loads, analyzes, prints, and optionally writes deterministic JSON.

---

### Task 1: Strict JSONL record loading

**Files:**
- Create: `tests/test_study_loading.py`
- Create: `src/connect_x_agent/study/loading.py`

**Interfaces:**
- Consumes: `EpisodeRecord`, `PlyRecord`, `PositionFeatures`, `CandidateResult`, and `FailureKind` from `connect_x_agent.study.records`.
- Produces: `load_jsonl(path: Path) -> tuple[EpisodeRecord, ...]`.

- [ ] **Step 1: Write the first failing round-trip and ordering tests**

Create a compact valid `EpisodeRecord`, serialize it with the existing `write_jsonl()`, then require the future loader to reconstruct the same immutable values and preserve line order:

```python
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
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
uv run pytest tests/test_study_loading.py -q
```

Expected: collection/import failure because `connect_x_agent.study.loading` does not exist.

- [ ] **Step 3: Implement the minimum typed decoder needed for round-trip GREEN**

Create `loading.py` with `load_jsonl()` plus small private conversion helpers. Parsing must use `json.loads()` line-by-line, reject blank lines, and reconstruct tuples rather than leaving lists/dicts downstream.

Core shape:

```python
import json
from pathlib import Path
from typing import Any, cast

from .records import CandidateResult, EpisodeRecord, FailureKind, PlyRecord, PositionFeatures


def load_jsonl(path: Path) -> tuple[EpisodeRecord, ...]:
    episodes: list[EpisodeRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record at line {line_number}")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at line {line_number}: {error.msg}") from error
        episodes.append(_episode_from_json(raw, line_number=line_number))
    return tuple(episodes)
```

Use explicit constructors for nested records; do not call `EpisodeRecord(**raw)` because nested values must be validated and converted.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
uv run pytest tests/test_study_loading.py -q
```

Expected: 2 passing tests.

- [ ] **Step 5: Add RED schema-integrity tests**

Add tests that mutate one valid serialized dictionary at a time and assert descriptive exceptions for:

```python
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("candidate_result", "victory"),
        ("failure_kind", "study_decode"),
        ("columns", "4"),
        ("rows", 0),
        ("inarow", 0),
    ],
)
def test_load_jsonl_rejects_invalid_top_level_schema(...): ...


def test_load_jsonl_rejects_missing_required_field(...): ...

def test_load_jsonl_rejects_board_length_not_matching_configuration(...): ...

def test_load_jsonl_rejects_action_outside_recorded_columns(...): ...

def test_load_jsonl_preserves_runtime_failure_provenance(...): ...
```

The failure-provenance case must round-trip an episode with:

```python
candidate_result="failure"
failure_kind="candidate_runtime"
failure_reason="INVALID"
```

- [ ] **Step 6: Run schema tests and verify RED**

Run:

```powershell
uv run pytest tests/test_study_loading.py -q
```

Expected: failures specifically where the minimal loader still accepts invalid schema values.

- [ ] **Step 7: Implement strict validation**

Add helpers with exact-value/type checks. In particular:

```python
def _required(mapping: dict[str, Any], key: str, *, line_number: int) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing {key!r} at line {line_number}")
    return mapping[key]


def _int(value: Any, *, name: str, line_number: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be int at line {line_number}")
    return value
```

Validate:

- top-level JSON value is a mapping;
- `candidate_result` is one of `win`, `draw`, `loss`, `failure`;
- `failure_kind` is `None`, `candidate_runtime`, or `opponent_runtime`;
- `columns`, `rows`, and `inarow` are positive integers;
- every `board_before` and `board_after` has exactly `columns * rows` integer cells in `{0, 1, 2}`;
- ply `mark` is `1` or `2`;
- ply `action` is in `[0, columns)`;
- tuple-valued feature fields are JSON lists of integers before conversion;
- integer count fields are true integers, not booleans;
- `candidate_mark` is `1` or `2`;
- nullable integer fields accept only `None` or a true integer.

Do not infer missing fields or defaults during loading: serialized study data must be self-describing.

- [ ] **Step 8: Run loader tests and repository static checks**

Run:

```powershell
uv run pytest tests/test_study_loading.py -q
uv run ruff check src/connect_x_agent/study/loading.py tests/test_study_loading.py
uv run mypy src/connect_x_agent/study/loading.py tests/test_study_loading.py
```

Expected: all pass.

- [ ] **Step 9: Commit the loader slice**

```powershell
git add src/connect_x_agent/study/loading.py tests/test_study_loading.py
git commit -m "feat: load typed study evidence"
```

---

### Task 2: Corpus integrity, trajectory identity, and opening summaries

**Files:**
- Create: `tests/test_study_analysis.py`
- Create: `src/connect_x_agent/study/analysis.py`

**Interfaces:**
- Consumes: `tuple[EpisodeRecord, ...]` and the existing record fields only.
- Produces: `analyze_episodes(episodes: tuple[EpisodeRecord, ...]) -> StudyReport`, `trajectory_fingerprint(episode: EpisodeRecord) -> str`, and immutable report dataclasses.

- [ ] **Step 1: Define RED tests for corpus, duplicate, and opening behavior**

Use synthetic records with short coherent ply streams. Build two episodes with identical `(columns, rows, inarow, mark/action sequence)` but different `episode_id`, opponent, and outcome, plus a third episode with a different action sequence.

Require:

```python
report = analyze_episodes((first, duplicate, different))

assert report.corpus.episodes == 3
assert report.corpus.unique_trajectories == 2
assert report.corpus.duplicate_trajectories == 1
assert report.corpus.duplicate_rate == pytest.approx(1 / 3)
assert trajectory_fingerprint(first) == trajectory_fingerprint(duplicate)
assert trajectory_fingerprint(first) != trajectory_fingerprint(different)
```

Also assert deterministic typed rows for opponent/configuration distributions and opening/first-response distributions.

- [ ] **Step 2: Run analysis tests and verify RED**

```powershell
uv run pytest tests/test_study_analysis.py -q
```

Expected: import failure because `study.analysis` does not exist.

- [ ] **Step 3: Implement immutable report types and corpus aggregation**

Use explicit dataclasses rather than public dictionaries:

```python
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
```

Trajectory identity must be stable across processes. Serialize only the recorded configuration and ordered `(mark, action)` sequence to compact JSON with sorted keys, then SHA-256 it:

```python
def trajectory_fingerprint(episode: EpisodeRecord) -> str:
    payload = {
        "columns": episode.columns,
        "rows": episode.rows,
        "inarow": episode.inarow,
        "plies": [[ply.mark, ply.action] for ply in episode.plies],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Sort report rows deterministically: opponent name lexicographically; configurations by `(columns, rows, inarow)`.

- [ ] **Step 4: Implement opening report types and aggregation**

Use:

```python
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
```

Find the first P1 action from the ply stream and the first later P2 action. Do not assume column 3 or any standard-board center. Exclude episodes lacking the relevant move from only that metric.

- [ ] **Step 5: Compose `StudyReport` and verify GREEN**

Begin with placeholders only as fully typed empty sections that will be populated in Task 3; do not expose untyped dicts:

```python
@dataclass(frozen=True)
class StudyReport:
    corpus: CorpusSummary
    opening: OpeningSummary
    forced_defense: ForcedDefenseSummary
    counterattack: CounterattackSummary
    forcing: ForcingSummary
    loss_anatomy: tuple[LossAnatomy, ...]
```

Define the Task-3 section dataclasses now with empty event tuples so later interfaces do not change.

Run:

```powershell
uv run pytest tests/test_study_analysis.py -q
uv run ruff check src/connect_x_agent/study/analysis.py tests/test_study_analysis.py
uv run mypy src/connect_x_agent/study/analysis.py tests/test_study_analysis.py
```

Expected: all Task-2 tests pass.

- [ ] **Step 6: Commit the corpus/opening slice**

```powershell
git add src/connect_x_agent/study/analysis.py tests/test_study_analysis.py
git commit -m "feat: summarize study corpus and openings"
```

---

### Task 3: Mechanical turning-point analysis

**Files:**
- Modify: `tests/test_study_analysis.py`
- Modify: `src/connect_x_agent/study/analysis.py`

**Interfaces:**
- Consumes: `EpisodeRecord.plies`, `PlyRecord.features_before`, `PlyRecord.board_after`, episode configuration, and `position_features()` from `study.instrumentation`.
- Produces: populated `ForcedDefenseSummary`, `CounterattackSummary`, `ForcingSummary`, and `LossAnatomy` fields in `StudyReport`.

- [ ] **Step 1: Add RED forced-defence tests**

Construct a P2 ply whose pre-move features contain exactly one `p2_surviving_replies` column. Test both compliance and a miss:

```python
assert report.forced_defense.episodes_with_forced_defense == 2
assert report.forced_defense.events[0].first_ply == 2
assert report.forced_defense.events[0].total_positions == 1
assert report.forced_defense.events[0].first_obeyed is True
assert report.forced_defense.events[1].first_obeyed is False
```

Use the following immutable event shape:

```python
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
```

- [ ] **Step 2: Run the forced-defence test and verify RED**

```powershell
uv run pytest tests/test_study_analysis.py -q -k forced_defense
```

Expected: assertions fail because Task-2 section values are still empty.

- [ ] **Step 3: Implement forced-defence aggregation and verify GREEN**

Only inspect P2 plies. A forced defence is exactly:

```python
len(ply.features_before.p2_surviving_replies) == 1
```

Compliance is:

```python
ply.action == ply.features_before.p2_surviving_replies[0]
```

Run the focused test again and require PASS.

- [ ] **Step 4: Add RED counterattack and P2-forcing tests**

Build coherent adjacent plies so the next ply's `features_before` supplies post-move structure. Require counterattack only when a P1 move changes P2 fork options from empty to non-empty, and record whether the next P2 move selects one of those fork columns.

Use:

```python
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
```

Test terminal-last-move fallback separately by monkeypatching nothing: provide a real legal small-board `board_after` and assert the result matches the existing pure instrumentation.

- [ ] **Step 5: Implement a single post-move feature helper**

Centralize the transition rule:

```python
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
```

Do not duplicate fork/surviving-reply calculations in `analysis.py`.

- [ ] **Step 6: Implement first-occurrence counterattack and forcing events**

Counterattack condition on a P1 ply:

```python
not ply.features_before.p2_fork_moves and _features_after(...).p2_fork_moves
```

Record only the first such P1 move per episode. `next_p2_used_fork` is `None` if no subsequent P2 ply exists; otherwise whether that next P2 action is in the newly available fork columns.

Forcing condition on a P2 ply:

```python
len(_features_after(...).p1_surviving_replies) == 1
```

Record only the first such P2 move per episode.

- [ ] **Step 7: Add RED loss-anatomy tests**

For loss episodes only, require:

```python
@dataclass(frozen=True)
class LossAnatomy:
    episode_id: int
    first_zero_survival_ply: int | None
    missed_forced_defense: bool
    ever_fork_opportunity: bool
    ever_forced_p1_reply: bool
```

Mechanical definitions:

- `first_zero_survival_ply`: first P2 ply with `len(p2_surviving_replies) == 0`;
- `missed_forced_defense`: any P2 ply with exactly one surviving reply where `action` differs from it;
- `ever_fork_opportunity`: any P2 pre-move state with non-empty `p2_fork_moves`, or any counterattack event exposing such a fork before the next P2 action;
- `ever_forced_p1_reply`: whether the episode has a P2 forcing event.

Assert non-loss episodes do not appear in `report.loss_anatomy`.

- [ ] **Step 8: Implement loss anatomy and failure-episode exclusion**

Keep all failure episodes in `CorpusSummary`; only include them in opening/gameplay metrics when the specific recorded ply data required by that metric exists unambiguously. For loss anatomy, only `candidate_result == "loss"` qualifies.

- [ ] **Step 9: Run all analysis tests and static checks**

```powershell
uv run pytest tests/test_study_analysis.py -q
uv run ruff check src/connect_x_agent/study/analysis.py tests/test_study_analysis.py
uv run mypy src/connect_x_agent/study/analysis.py tests/test_study_analysis.py
```

Expected: all pass.

- [ ] **Step 10: Commit the turning-point slice**

```powershell
git add src/connect_x_agent/study/analysis.py tests/test_study_analysis.py
git commit -m "feat: analyze P2 tactical turning points"
```

---

### Task 4: Deterministic report serialization and analysis script

**Files:**
- Modify: `src/connect_x_agent/study/analysis.py`
- Modify: `tests/test_study_analysis.py`
- Create: `scripts/cx_gt_02_analyze.py`

**Interfaces:**
- Consumes: `StudyReport` and `load_jsonl()`.
- Produces: `report_to_dict(report: StudyReport) -> dict[str, object]`; script accepts input path and optional output path.

- [ ] **Step 1: Add RED deterministic serialization test**

Require two calls on the same report to yield equal JSON-ready structures with only lists/scalars/dicts at the serialization boundary:

```python
serialized = report_to_dict(report)
assert serialized == report_to_dict(report)
assert serialized["corpus"]["episodes"] == len(episodes)
assert isinstance(serialized["forced_defense"]["events"], list)
```

The library report remains typed dataclasses; dictionaries are permitted only at this explicit output boundary.

- [ ] **Step 2: Implement `report_to_dict()` and verify GREEN**

Use explicit field mapping or `dataclasses.asdict()` followed by no semantic reinterpretation. Keep tuple ordering deterministic from the analyzer.

Run:

```powershell
uv run pytest tests/test_study_analysis.py -q -k serialization
```

Expected: PASS.

- [ ] **Step 3: Create the thin script**

Use `argparse` from the standard library:

```python
from argparse import ArgumentParser
import json
from pathlib import Path

from connect_x_agent.study.analysis import analyze_episodes, report_to_dict
from connect_x_agent.study.loading import load_jsonl


def main() -> None:
    parser = ArgumentParser(description="Analyze recorded CX-GT-02 Player-2 episodes")
    parser.add_argument("input", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    episodes = load_jsonl(args.input)
    report = analyze_episodes(episodes)
    payload = report_to_dict(report)

    print(f"episodes: {report.corpus.episodes}")
    print(f"wins: {report.corpus.wins}")
    print(f"draws: {report.corpus.draws}")
    print(f"losses: {report.corpus.losses}")
    print(f"failures: {report.corpus.failures}")
    print(f"unique trajectories: {report.corpus.unique_trajectories}")
    print(f"duplicate trajectories: {report.corpus.duplicate_trajectories}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
```

Keep script output concise; detailed event data lives in optional JSON.

- [ ] **Step 4: Add a script smoke test without invoking Kaggle**

In `tests/test_study_analysis.py`, use `runpy.run_path()` plus `monkeypatch` for `sys.argv` against a temporary JSONL file created with `write_jsonl()`. Assert console output contains corpus totals and the optional JSON file parses to the expected episode count. This verifies the script remains a consumer only.

- [ ] **Step 5: Run focused script/analysis checks**

```powershell
uv run pytest tests/test_study_analysis.py -q
uv run ruff check src/connect_x_agent/study/analysis.py scripts/cx_gt_02_analyze.py tests/test_study_analysis.py
uv run mypy src/connect_x_agent/study/analysis.py scripts/cx_gt_02_analyze.py tests/test_study_analysis.py
```

Expected: all pass.

- [ ] **Step 6: Commit the output slice**

```powershell
git add src/connect_x_agent/study/analysis.py tests/test_study_analysis.py scripts/cx_gt_02_analyze.py
git commit -m "feat: expose deterministic study analysis report"
```

---

### Task 5: Repository acceptance and 100-game analysis rehearsal

**Files:**
- No production files expected unless a gate reveals a defect in CX-GT-02B.
- Runtime input: `evidence/cx_gt_02/raw/p2_random_smoke.jsonl` if present locally.
- Optional generated summary: keep under `evidence/cx_gt_02/summaries/` and do not commit unless separately approved.

**Interfaces:**
- Consumes: completed CX-GT-02B files.
- Produces: verification evidence only.

- [ ] **Step 1: Run focused CX-GT-02B tests**

```powershell
uv run pytest tests/test_study_loading.py tests/test_study_analysis.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the complete repository gate**

```powershell
Write-Host "`n=== FULL PYTEST ==="
uv run pytest -q

Write-Host "`n=== RUFF ==="
uv run ruff check .

Write-Host "`n=== MYPY ==="
uv run mypy src scripts

Write-Host "`n=== DIFF CHECK ==="
git diff --check

Write-Host "`n=== STATUS ==="
git status --short
```

Expected: all tests pass, Ruff clean, Mypy clean, diff check clean, and no unexpected working-tree files.

- [ ] **Step 3: Run the existing Random-smoke analysis rehearsal**

Only if the local ignored raw file exists:

```powershell
uv run python scripts/cx_gt_02_analyze.py `
  evidence/cx_gt_02/raw/p2_random_smoke.jsonl `
  --json-output evidence/cx_gt_02/summaries/p2_random_smoke_analysis.json
```

Inspect that:

- episode total equals the raw file's 100 episodes;
- wins/draws/losses/failures reconcile to 100;
- duplicate counts are between 0 and 99;
- opening/response columns are within recorded configuration bounds;
- forced-defence/counterattack/forcing counts do not exceed episode count where they are episode-level;
- no loader or analysis exception occurs.

These are pipeline-coherence checks only. Do not infer strategy quality from the Random sample.

- [ ] **Step 4: Re-run repository gates if any rehearsal defect required a code change**

Do not claim completion from a previous green run after modifying code. Repeat Step 1 and Step 2 from the modified HEAD.

- [ ] **Step 5: Inspect final scope before promotion**

```powershell
git diff main...HEAD --stat
git diff main...HEAD --name-only
git log --oneline main..HEAD
```

Expected implementation scope: approved design/plan docs plus exactly the five CX-GT-02B source/test/script files.

- [ ] **Step 6: Stop at the authorization boundary**

Do not merge, fast-forward `main`, push a new external branch state beyond already authorized branch writes, create a PR, or promote any empirical rule into the agent without explicit authorization after the final evidence review.

---

## Plan self-review

- **Spec coverage:** Loader, fail-loud schema integrity, immutable typed reports, duplicate detection, opening structure, forced defence, newly exposed P2 fork opportunities, P2 forcing moves, loss anatomy, failure handling, deterministic JSON output, Random-smoke rehearsal, and all acceptance gates each have an implementation task.
- **Placeholder scan:** No TBD/TODO or deferred implementation instructions remain.
- **Type consistency:** `load_jsonl() -> tuple[EpisodeRecord, ...]`, `analyze_episodes() -> StudyReport`, `trajectory_fingerprint() -> str`, and `report_to_dict() -> dict[str, object]` are used consistently across tasks. Event dataclass field names are fixed before their implementation tasks.
- **Scope:** No live-agent, Kaggle-runtime, search, ML/RL, plotting, or dependency changes are included.
