# CX-GT-02B Descriptive Player-2 Analysis Design

## Purpose

CX-GT-02B adds a reusable descriptive-analysis layer for Player-2 study data. It consumes recorded CX-GT-02 episodes and produces deterministic, inspectable summaries of openings, forced defence, counterattack opportunities, forcing moves, duplicate trajectories, and loss anatomy.

This subsystem is observational only. It must not choose agent actions, rank moves, train a model, alter the champion agent, or promote empirical correlations into strategy rules.

## Architectural boundary

The study pipeline becomes:

```text
raw JSONL
    -> study/loading.py
    -> tuple[EpisodeRecord, ...]
    -> study/analysis.py
    -> typed StudyReport
    -> scripts/cx_gt_02_analyze.py
    -> console summary + deterministic JSON summary
```

The existing runner remains responsible only for generating `EpisodeRecord` values. The analyzer never calls Kaggle and never mutates study records.

## Files

Create only:

```text
src/connect_x_agent/study/loading.py
src/connect_x_agent/study/analysis.py
tests/test_study_loading.py
tests/test_study_analysis.py
scripts/cx_gt_02_analyze.py
```

No pandas, NumPy, notebooks, plotting, machine learning, reinforcement learning, solver labels, or agent changes are part of CX-GT-02B.

## JSONL loading

`study/loading.py` reconstructs the existing immutable record types from JSONL:

- `EpisodeRecord`
- `PlyRecord`
- `PositionFeatures`

The loader validates the required schema instead of passing untyped dictionaries downstream. Invalid JSON, missing required fields, invalid enum values, malformed board/action shapes, or inconsistent field types must raise a descriptive exception rather than silently skipping the record.

The loader does not reinterpret outcomes, infer missing configuration, or repair records. Study integrity takes priority over permissive parsing.

Primary interface:

```python
def load_jsonl(path: Path) -> tuple[EpisodeRecord, ...]: ...
```

## Analysis data model

`study/analysis.py` defines immutable report structures. The exact decomposition may be refined during TDD, but the public result must expose the following categories without requiring callers to inspect internal dictionaries:

- corpus integrity
- opening statistics
- forced-defence statistics
- counterattack statistics
- forcing statistics
- loss anatomy

A top-level immutable `StudyReport` is the stable analysis result.

The analyzer accepts typed episodes directly:

```python
def analyze_episodes(episodes: tuple[EpisodeRecord, ...]) -> StudyReport: ...
```

It performs no file I/O.

## Corpus integrity metrics

The report records:

- episode count
- wins, draws, losses, failures
- distribution by opponent
- distribution by `(columns, rows, inarow)` configuration
- unique trajectory count
- duplicate trajectory count
- duplicate trajectory rate

### Trajectory fingerprint

A trajectory fingerprint is deterministic and ignores `episode_id`.

It is derived from:

- `columns`
- `rows`
- `inarow`
- ordered `(mark, action)` ply sequence

Opponent name and outcome are intentionally excluded from identity. If two episodes have the same game configuration and exact move sequence, they are the same trajectory for duplicate-detection purposes.

The fingerprint may be represented internally as a tuple or stable hash, but analysis tests must prove equal trajectories receive equal identities and different action sequences receive different identities.

## Opening metrics

The report records:

- P1 opening-column distribution
- final result by P1 opening column
- P2 first-response distribution
- P2 first response conditional on P1 opening column

Opening analysis uses the recorded ply sequence, not assumptions about standard 7-column geometry.

Episodes without a valid opening or first P2 move remain in corpus totals but are excluded from the relevant opening metric.

## State-transition rule

Each `PlyRecord` contains `features_before` but not `features_after`.

For a nonterminal move, the analyzer treats the next ply's `features_before` as the authoritative post-move state.

For a terminal last move, when no next ply exists, the analyzer may call the existing pure `position_features()` instrumentation on `board_after` using the episode's recorded configuration.

The analyzer must not duplicate game-theory logic already implemented by the instrumentation layer.

## Forced defence

A P2 forced-defence position exists when, before a P2 move:

```text
len(features_before.p2_surviving_replies) == 1
```

For each episode, analysis records:

- whether at least one forced-defence position occurred
- first forced-defence ply
- total forced-defence positions
- whether P2 selected the sole surviving reply at each such position
- whether the first forced defence was obeyed
- final episode result

This is a mechanical descriptor. It does not claim that surviving replies are globally optimal beyond the tactical horizon encoded by `surviving_replies`.

## P1 mistake creates P2 counterattack

A counterattack opportunity is detected on a P1 move when:

```text
P2 fork moves before P1 move == empty
P2 fork moves after P1 move != empty
```

For each episode, analysis records:

- whether such an opportunity occurred
- first P1 ply that created it
- the P2 fork moves available immediately afterward
- whether P2's next move selected one of those fork moves
- final episode result

The analyzer must use post-move structural features as defined by the state-transition rule above.

The term "mistake" is operational for this study only: it means a P1 move that newly exposes a P2 fork under current first-principles diagnostics. It is not yet a minimax error label.

## P2 begins forcing

A P2 forcing move exists when, after a P2 move:

```text
len(p1_surviving_replies) == 1
```

For each episode, analysis records:

- whether P2 created a forced P1 reply
- first P2 ply that did so
- the sole P1 surviving reply
- final result

Again, this is tactical structure, not a claim of solved-game optimality.

## Loss anatomy

For episodes with `candidate_result == "loss"`, the report records mechanically observable descriptors:

- first P2 turn with zero surviving replies
- whether P2 previously missed a forced defence
- whether P2 ever obtained a fork opportunity
- whether P2 ever created a forced P1 reply

These fields are descriptive and must not be phrased as causal explanations.

## Failure handling

Episodes with `candidate_result == "failure"` remain visible in corpus integrity metrics.

They are excluded from strategic descriptive metrics that require coherent completed gameplay unless the required ply data is present and the metric is unambiguous. The first implementation should prefer exclusion over inference.

Pipeline/loader errors are not converted into study failures; they raise.

## Output script

`scripts/cx_gt_02_analyze.py` is a thin consumer:

1. accept a JSONL path
2. load typed episodes
3. analyze them
4. print a concise human-readable summary
5. optionally write a deterministic JSON summary to a requested output path

The script contains no game-theory calculations and no metric definitions.

The first real-data rehearsal target is:

```text
evidence/cx_gt_02/raw/p2_random_smoke.jsonl
```

The resulting numbers validate the analysis pipeline only. They must not be treated as strategic truth because the corpus is a small Random-opponent smoke sample.

## Testing strategy

Implementation follows RED -> GREEN.

### Loader tests

Use temporary JSONL files to prove:

- one valid serialized episode round-trips back to the same typed record
- multiple lines preserve order
- malformed JSON raises
- missing required fields raise
- invalid enum values raise
- configuration and failure provenance are preserved

### Analysis tests

Use tiny synthetic `EpisodeRecord` values with deliberately understandable ply streams to prove independently:

- trajectory duplicate detection
- opening and first-response counting
- forced-defence detection and compliance
- newly created P2 fork detection
- P2 forcing-move detection
- loss-anatomy descriptors
- failure episodes remain in corpus counts without contaminating gameplay metrics

Tests should assert observable report values, not internal helper calls.

### Acceptance gates

Before promotion:

```text
focused CX-GT-02B tests
full pytest suite
ruff check .
mypy src scripts
git diff --check
clean working tree
```

After code acceptance, run the analyzer against the existing 100-game Random smoke JSONL and inspect the report for internal coherence before any larger campaign.

## Non-goals

CX-GT-02B does not:

- change CX-02, CX-03, or any agent
- choose or recommend moves
- add minimax or alpha-beta search
- label positions as wins/losses by solved-game truth
- train or fit parameters
- calculate statistical significance
- infer causality
- promote observations into challenger features
- run the 5,000-game study campaign

Those remain later stages after descriptive instrumentation is trusted.

## Success criteria

CX-GT-02B is complete when:

1. raw study JSONL can be reloaded into immutable typed records without loss of recorded fields;
2. deterministic descriptive metrics expose the agreed opening, forced-defence, counterattack, forcing, duplicate, and loss-anatomy views;
3. study-pipeline errors fail loudly;
4. the analysis layer remains independent of the live agent and Kaggle runtime;
5. all repository acceptance gates pass;
6. the existing Random smoke corpus can be analyzed repeatedly without rerunning games.
