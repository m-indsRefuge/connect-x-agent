# CX-06A: Specialized Transposition Search — Design

**Status:** Approved design, pre-implementation
**Project:** Kaggle Connect X Competition
**Phase:** CX-06A
**Geometry:** Standard Connect Four — 7 columns × 6 rows × 4 in a row

## 1. Purpose

CX-06A introduces the first optimized search engine for the Connect X competition.

Its purpose is to determine how much practical search depth can be gained from one isolated optimization: a per-search transposition table.

CX-06A is not a replacement for CX-04 as the source of truth. CX-04 remains the generic, deliberately straightforward reference solver. CX-06A is a competition-specialized implementation whose correctness is continuously checked against CX-04.

The architectural invariant is:

> Whenever CX-04 and CX-06A resolve the same position under the same search boundary, they must agree exactly on the root game value, every legal root move classification, and completeness.

Performance never overrides correctness.

## 2. Evidence Motivating CX-06A

CX-05 demonstrated that deeper search produces meaningful competitive value.

Observed local Negamax results:

- CX-05 pure depth 4: 7 wins, 0 draws, 13 losses — 35% win rate.
- CX-05 hybrid depth 4: 7 wins, 1 draw, 12 losses — 35% win rate.
- CX-05 pure depth 6: 10 wins, 0 draws, 10 losses — 50% win rate.
- CX-05 hybrid depth 6: 10 wins, 0 draws, 10 losses — 50% win rate.

Depth 6 also exposed game-theoretic information unavailable at depth 4.

In the hybrid pre-loss Negamax position at ply 15, depth 4 classified five moves as UNKNOWN and one as LOSS. Depth 6 proved five of the six legal moves LOSS and left only column 5 unresolved.

Representative naïve-solver timings were:

| Position | Depth 4 | Depth 5 | Depth 6 |
| --- | ---: | ---: | ---: |
| Empty opening | ~9.7 ms | ~76.2 ms | ~546.3 ms |
| Game 1 danger | ~4.2 ms | ~14.5 ms | ~54.5 ms |
| Negamax danger | ~11.7 ms | ~77.2 ms | ~345.2 ms |
| Hybrid pre-loss | ~15.6 ms | ~97.3 ms | ~528.9 ms |

Complete depth-6 Negamax probes recorded approximately:

- mean solve time: 473–475 ms per candidate move;
- median: 437–471 ms;
- p95: 926–961 ms;
- maximum: 1.20–1.33 seconds.

The evidence therefore supports deeper search while also showing that the reference recursion is approaching the practical competition move-time boundary.

Optimization is now justified.

## 3. Architectural Roles

### 3.1 CX-04 — Reference Oracle

`src/connect_x_agent/search.py` remains:

- geometry-generic;
- list-based;
- deliberately straightforward;
- authoritative for search semantics and correctness;
- unchanged by CX-06A.

CX-04 supports arbitrary valid Connect X geometries.

CX-06A must not import or depend on private CX-04 search or validation helpers.

It may reuse the public result contracts:

- `GameValue`
- `MoveAnalysis`
- `PositionSolution`

### 3.2 CX-06A — Competition Search Engine

CX-06A is specialized to exactly:

- `COLUMNS = 7`
- `ROWS = 6`
- `INAROW = 4`

Its job is to:

- preserve CX-04 game-theoretic semantics;
- search the standard competition geometry;
- eliminate repeated subtree evaluation using transposition caching;
- expose diagnostics that show whether the optimization is effective.

The optimized public solver does not accept geometry arguments.

## 4. Scope

CX-06A includes:

1. a fixed-geometry optimized solver;
2. independent structural validation for 7×6×4 positions;
3. a per-call transposition table;
4. search statistics for diagnostic use;
5. bounded and exhaustive oracle parity with CX-04;
6. fixed-position performance comparison against CX-04;
7. a depth-7 feasibility probe only after depth-6 correctness is established.

CX-06A explicitly excludes:

- alpha-beta pruning;
- heuristic evaluation;
- center-first or other move ordering;
- iterative deepening;
- symmetry canonicalization;
- persistent cross-turn caching;
- persistent cross-game caching;
- bitboards;
- opening books;
- tablebases;
- time-budget search;
- new playing-policy behavior;
- Kaggle submission packaging.

No excluded optimization may be introduced merely because it is convenient.

## 5. Proposed Files

Expected implementation scope:

```text
src/connect_x_agent/optimized_search.py
tests/test_optimized_search.py
tests/test_optimized_search_oracle.py
scripts/cx06a_cache_probe.py
```

CX-06A must not modify:

```text
src/connect_x_agent/search.py
```

unless a separately approved correctness defect is discovered in CX-04.

## 6. Public Solver Contract

The public entry point is:

```python
solve_optimized_position(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> PositionSolution
```

Geometry arguments are intentionally absent.

The solver always operates on 7×6×4.

Semantics match CX-04:

- values are relative to the player to move;
- legal root moves are classified individually;
- any root WIN makes the root WIN;
- otherwise any UNKNOWN makes the root UNKNOWN;
- otherwise any DRAW makes the root DRAW;
- otherwise the root is LOSS;
- terminal truth takes precedence over a zero-depth boundary;
- `max_depth=0` returns UNKNOWN for each legal root move unless the supplied position is already terminal;
- exhaustive search is requested with `max_depth=None`;
- invalid positions raise `ValueError`, never UNKNOWN.

Root move ordering remains deterministic and left-to-right.

## 7. Fixed-Geometry Validation

CX-06A owns an independent 7×6×4 validator.

It verifies:

- board length is exactly 42;
- every cell is `0`, `1`, or `2`;
- mark is `1` or `2`;
- gravity is valid in every column;
- piece counts are compatible with alternating play and the supplied mark;
- simultaneous winners are rejected;
- an existing winner must be the previous mover;
- terminal winner positions satisfy local last-move consistency.

CX-06A does not attempt a complete historical reachability proof.

Validation behavior must agree with CX-04 on the representative parity matrix, but exact exception wording does not need to match.

The independent validator must not call private CX-04 validation functions.

## 8. Search Representation

CX-06A deliberately retains the existing ordinary board representation for this phase.

Public input:

```python
list[int]
```

Transposition keys use an immutable board:

```python
tuple[int, ...]
```

CX-06A does not introduce bitboards or an alternative board representation in this phase because doing so would confound measurement of transposition caching.

## 9. Transposition Table

A fresh transposition table is created for every call to `solve_optimized_position`.

It is never shared:

- between solver calls;
- between turns;
- between games.

The conceptual key is:

```python
(
    tuple(board),
    mark,
    remaining_depth,
)
```

`remaining_depth` is either:

- a non-negative integer for bounded search; or
- `None` for exhaustive search.

The cached result is a `GameValue`, not a full `PositionSolution`.

`mark` is included because values are relative to the player to move.

`remaining_depth` is included because UNKNOWN means unresolved at a specific search boundary. A value calculated at one boundary must not be reused as though it were calculated at another.

UNKNOWN is therefore safe to cache.

No cache eviction policy is introduced in CX-06A.

## 10. Internal Search Flow

Conceptually:

```text
solve_optimized_position
    ↓
validate position
    ↓
resolve terminal root if applicable
    ↓
create empty transposition table
    ↓
iterate legal root moves left to right
    ↓
apply root move
    ↓
immediate win?
    ├─ yes → classify WIN
    └─ no  → invert cached child search value
    ↓
aggregate all root move classifications
    ↓
PositionSolution
```

Internal recursive search is conceptually:

```python
_search_value(
    board,
    mark,
    remaining_depth,
    table,
    stats,
) -> GameValue
```

The recursive game semantics must remain equivalent to CX-04.

The only intended search optimization in CX-06A is memoized reuse of already computed `(board, mark, remaining_depth)` values.

## 11. Search Statistics

Diagnostics must not alter `PositionSolution`.

A separate immutable statistics record should expose at least:

```text
nodes_visited
cache_hits
cache_misses
cache_entries
max_recursion_depth
```

For CX-06A:

- `nodes_visited` counts internal search requests before cache lookup;
- `cache_hits` counts requests served directly from the transposition table;
- `cache_misses` counts requests not already present in the table;
- `cache_entries` is the final table size;
- `max_recursion_depth` records the deepest recursive level entered.

A private or diagnostic helper may return:

```python
(PositionSolution, SearchStats)
```

for tests and benchmark scripts.

The normal production API returns only `PositionSolution`.

## 12. Oracle Parity

Oracle parity is a hard requirement.

For shared valid positions evaluated under the same depth boundary:

```python
cx04_solution == cx06_solution
```

must hold exactly.

Equality includes:

- root `value`;
- ordered `moves`;
- every `MoveAnalysis.column`;
- every `MoveAnalysis.value`;
- `complete`.

Parity testing must include partial searches containing UNKNOWN.

UNKNOWN semantics are part of the solver contract, not merely an implementation detail.

## 13. Exact Endgame Parity

Several standard 7×6 positions with sufficiently few empty cells must be solved exhaustively by both engines.

For those fixtures:

```python
solve_position(..., max_depth=None)
==
solve_optimized_position(..., max_depth=None)
```

must hold.

Exact fixtures must cover at least:

- a forced win;
- a forced draw where available;
- a forced loss;
- multiple legal root moves with different exact classifications.

## 14. Bounded-Depth Parity

Representative positions must be compared at several boundaries, including:

```text
max_depth = 0
max_depth = 1
max_depth = 4
max_depth = 6
```

The depth-frontier positions captured during CX-05 should be retained as high-value regression fixtures.

Terminal truth must remain identical at all depth boundaries.

## 15. Validation Parity

A representative invalid-position matrix must be run through both solvers.

Cases include:

- incorrect board length;
- invalid cell value;
- invalid mark;
- gravity violation;
- piece-count or turn inconsistency;
- simultaneous winners;
- winner inconsistent with the previous mover;
- terminal board inconsistent with a legal last move.

The required invariant is accept/reject parity.

Exception wording need not match.

## 16. Cache-Specific Verification

Correctness parity alone does not prove caching is active.

Tests must establish that a known transposition-rich search produces:

```text
cache_hits > 0
cache_entries > 0
```

A depth-isolation test must demonstrate that values calculated under one remaining-depth boundary cannot be reused as though they belonged to another.

A mutation-style verification must temporarily bypass cache lookup while leaving game semantics intact.

Under that mutation:

- oracle parity should remain green;
- the cache-effectiveness assertion must fail.

This demonstrates that the optimization test is genuinely sensitive to the optimization.

## 17. Performance Measurement

Wall-clock performance is evidence, not a correctness assertion.

Unit tests must not contain fragile machine-specific timing thresholds.

`scripts/cx06a_cache_probe.py` benchmarks the fixed CX-05 frontier positions against the CX-04 implementation.

The benchmark must report at least:

```text
solver
position
depth
elapsed time
result value
unknown root move count
```

CX-06A additionally reports:

```text
nodes visited
cache hits
cache misses
cache entries
```

Repeated trials should be used where practical so that one noisy timing does not determine the conclusion.

## 18. Optimization Usefulness Criterion

CX-06A is correct if it satisfies oracle and validation parity regardless of measured speedup.

Its optimization is considered materially useful if transposition caching produces a clear reduction in repeated work and wall-clock cost.

A roughly 25% or larger reduction on meaningful depth-6 workloads is a useful decision threshold, not a correctness gate.

If the gain is substantially smaller, the result is still retained as evidence that transposition caching alone is insufficient.

## 19. Depth-7 Feasibility Probe

Depth 7 is not enabled by default merely because CX-06A exists.

Only after:

1. depth-6 oracle parity passes;
2. cache-effectiveness tests pass;
3. depth-6 performance is measured;

may the benchmark harness probe depth 7.

The depth-7 question is:

> Does transposition caching make depth 7 practically searchable with useful timing headroom?

A successful probe does not automatically change the playing agent's depth.

Playing-policy changes require separate evidence and approval.

## 20. Acceptance Gates

CX-06A implementation acceptance requires:

### Correctness

- all CX-06A unit tests pass;
- bounded oracle parity passes;
- exhaustive endgame parity passes;
- validation parity passes;
- terminal precedence remains correct;
- no root classification differs from CX-04 on shared fixtures.

### Optimization

- a transposition-rich fixture records cache hits;
- cache entries are populated;
- depth-isolated cache semantics are verified;
- mutation verification proves the cache-effectiveness test is meaningful.

### Repository quality

- full `pytest` passes;
- Ruff passes;
- Mypy passes for `src` and `scripts`;
- `git diff --check` is clean;
- scope audit contains only approved CX-06A files plus its design and plan documentation.

### Evidence

- depth-6 fixed-position benchmark is recorded;
- node/cache statistics are recorded;
- depth 7 is probed only after the depth-6 gates pass.

No performance result can waive a correctness failure.

## 21. Future Optimization Sequence

If CX-06A demonstrates correct but insufficient acceleration, later phases may evaluate additional optimizations one at a time.

Preferred progression:

```text
CX-06A
7×6×4 specialization + transposition cache

CX-06B
+ alpha-beta pruning

CX-06C
+ deliberate move ordering and/or symmetry

later, only if justified
+ compact bitboard representation
+ iterative deepening / time-budget control
```

Every future optimized engine remains subject to the CX-04 oracle invariant.

## 22. Non-Goals

CX-06A does not attempt to:

- solve the empty standard Connect Four board exhaustively;
- prove theoretical optimality of the competition agent;
- replace the generic reference solver;
- improve UNKNOWN positions with heuristic evaluation;
- change CX-05 tactical behavior;
- select the final Kaggle submission agent.

Its purpose is narrower:

> Preserve CX-04 truth while measuring exactly how much per-search transposition caching improves practical search depth on standard 7×6×4 Connect Four.
