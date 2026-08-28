# CX-06A Specialized Transposition Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed 7×6×4 Connect Four search engine that preserves CX-04 game-theoretic semantics exactly while measuring the isolated benefit of a per-search transposition table.

**Architecture:** CX-04 remains unchanged as the generic reference oracle. CX-06A lives in a new `optimized_search.py` module, owns independent fixed-geometry validation, uses ordinary row-major boards and tuple cache keys, and adds only one search optimization: memoized node values scoped to a single solver invocation. Correctness parity is established before performance evidence or depth-7 probing.

**Tech Stack:** Python 3.12+, pytest, Ruff, Mypy, uv, standard-library `dataclasses`, `time.perf_counter`, `statistics.median`, `argparse`.

**Spec:** `docs/superpowers/specs/2026-08-28-cx-06a-transposition-search-design.md`

## Global Constraints

- CX-04 `src/connect_x_agent/search.py` remains unchanged and authoritative.
- CX-06A supports exactly 7 columns × 6 rows × 4 in a row.
- Reuse only the public CX-04 contracts `GameValue`, `MoveAnalysis`, and `PositionSolution`; do not import CX-04 private search or validation helpers.
- Public optimized solver signature is `solve_optimized_position(board: list[int], mark: int, max_depth: int | None = None) -> PositionSolution`.
- Values remain relative to the player to move.
- Root aggregation priority remains WIN → UNKNOWN → DRAW → LOSS.
- Terminal truth takes precedence over the depth boundary.
- A fresh transposition table is created for every solver call.
- Cache key is `(tuple(board), mark, remaining_depth)`.
- Cached values are `GameValue`, including safe caching of UNKNOWN because remaining depth is in the key.
- No alpha-beta pruning, heuristic evaluation, center-first ordering, iterative deepening, symmetry canonicalization, persistent cache, bitboards, opening books, tablebases, time-budget search, playing-policy changes, or Kaggle packaging.
- Wall-clock timings are evidence only; no machine-specific timing assertion belongs in unit tests.
- Do not modify or stage unrelated `docs/training/` or `evidence/cx_gt_02/`.
- Every production-code slice follows RED → verify correct failure → minimal GREEN → focused tests → Ruff → Mypy → `git diff --check`.
- Every task commit stages only the files named by that task.

---

## File Structure

### Create: `src/connect_x_agent/optimized_search.py`

Responsibility:
- fixed 7×6×4 constants;
- local fixed-geometry board primitives;
- independent structural validation;
- public optimized solver;
- internal recursive value search;
- per-call transposition table;
- immutable `SearchStats`;
- diagnostic `solve_optimized_position_with_stats`.

Public interfaces produced by the completed module:

```python
COLUMNS: int = 7
ROWS: int = 6
INAROW: int = 4
BOARD_SIZE: int = 42

@dataclass(frozen=True)
class SearchStats:
    nodes_visited: int
    cache_hits: int
    cache_misses: int
    cache_entries: int
    max_recursion_depth: int

def solve_optimized_position(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> PositionSolution: ...

def solve_optimized_position_with_stats(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> tuple[PositionSolution, SearchStats]: ...
```

### Create: `tests/test_optimized_search.py`

Responsibility:
- fixed-geometry public contract;
- terminal and zero-depth semantics;
- independent validation behavior;
- cache statistics;
- key depth isolation;
- per-call cache lifetime;
- benchmark-script structure smoke test.

### Create: `tests/test_optimized_search_oracle.py`

Responsibility:
- bounded parity against CX-04;
- depth-frontier regression fixtures;
- exact 7×6 endgame parity;
- forced win/draw/loss exact fixtures;
- mixed root move classifications;
- validation accept/reject parity.

### Create: `scripts/cx06a_cache_probe.py`

Responsibility:
- repeated wall-clock comparison of CX-04 and CX-06A;
- fixed CX-05 frontier positions;
- depth-6 benchmark;
- optimized-only depth-7 feasibility mode;
- report result value, UNKNOWN count, timings, and cache/node statistics.

### Do not modify

```text
src/connect_x_agent/search.py
src/connect_x_agent/solver_agent.py
src/connect_x_agent/tactical.py
src/connect_x_agent/arena.py
```

---

### Task 0: Preflight the Approved Architecture Checkpoint

**Files:**
- Existing: `docs/superpowers/specs/2026-08-28-cx-06a-transposition-search-design.md`
- Create later: `docs/superpowers/plans/2026-08-28-cx-06a-transposition-search.md`

**Interfaces:**
- Consumes: approved CX-06A design spec.
- Produces: a verified branch baseline from which implementation begins.

- [ ] **Step 1: Verify branch, HEAD, recent commits, and status**

Run:

```powershell
Write-Host "`n=== BRANCH / HEAD ==="
git branch --show-current
git rev-parse HEAD

Write-Host "`n=== RECENT COMMITS ==="
git log -3 --oneline

Write-Host "`n=== STATUS ==="
git status --short
```

Expected:
- branch is `build/cx-06a-transposition-search`;
- the approved design spec is already committed before implementation begins;
- the implementation-plan file may be untracked until its plan-only commit;
- `docs/training/` and `evidence/cx_gt_02/` may remain untracked and must not be touched.

If the design spec is still staged or uncommitted, stop and complete the design-only commit before continuing.

- [ ] **Step 2: Run the repository baseline gate**

Run:

```powershell
uv run pytest -q
if ($LASTEXITCODE -ne 0) { throw "Baseline pytest failed. Stop." }

uv run ruff check .
if ($LASTEXITCODE -ne 0) { throw "Baseline Ruff failed. Stop." }

uv run mypy src scripts
if ($LASTEXITCODE -ne 0) { throw "Baseline Mypy failed. Stop." }

git diff --check
if ($LASTEXITCODE -ne 0) { throw "Baseline diff check failed. Stop." }
```

Expected: all gates pass. Existing third-party OpenSpiel startup noise is not a Connect X failure unless it changes process exit status or game rewards.

- [ ] **Step 3: Commit only this implementation plan**

Run after saving this file at the spec-defined plan path:

```powershell
git add -- docs/superpowers/plans/2026-08-28-cx-06a-transposition-search.md
git diff --cached --name-only
git diff --cached --check
```

Expected staged file list:

```text
docs/superpowers/plans/2026-08-28-cx-06a-transposition-search.md
```

Commit:

```powershell
git commit -m "docs: add CX-06A implementation plan"
```

Do not stage `docs/training/` or `evidence/cx_gt_02/`.

---

### Task 1: Establish Fixed-Geometry Solver Semantics

**Files:**
- Create: `src/connect_x_agent/optimized_search.py`
- Create: `tests/test_optimized_search.py`

**Interfaces:**
- Consumes: `GameValue`, `MoveAnalysis`, `PositionSolution` from `connect_x_agent.search`.
- Produces:
  - constants `COLUMNS`, `ROWS`, `INAROW`, `BOARD_SIZE`;
  - `solve_optimized_position(...)`;
  - local board primitives `_legal_columns`, `_drop_piece`, `_is_win`;
  - terminal and zero-depth behavior matching CX-04.

- [ ] **Step 1: Write the first failing tests**

Create `tests/test_optimized_search.py` with:

```python
from connect_x_agent.search import MoveAnalysis, PositionSolution
from connect_x_agent.optimized_search import (
    BOARD_SIZE,
    COLUMNS,
    INAROW,
    ROWS,
    solve_optimized_position,
)


FULL_DRAW_BOARD = [
    1, 1, 2, 2, 2, 1, 1,
    2, 2, 1, 1, 2, 2, 2,
    1, 1, 2, 2, 1, 2, 1,
    2, 2, 2, 1, 1, 2, 1,
    2, 1, 2, 1, 2, 1, 1,
    1, 2, 1, 1, 2, 1, 2,
]


def test_optimized_search_is_fixed_to_standard_connect_four() -> None:
    assert (COLUMNS, ROWS, INAROW, BOARD_SIZE) == (7, 6, 4, 42)


def test_zero_depth_returns_unknown_for_every_legal_root_move() -> None:
    solution = solve_optimized_position(
        [0] * 42,
        mark=1,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="unknown",
        moves=tuple(
            MoveAnalysis(column, "unknown")
            for column in range(7)
        ),
        complete=False,
    )


def test_terminal_previous_player_win_precedes_zero_depth() -> None:
    board = [0] * 35 + [1, 1, 1, 1, 2, 2, 2]

    solution = solve_optimized_position(
        board,
        mark=2,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="loss",
        moves=(),
        complete=True,
    )


def test_full_no_win_board_is_terminal_draw() -> None:
    solution = solve_optimized_position(
        FULL_DRAW_BOARD,
        mark=1,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="draw",
        moves=(),
        complete=True,
    )


def test_solver_does_not_mutate_input_board() -> None:
    board = [0] * 42
    before = board.copy()

    solve_optimized_position(
        board,
        mark=1,
        max_depth=0,
    )

    assert board == before
```

- [ ] **Step 2: Run RED and verify the correct failure**

Run:

```powershell
uv run pytest tests/test_optimized_search.py -q
```

Expected: collection/import fails because `connect_x_agent.optimized_search` does not exist.

A syntax error or unrelated import failure is not an acceptable RED state.

- [ ] **Step 3: Implement only fixed constants, board primitives, terminal truth, aggregation helpers, depth validation, and zero-depth root behavior**

Create `src/connect_x_agent/optimized_search.py` with this structure:

```python
from __future__ import annotations

from connect_x_agent.search import (
    GameValue,
    MoveAnalysis,
    PositionSolution,
)

COLUMNS = 7
ROWS = 6
INAROW = 4
BOARD_SIZE = COLUMNS * ROWS


def _legal_columns(board: list[int]) -> list[int]:
    return [
        column
        for column in range(COLUMNS)
        if board[column] == 0
    ]


def _drop_piece(
    board: list[int],
    column: int,
    mark: int,
) -> list[int]:
    child = board.copy()

    for row in range(ROWS - 1, -1, -1):
        index = row * COLUMNS + column
        if child[index] == 0:
            child[index] = mark
            return child

    raise ValueError(f"Column {column} is full")


def _is_win(
    board: list[int],
    mark: int,
) -> bool:
    directions = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    )

    for row in range(ROWS):
        for column in range(COLUMNS):
            if board[row * COLUMNS + column] != mark:
                continue

            for row_step, column_step in directions:
                if all(
                    0 <= row + row_step * offset < ROWS
                    and 0 <= column + column_step * offset < COLUMNS
                    and board[
                        (row + row_step * offset) * COLUMNS
                        + column
                        + column_step * offset
                    ]
                    == mark
                    for offset in range(1, INAROW)
                ):
                    return True

    return False


def _invert_value(value: GameValue) -> GameValue:
    if value == "win":
        return "loss"
    if value == "loss":
        return "win"
    return value


def _aggregate_values(
    values: tuple[GameValue, ...],
) -> GameValue:
    if "win" in values:
        return "win"
    if "unknown" in values:
        return "unknown"
    if "draw" in values:
        return "draw"
    return "loss"


def _validate_max_depth(
    max_depth: int | None,
) -> None:
    if max_depth is None:
        return
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise ValueError("max_depth must be None or a non-negative integer")
    if max_depth < 0:
        raise ValueError("max_depth must be None or a non-negative integer")


def _terminal_value(
    board: list[int],
    mark: int,
) -> GameValue | None:
    previous_mark = 3 - mark

    if _is_win(board, previous_mark):
        return "loss"

    if not _legal_columns(board):
        return "draw"

    return None


def solve_optimized_position(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> PositionSolution:
    _validate_max_depth(max_depth)

    terminal = _terminal_value(board, mark)
    if terminal is not None:
        return PositionSolution(
            value=terminal,
            moves=(),
            complete=True,
        )

    legal = _legal_columns(board)

    if max_depth == 0:
        moves = tuple(
            MoveAnalysis(column, "unknown")
            for column in legal
        )
        return PositionSolution(
            value="unknown",
            moves=moves,
            complete=False,
        )

    raise NotImplementedError(
        "Recursive optimized search is introduced in Task 3"
    )
```

At this task boundary, only valid fixed-geometry fixtures are exercised. Full structural validation is intentionally introduced in Task 2 before recursive search.

- [ ] **Step 4: Run focused GREEN**

Run:

```powershell
uv run pytest tests/test_optimized_search.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Run code-quality gates**

Run:

```powershell
uv run ruff check `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py

uv run mypy `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py

git diff --check -- `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py
```

Expected: all clean.

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add -- `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py

git diff --cached --name-only
git diff --cached --check
git commit -m "feat: establish CX-06A fixed search semantics"
```

---

### Task 2: Add Independent Fixed-Geometry Validation

**Files:**
- Modify: `src/connect_x_agent/optimized_search.py`
- Modify: `tests/test_optimized_search.py`
- Create: `tests/test_optimized_search_oracle.py`

**Interfaces:**
- Consumes: Task 1 fixed board primitives and public solver.
- Produces:
  - `_validate_position(board, mark) -> None`;
  - independent structural rejection rules;
  - accept/reject parity against CX-04 on representative invalid positions.

- [ ] **Step 1: Add RED tests for invalid local structure**

Append to `tests/test_optimized_search.py`:

```python
import pytest


@pytest.mark.parametrize(
    ("board", "mark"),
    [
        ([0] * 41, 1),
        ([0] * 41 + [3], 1),
        ([1] + [0] * 41, 2),
        ([0] * 35 + [1, 0, 0, 0, 0, 0, 0], 1),
        (
            [0] * 28
            + [2, 2, 2, 2, 1, 1, 1]
            + [1, 1, 1, 1, 2, 2, 2],
            1,
        ),
        (
            [0] * 28
            + [0, 0, 0, 0, 2, 0, 0]
            + [1, 1, 1, 1, 2, 2, 2],
            1,
        ),
        (
            [
                0, 0, 2, 0, 0, 0, 0,
                0, 0, 1, 1, 2, 1, 2,
                0, 0, 2, 2, 1, 2, 1,
                0, 1, 1, 1, 2, 1, 1,
                2, 1, 1, 1, 2, 1, 1,
                2, 2, 2, 1, 2, 2, 2,
            ],
            2,
        ),
    ],
    ids=[
        "wrong-board-length",
        "invalid-cell",
        "floating-piece",
        "turn-count-mismatch",
        "simultaneous-winners",
        "winner-not-previous-mover",
        "winner-not-created-by-legal-last-move",
    ],
)
def test_invalid_positions_raise_value_error(
    board: list[int],
    mark: int,
) -> None:
    with pytest.raises(ValueError):
        solve_optimized_position(
            board,
            mark=mark,
            max_depth=0,
        )


@pytest.mark.parametrize("mark", [0, 3])
def test_invalid_mark_raises_value_error(mark: int) -> None:
    with pytest.raises(ValueError):
        solve_optimized_position(
            [0] * 42,
            mark=mark,
            max_depth=0,
        )


@pytest.mark.parametrize("max_depth", [-1, -2])
def test_negative_depth_raises_value_error(max_depth: int) -> None:
    with pytest.raises(ValueError):
        solve_optimized_position(
            [0] * 42,
            mark=1,
            max_depth=max_depth,
        )
```

- [ ] **Step 2: Run RED**

Run:

```powershell
uv run pytest tests/test_optimized_search.py -q
```

Expected: new invalid-position tests fail because Task 1 does not yet perform structural validation.

- [ ] **Step 3: Implement the independent validator**

Add these helpers to `optimized_search.py`:

```python
def _validate_gravity(board: list[int]) -> None:
    for column in range(COLUMNS):
        seen_empty = False

        for row in range(ROWS - 1, -1, -1):
            value = board[row * COLUMNS + column]

            if value == 0:
                seen_empty = True
            elif seen_empty:
                raise ValueError("Board violates gravity")


def _topmost_occupied_index(
    board: list[int],
    column: int,
) -> int | None:
    for row in range(ROWS):
        index = row * COLUMNS + column
        if board[index] != 0:
            return index
    return None


def _winner_has_consistent_last_move(
    board: list[int],
    winner: int,
) -> bool:
    for column in range(COLUMNS):
        index = _topmost_occupied_index(
            board,
            column,
        )

        if index is None or board[index] != winner:
            continue

        predecessor = board.copy()
        predecessor[index] = 0

        if not _is_win(predecessor, winner):
            return True

    return False


def _validate_position(
    board: list[int],
    mark: int,
) -> None:
    if len(board) != BOARD_SIZE:
        raise ValueError(
            f"Board must contain exactly {BOARD_SIZE} cells"
        )

    if any(cell not in (0, 1, 2) for cell in board):
        raise ValueError("Board cells must be 0, 1, or 2")

    if mark not in (1, 2):
        raise ValueError("mark must be 1 or 2")

    _validate_gravity(board)

    player_one = board.count(1)
    player_two = board.count(2)

    if mark == 1 and player_one != player_two:
        raise ValueError(
            "Piece counts are inconsistent with mark 1 to move"
        )

    if mark == 2 and player_one != player_two + 1:
        raise ValueError(
            "Piece counts are inconsistent with mark 2 to move"
        )

    player_one_wins = _is_win(board, 1)
    player_two_wins = _is_win(board, 2)

    if player_one_wins and player_two_wins:
        raise ValueError("Both players cannot be winners")

    winner: int | None = None

    if player_one_wins:
        winner = 1
    elif player_two_wins:
        winner = 2

    if winner is None:
        return

    previous_mark = 3 - mark

    if winner != previous_mark:
        raise ValueError(
            "Winner must be the previous mover"
        )

    if not _winner_has_consistent_last_move(
        board,
        winner,
    ):
        raise ValueError(
            "Winner is inconsistent with a legal last move"
        )
```

Call `_validate_position(board, mark)` at the start of `solve_optimized_position`, before `_terminal_value`.

Keep `_validate_max_depth` separate and call it immediately after position validation.

- [ ] **Step 4: Run local validation GREEN**

Run:

```powershell
uv run pytest tests/test_optimized_search.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Add validation parity against CX-04**

Create `tests/test_optimized_search_oracle.py`:

```python
import pytest

from connect_x_agent.optimized_search import (
    solve_optimized_position,
)
from connect_x_agent.search import solve_position


INVALID_CASES = (
    ([0] * 41, 1),
    ([0] * 41 + [3], 1),
    ([1] + [0] * 41, 2),
    ([0] * 35 + [1, 0, 0, 0, 0, 0, 0], 1),
    (
        [0] * 28
        + [2, 2, 2, 2, 1, 1, 1]
        + [1, 1, 1, 1, 2, 2, 2],
        1,
    ),
    (
        [0] * 28
        + [0, 0, 0, 0, 2, 0, 0]
        + [1, 1, 1, 1, 2, 2, 2],
        1,
    ),
    (
        [
            0, 0, 2, 0, 0, 0, 0,
            0, 0, 1, 1, 2, 1, 2,
            0, 0, 2, 2, 1, 2, 1,
            0, 1, 1, 1, 2, 1, 1,
            2, 1, 1, 1, 2, 1, 1,
            2, 2, 2, 1, 2, 2, 2,
        ],
        2,
    ),
)


def _reference_rejects(
    board: list[int],
    mark: int,
) -> bool:
    try:
        solve_position(
            board,
            mark=mark,
            columns=7,
            rows=6,
            inarow=4,
            max_depth=0,
        )
    except ValueError:
        return True
    return False


@pytest.mark.parametrize(("board", "mark"), INVALID_CASES)
def test_validation_accept_reject_parity(
    board: list[int],
    mark: int,
) -> None:
    reference_rejects = _reference_rejects(
        board,
        mark,
    )

    try:
        solve_optimized_position(
            board,
            mark=mark,
            max_depth=0,
        )
    except ValueError:
        optimized_rejects = True
    else:
        optimized_rejects = False

    assert optimized_rejects == reference_rejects
    assert reference_rejects is True
```

Run:

```powershell
uv run pytest tests/test_optimized_search_oracle.py -q
```

Expected: PASS. If any case differs, stop and inspect CX-04 behavior rather than weakening the parity test.

- [ ] **Step 6: Run quality gates**

```powershell
uv run pytest `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py `
    -q

uv run ruff check `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py

uv run mypy `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py

git diff --check
```

Expected: all clean.

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py

git diff --cached --name-only
git diff --cached --check
git commit -m "feat: validate CX-06A fixed positions"
```

---

### Task 3: Implement Recursive Search and Prove CX-04 Oracle Parity Before Caching

**Files:**
- Modify: `src/connect_x_agent/optimized_search.py`
- Modify: `tests/test_optimized_search_oracle.py`

**Interfaces:**
- Consumes: Task 2 validated fixed boards.
- Produces:
  - `_search_value_uncached(...) -> GameValue`;
  - complete bounded and exhaustive specialized solver semantics;
  - exact parity with CX-04 before introducing the transposition table.

- [ ] **Step 1: Add bounded-depth RED parity fixtures**

Append to `tests/test_optimized_search_oracle.py`:

```python
from connect_x_agent.search import (
    MoveAnalysis,
    PositionSolution,
)


FRONTIER_CASES = (
    (
        "empty",
        [0] * 42,
        1,
    ),
    (
        "game1-danger-ply20",
        [
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 2, 0, 0, 0,
            1, 1, 1, 2, 0, 0, 0,
        ],
        1,
    ),
    (
        "negamax-danger-ply17",
        [
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 1,
            2, 1, 2, 0, 1, 0, 1,
            2, 1, 2, 2, 1, 1, 1,
        ],
        2,
    ),
    (
        "hybrid-pre-loss-ply15",
        [
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0,
            2, 2, 0, 0, 0, 0, 0,
            2, 1, 0, 0, 1, 0, 1,
            2, 2, 1, 0, 1, 0, 1,
        ],
        2,
    ),
)


@pytest.mark.parametrize(
    ("name", "board", "mark"),
    FRONTIER_CASES,
    ids=[case[0] for case in FRONTIER_CASES],
)
@pytest.mark.parametrize("max_depth", [0, 1, 4, 6])
def test_bounded_search_matches_cx04(
    name: str,
    board: list[int],
    mark: int,
    max_depth: int,
) -> None:
    del name

    reference = solve_position(
        board,
        mark=mark,
        columns=7,
        rows=6,
        inarow=4,
        max_depth=max_depth,
    )
    optimized = solve_optimized_position(
        board,
        mark=mark,
        max_depth=max_depth,
    )

    assert optimized == reference


def test_depth_one_preserves_proven_win_with_unknown_alternatives() -> None:
    board = [0] * 42
    board[35:42] = [1, 1, 1, 0, 2, 2, 2]

    optimized = solve_optimized_position(
        board,
        mark=1,
        max_depth=1,
    )

    assert optimized == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(0, "unknown"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "unknown"),
            MoveAnalysis(3, "win"),
            MoveAnalysis(4, "unknown"),
            MoveAnalysis(5, "unknown"),
            MoveAnalysis(6, "unknown"),
        ),
        complete=False,
    )
```

- [ ] **Step 2: Run RED**

Run:

```powershell
uv run pytest `
    tests/test_optimized_search_oracle.py::test_bounded_search_matches_cx04 `
    tests/test_optimized_search_oracle.py::test_depth_one_preserves_proven_win_with_unknown_alternatives `
    -q
```

Expected: cases with depth greater than zero fail because Task 1 deliberately raises `NotImplementedError`.

- [ ] **Step 3: Implement uncached recursive semantics**

Add:

```python
def _search_value_uncached(
    board: list[int],
    mark: int,
    remaining_depth: int | None,
) -> GameValue:
    terminal = _terminal_value(
        board,
        mark,
    )
    if terminal is not None:
        return terminal

    if remaining_depth == 0:
        return "unknown"

    child_depth = (
        None
        if remaining_depth is None
        else remaining_depth - 1
    )

    move_values: list[GameValue] = []

    for column in _legal_columns(board):
        child = _drop_piece(
            board,
            column,
            mark,
        )

        if _is_win(child, mark):
            move_values.append("win")
            continue

        child_value = _search_value_uncached(
            child,
            3 - mark,
            child_depth,
        )
        move_values.append(
            _invert_value(child_value)
        )

    return _aggregate_values(
        tuple(move_values)
    )


def _solve_uncached(
    board: list[int],
    mark: int,
    max_depth: int | None,
) -> PositionSolution:
    terminal = _terminal_value(
        board,
        mark,
    )
    if terminal is not None:
        return PositionSolution(
            value=terminal,
            moves=(),
            complete=True,
        )

    legal = _legal_columns(board)

    if max_depth == 0:
        moves = tuple(
            MoveAnalysis(column, "unknown")
            for column in legal
        )
        return PositionSolution(
            value="unknown",
            moves=moves,
            complete=False,
        )

    child_depth = (
        None
        if max_depth is None
        else max_depth - 1
    )

    moves: list[MoveAnalysis] = []

    for column in legal:
        child = _drop_piece(
            board,
            column,
            mark,
        )

        if _is_win(child, mark):
            value: GameValue = "win"
        else:
            value = _invert_value(
                _search_value_uncached(
                    child,
                    3 - mark,
                    child_depth,
                )
            )

        moves.append(
            MoveAnalysis(
                column=column,
                value=value,
            )
        )

    move_tuple = tuple(moves)

    return PositionSolution(
        value=_aggregate_values(
            tuple(
                move.value
                for move in move_tuple
            )
        ),
        moves=move_tuple,
        complete=all(
            move.value != "unknown"
            for move in move_tuple
        ),
    )
```

Update `solve_optimized_position` after validation to return `_solve_uncached(board, mark, max_depth)`.

- [ ] **Step 4: Run bounded GREEN**

```powershell
uv run pytest `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py::test_bounded_search_matches_cx04 `
    tests/test_optimized_search_oracle.py::test_depth_one_preserves_proven_win_with_unknown_alternatives `
    -q
```

Expected: PASS.

- [ ] **Step 5: Add exact 7×6 endgame parity tests**

Append these fixtures and test:

```python
EXACT_CASES = (
    (
        "forced-win-mixed-root-values",
        [
            0, 2, 0, 0, 2, 0, 0,
            0, 1, 0, 0, 2, 2, 1,
            0, 2, 0, 0, 1, 2, 2,
            2, 1, 1, 0, 2, 1, 1,
            2, 2, 1, 2, 1, 1, 1,
            2, 1, 1, 1, 2, 2, 1,
        ],
        1,
        PositionSolution(
            value="win",
            moves=(
                MoveAnalysis(0, "win"),
                MoveAnalysis(2, "win"),
                MoveAnalysis(3, "win"),
                MoveAnalysis(5, "loss"),
                MoveAnalysis(6, "loss"),
            ),
            complete=True,
        ),
    ),
    (
        "forced-draw",
        [
            1, 1, 2, 0, 1, 0, 0,
            2, 1, 1, 0, 2, 0, 0,
            2, 1, 2, 0, 2, 0, 0,
            1, 2, 1, 0, 1, 2, 0,
            2, 2, 1, 2, 1, 1, 0,
            1, 2, 1, 2, 2, 2, 1,
        ],
        1,
        PositionSolution(
            value="draw",
            moves=(
                MoveAnalysis(3, "loss"),
                MoveAnalysis(5, "draw"),
                MoveAnalysis(6, "loss"),
            ),
            complete=True,
        ),
    ),
    (
        "forced-loss",
        [
            1, 2, 1, 0, 0, 2, 0,
            2, 1, 1, 0, 0, 1, 0,
            1, 2, 2, 2, 0, 1, 0,
            2, 1, 1, 2, 0, 2, 0,
            1, 2, 2, 2, 0, 1, 0,
            2, 1, 1, 1, 2, 1, 2,
        ],
        1,
        PositionSolution(
            value="loss",
            moves=(
                MoveAnalysis(3, "loss"),
                MoveAnalysis(4, "loss"),
                MoveAnalysis(6, "loss"),
            ),
            complete=True,
        ),
    ),
)


@pytest.mark.parametrize(
    ("name", "board", "mark", "expected"),
    EXACT_CASES,
    ids=[case[0] for case in EXACT_CASES],
)
def test_exhaustive_standard_positions_match_cx04(
    name: str,
    board: list[int],
    mark: int,
    expected: PositionSolution,
) -> None:
    del name

    reference = solve_position(
        board,
        mark=mark,
        columns=7,
        rows=6,
        inarow=4,
        max_depth=None,
    )
    optimized = solve_optimized_position(
        board,
        mark=mark,
        max_depth=None,
    )

    assert reference == expected
    assert optimized == reference
```

Run:

```powershell
uv run pytest `
    tests/test_optimized_search_oracle.py::test_exhaustive_standard_positions_match_cx04 `
    -q
```

Expected: PASS. If CX-04 disagrees with any literal expected classification, stop and investigate the fixture rather than changing CX-06A to fit the literal.

- [ ] **Step 6: Run Task 3 quality gates**

```powershell
uv run pytest `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py `
    -q

uv run ruff check `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py

uv run mypy `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py

git diff --check
```

Expected: all clean.

- [ ] **Step 7: Commit the uncached parity checkpoint**

```powershell
git add -- `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search_oracle.py

git diff --cached --name-only
git diff --cached --check
git commit -m "feat: match CX-04 with specialized search"
```

This checkpoint is intentionally still uncached. It proves the specialized game semantics independently before the optimization is introduced.

---

### Task 4: Add Per-Search Transposition Caching and Immutable Search Statistics

**Files:**
- Modify: `src/connect_x_agent/optimized_search.py`
- Modify: `tests/test_optimized_search.py`
- Verify: `tests/test_optimized_search_oracle.py`

**Interfaces:**
- Consumes: Task 3 proven uncached solver semantics.
- Produces:
  - immutable `SearchStats`;
  - diagnostic `solve_optimized_position_with_stats`;
  - cache key `(tuple(board), mark, remaining_depth)`;
  - fresh table per invocation;
  - cached `GameValue` including UNKNOWN;
  - identical public `solve_optimized_position` result.

- [ ] **Step 1: Write cache/statistics RED tests**

Append to `tests/test_optimized_search.py`:

```python
from connect_x_agent.optimized_search import (
    SearchStats,
    _cache_key,
    solve_optimized_position_with_stats,
)


def test_cache_key_isolates_depth_and_player() -> None:
    board = [0] * 42

    depth_four = _cache_key(
        board,
        mark=1,
        remaining_depth=4,
    )
    depth_six = _cache_key(
        board,
        mark=1,
        remaining_depth=6,
    )
    other_player = _cache_key(
        board,
        mark=2,
        remaining_depth=4,
    )
    exhaustive = _cache_key(
        board,
        mark=1,
        remaining_depth=None,
    )

    assert depth_four != depth_six
    assert depth_four != other_player
    assert depth_four != exhaustive


def test_transposition_rich_search_records_cache_hits() -> None:
    solution, stats = solve_optimized_position_with_stats(
        [0] * 42,
        mark=1,
        max_depth=4,
    )

    assert solution.value == "unknown"
    assert isinstance(stats, SearchStats)
    assert stats.nodes_visited > 0
    assert stats.cache_hits > 0
    assert stats.cache_misses > 0
    assert stats.cache_entries > 0
    assert stats.cache_entries <= stats.cache_misses


def test_transposition_table_is_fresh_per_solver_call() -> None:
    first_solution, first_stats = solve_optimized_position_with_stats(
        [0] * 42,
        mark=1,
        max_depth=4,
    )
    second_solution, second_stats = solve_optimized_position_with_stats(
        [0] * 42,
        mark=1,
        max_depth=4,
    )

    assert first_solution == second_solution
    assert first_stats == second_stats


def test_production_api_matches_diagnostic_api() -> None:
    board = [0] * 42

    normal = solve_optimized_position(
        board,
        mark=1,
        max_depth=4,
    )
    diagnostic, _ = solve_optimized_position_with_stats(
        board,
        mark=1,
        max_depth=4,
    )

    assert normal == diagnostic
```

- [ ] **Step 2: Run RED**

```powershell
uv run pytest `
    tests/test_optimized_search.py::test_cache_key_isolates_depth_and_player `
    tests/test_optimized_search.py::test_transposition_rich_search_records_cache_hits `
    tests/test_optimized_search.py::test_transposition_table_is_fresh_per_solver_call `
    tests/test_optimized_search.py::test_production_api_matches_diagnostic_api `
    -q
```

Expected: import/attribute failures because cache/statistics interfaces do not exist yet.

- [ ] **Step 3: Add immutable stats and mutable internal counters**

Add near the top of `optimized_search.py`:

```python
from dataclasses import dataclass


CacheKey = tuple[
    tuple[int, ...],
    int,
    int | None,
]


@dataclass(frozen=True)
class SearchStats:
    nodes_visited: int
    cache_hits: int
    cache_misses: int
    cache_entries: int
    max_recursion_depth: int


@dataclass
class _SearchCounters:
    nodes_visited: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    max_recursion_depth: int = 0


def _cache_key(
    board: list[int],
    mark: int,
    remaining_depth: int | None,
) -> CacheKey:
    return (
        tuple(board),
        mark,
        remaining_depth,
    )
```

- [ ] **Step 4: Replace the uncached recursion with cache-aware recursion**

Implement:

```python
def _search_value(
    board: list[int],
    mark: int,
    remaining_depth: int | None,
    table: dict[CacheKey, GameValue],
    counters: _SearchCounters,
    recursion_depth: int,
) -> GameValue:
    counters.nodes_visited += 1
    counters.max_recursion_depth = max(
        counters.max_recursion_depth,
        recursion_depth,
    )

    key = _cache_key(
        board,
        mark,
        remaining_depth,
    )
    cached = table.get(key)

    if cached is not None:
        counters.cache_hits += 1
        return cached

    counters.cache_misses += 1

    terminal = _terminal_value(
        board,
        mark,
    )
    if terminal is not None:
        table[key] = terminal
        return terminal

    if remaining_depth == 0:
        table[key] = "unknown"
        return "unknown"

    child_depth = (
        None
        if remaining_depth is None
        else remaining_depth - 1
    )

    move_values: list[GameValue] = []

    for column in _legal_columns(board):
        child = _drop_piece(
            board,
            column,
            mark,
        )

        if _is_win(child, mark):
            move_values.append("win")
            continue

        child_value = _search_value(
            child,
            3 - mark,
            child_depth,
            table,
            counters,
            recursion_depth + 1,
        )
        move_values.append(
            _invert_value(child_value)
        )

    result = _aggregate_values(
        tuple(move_values)
    )
    table[key] = result
    return result
```

Implement one shared root function and the two public entry points:

```python
def _solve_with_stats(
    board: list[int],
    mark: int,
    max_depth: int | None,
) -> tuple[PositionSolution, SearchStats]:
    _validate_position(
        board,
        mark,
    )
    _validate_max_depth(
        max_depth,
    )

    terminal = _terminal_value(
        board,
        mark,
    )
    if terminal is not None:
        return (
            PositionSolution(
                value=terminal,
                moves=(),
                complete=True,
            ),
            SearchStats(
                nodes_visited=0,
                cache_hits=0,
                cache_misses=0,
                cache_entries=0,
                max_recursion_depth=0,
            ),
        )

    legal = _legal_columns(board)

    if max_depth == 0:
        moves = tuple(
            MoveAnalysis(column, "unknown")
            for column in legal
        )
        return (
            PositionSolution(
                value="unknown",
                moves=moves,
                complete=False,
            ),
            SearchStats(
                nodes_visited=0,
                cache_hits=0,
                cache_misses=0,
                cache_entries=0,
                max_recursion_depth=0,
            ),
        )

    table: dict[CacheKey, GameValue] = {}
    counters = _SearchCounters()

    child_depth = (
        None
        if max_depth is None
        else max_depth - 1
    )

    moves: list[MoveAnalysis] = []

    for column in legal:
        child = _drop_piece(
            board,
            column,
            mark,
        )

        if _is_win(child, mark):
            value: GameValue = "win"
        else:
            value = _invert_value(
                _search_value(
                    child,
                    3 - mark,
                    child_depth,
                    table,
                    counters,
                    recursion_depth=1,
                )
            )

        moves.append(
            MoveAnalysis(
                column=column,
                value=value,
            )
        )

    move_tuple = tuple(moves)

    solution = PositionSolution(
        value=_aggregate_values(
            tuple(
                move.value
                for move in move_tuple
            )
        ),
        moves=move_tuple,
        complete=all(
            move.value != "unknown"
            for move in move_tuple
        ),
    )

    stats = SearchStats(
        nodes_visited=counters.nodes_visited,
        cache_hits=counters.cache_hits,
        cache_misses=counters.cache_misses,
        cache_entries=len(table),
        max_recursion_depth=counters.max_recursion_depth,
    )

    return solution, stats


def solve_optimized_position(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> PositionSolution:
    solution, _ = _solve_with_stats(
        board,
        mark,
        max_depth,
    )
    return solution


def solve_optimized_position_with_stats(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> tuple[PositionSolution, SearchStats]:
    return _solve_with_stats(
        board,
        mark,
        max_depth,
    )
```

Delete `_search_value_uncached` and `_solve_uncached` after the cache-aware implementation is green; the final module must not retain a second hidden search engine.

- [ ] **Step 5: Run cache GREEN and full oracle parity**

```powershell
uv run pytest `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py `
    -q
```

Expected:
- cache tests pass with `cache_hits > 0`;
- every bounded and exhaustive CX-04 parity test remains green.

Any parity difference blocks performance work.

- [ ] **Step 6: Perform the required cache mutation verification**

Back up the optimized module:

```powershell
$Source = "src/connect_x_agent/optimized_search.py"
$Backup = Join-Path $env:TEMP "cx06a-optimized-search-backup.py"

Copy-Item $Source $Backup -Force
```

Apply a temporary cache-lookup bypass:

```powershell
@'
from pathlib import Path

path = Path("src/connect_x_agent/optimized_search.py")
text = path.read_text(encoding="utf-8")

old = """    if cached is not None:
        counters.cache_hits += 1
        return cached
"""

new = """    if False and cached is not None:
        counters.cache_hits += 1
        return cached
"""

if text.count(old) != 1:
    raise SystemExit(
        "Expected cache lookup block exactly once. Stop."
    )

path.write_text(
    text.replace(old, new, 1),
    encoding="utf-8",
)
'@ | uv run python -
```

First prove semantics remain correct:

```powershell
uv run pytest tests/test_optimized_search_oracle.py -q
```

Expected: PASS.

Then prove the optimization-sensitive test turns RED:

```powershell
uv run pytest `
    tests/test_optimized_search.py::test_transposition_rich_search_records_cache_hits `
    -q
```

Expected: FAIL specifically because `stats.cache_hits == 0`.

Restore immediately:

```powershell
Copy-Item $Backup $Source -Force

uv run pytest `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py `
    -q
```

Expected: PASS.

- [ ] **Step 7: Run Task 4 quality gates**

```powershell
uv run ruff check `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py

uv run mypy `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py

git diff --check
```

Expected: all clean.

- [ ] **Step 8: Commit Task 4**

```powershell
git add -- `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py

git diff --cached --name-only
git diff --cached --check
git commit -m "feat: cache CX-06A transpositions"
```

---

### Task 5: Add the Fixed-Position Cache Benchmark Harness

**Files:**
- Create: `scripts/cx06a_cache_probe.py`
- Modify: `tests/test_optimized_search.py`

**Interfaces:**
- Consumes:
  - `solve_position`;
  - `solve_optimized_position_with_stats`;
  - the four approved CX-05 frontier boards.
- Produces:
  - `ProbePosition`;
  - `FRONTIER_POSITIONS`;
  - command-line depth/trial selection;
  - repeated timing summaries;
  - optional optimized-only depth-7 mode.

- [ ] **Step 1: Write the script-structure RED test before creating the script**

Append to `tests/test_optimized_search.py`:

```python
from pathlib import Path
import runpy


def test_cx06a_cache_probe_declares_fixed_frontier_positions() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cx06a_cache_probe.py"
    )

    namespace = runpy.run_path(
        str(script),
        run_name="cx06a_probe_test",
    )

    positions = namespace["FRONTIER_POSITIONS"]

    assert tuple(
        position.name
        for position in positions
    ) == (
        "empty-opening",
        "game1-danger-ply20",
        "negamax-danger-ply17",
        "hybrid-pre-loss-ply15",
    )
    assert namespace["DEFAULT_TRIALS"] == 3
```

- [ ] **Step 2: Run RED**

```powershell
uv run pytest `
    tests/test_optimized_search.py::test_cx06a_cache_probe_declares_fixed_frontier_positions `
    -q
```

Expected: `FileNotFoundError` for `scripts/cx06a_cache_probe.py`.

- [ ] **Step 3: Implement the benchmark script**

Create `scripts/cx06a_cache_probe.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable

from connect_x_agent.optimized_search import (
    SearchStats,
    solve_optimized_position_with_stats,
)
from connect_x_agent.search import (
    PositionSolution,
    solve_position,
)


DEFAULT_TRIALS = 3


@dataclass(frozen=True)
class ProbePosition:
    name: str
    board: tuple[int, ...]
    mark: int


FRONTIER_POSITIONS = (
    ProbePosition(
        name="empty-opening",
        board=tuple([0] * 42),
        mark=1,
    ),
    ProbePosition(
        name="game1-danger-ply20",
        board=(
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 0, 0, 0, 0,
            1, 1, 2, 0, 0, 0, 0,
            2, 2, 1, 2, 0, 0, 0,
            1, 1, 1, 2, 0, 0, 0,
        ),
        mark=1,
    ),
    ProbePosition(
        name="negamax-danger-ply17",
        board=(
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 1,
            2, 1, 2, 0, 1, 0, 1,
            2, 1, 2, 2, 1, 1, 1,
        ),
        mark=2,
    ),
    ProbePosition(
        name="hybrid-pre-loss-ply15",
        board=(
            2, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0,
            2, 2, 0, 0, 0, 0, 0,
            2, 1, 0, 0, 1, 0, 1,
            2, 2, 1, 0, 1, 0, 1,
        ),
        mark=2,
    ),
)


def unknown_count(
    solution: PositionSolution,
) -> int:
    return sum(
        move.value == "unknown"
        for move in solution.moves
    )


def time_call(
    function: Callable[[], PositionSolution],
    trials: int,
) -> tuple[PositionSolution, tuple[float, ...]]:
    elapsed_ms: list[float] = []
    result: PositionSolution | None = None

    for _ in range(trials):
        started = perf_counter()
        result = function()
        elapsed_ms.append(
            (perf_counter() - started) * 1000.0
        )

    if result is None:
        raise RuntimeError("Benchmark executed no trials")

    return result, tuple(elapsed_ms)


def reference_solution(
    position: ProbePosition,
    depth: int,
) -> PositionSolution:
    return solve_position(
        list(position.board),
        mark=position.mark,
        columns=7,
        rows=6,
        inarow=4,
        max_depth=depth,
    )


def optimized_solution(
    position: ProbePosition,
    depth: int,
) -> tuple[PositionSolution, SearchStats]:
    return solve_optimized_position_with_stats(
        list(position.board),
        mark=position.mark,
        max_depth=depth,
    )


def run_reference(
    position: ProbePosition,
    depth: int,
    trials: int,
) -> None:
    solution, elapsed = time_call(
        lambda: reference_solution(
            position,
            depth,
        ),
        trials,
    )

    print("solver:      CX-04 reference")
    print(f"position:    {position.name}")
    print(f"depth:       {depth}")
    print(f"root:        {solution.value.upper()}")
    print(f"unknowns:    {unknown_count(solution)}")
    print(
        "trials ms:   "
        + ", ".join(
            f"{value:.3f}"
            for value in elapsed
        )
    )
    print(f"median ms:   {median(elapsed):.3f}")


def run_optimized(
    position: ProbePosition,
    depth: int,
    trials: int,
) -> None:
    elapsed_ms: list[float] = []
    result: PositionSolution | None = None
    stats: SearchStats | None = None

    for _ in range(trials):
        started = perf_counter()
        result, stats = optimized_solution(
            position,
            depth,
        )
        elapsed_ms.append(
            (perf_counter() - started) * 1000.0
        )

    if result is None or stats is None:
        raise RuntimeError("Benchmark executed no trials")

    print("solver:      CX-06A cached")
    print(f"position:    {position.name}")
    print(f"depth:       {depth}")
    print(f"root:        {result.value.upper()}")
    print(f"unknowns:    {unknown_count(result)}")
    print(
        "trials ms:   "
        + ", ".join(
            f"{value:.3f}"
            for value in elapsed_ms
        )
    )
    print(
        f"median ms:   "
        f"{median(elapsed_ms):.3f}"
    )
    print(f"nodes:       {stats.nodes_visited}")
    print(f"cache hits:  {stats.cache_hits}")
    print(f"cache miss:  {stats.cache_misses}")
    print(f"entries:     {stats.cache_entries}")
    print(
        f"max recurse: "
        f"{stats.max_recursion_depth}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CX-06A transposition-cache benchmark",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=(6, 7),
        default=6,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
    )
    parser.add_argument(
        "--optimized-only",
        action="store_true",
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")

    return args


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("CX-06A TRANSPOSITION CACHE PROBE")
    print(
        f"depth={args.depth} "
        f"trials={args.trials} "
        f"optimized_only={args.optimized_only}"
    )
    print("=" * 72)

    for position in FRONTIER_POSITIONS:
        print()
        print("-" * 72)

        if not args.optimized_only:
            run_reference(
                position,
                args.depth,
                args.trials,
            )
            print()

        run_optimized(
            position,
            args.depth,
            args.trials,
        )

    print()
    print("=" * 72)
    print("CX-06A CACHE PROBE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run script-structure GREEN**

```powershell
uv run pytest `
    tests/test_optimized_search.py::test_cx06a_cache_probe_declares_fixed_frontier_positions `
    -q
```

Expected: PASS and no benchmark execution because `runpy` uses a non-`__main__` name.

- [ ] **Step 5: Run script quality gates**

```powershell
uv run ruff check `
    scripts/cx06a_cache_probe.py `
    tests/test_optimized_search.py

uv run mypy `
    scripts/cx06a_cache_probe.py `
    tests/test_optimized_search.py

git diff --check
```

Expected: all clean. If Mypy objects to `argparse.Namespace` attributes, introduce a small typed parser-result dataclass rather than suppressing the type checker.

- [ ] **Step 6: Commit Task 5**

```powershell
git add -- `
    scripts/cx06a_cache_probe.py `
    tests/test_optimized_search.py

git diff --cached --name-only
git diff --cached --check
git commit -m "test: add CX-06A cache benchmark"
```

---

### Task 6: Record Depth-6 Performance Evidence, Then Probe Depth 7

**Files:**
- No production-code changes expected.
- Evidence remains console output unless a separately approved evidence artifact is requested.

**Interfaces:**
- Consumes: completed and oracle-verified CX-06A.
- Produces: performance evidence used to decide whether CX-06A caching is materially useful and whether depth 7 is computationally practical.

- [ ] **Step 1: Re-run correctness gates immediately before benchmarking**

```powershell
uv run pytest `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py `
    -q

uv run ruff check `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py `
    scripts/cx06a_cache_probe.py

uv run mypy `
    src/connect_x_agent/optimized_search.py `
    tests/test_optimized_search.py `
    tests/test_optimized_search_oracle.py `
    scripts/cx06a_cache_probe.py

git diff --check
```

Expected: all clean.

- [ ] **Step 2: Run the required depth-6 comparison**

```powershell
uv run python scripts/cx06a_cache_probe.py --depth 6 --trials 3
```

Record for every fixed position:
- CX-04 median milliseconds;
- CX-06A median milliseconds;
- result root value;
- UNKNOWN root count;
- nodes;
- cache hits;
- cache misses;
- entries;
- maximum recursive depth.

Correctness requirement: CX-04 and CX-06A results must match. If they do not, stop; do not interpret performance.

Decision evidence:
- calculate `(reference_median - optimized_median) / reference_median * 100`;
- roughly 25%+ reduction is the design’s “materially useful” decision threshold;
- smaller gains are still retained as evidence.

- [ ] **Step 3: Only after depth-6 parity and cache effectiveness are confirmed, run optimized depth 7**

```powershell
uv run python `
    scripts/cx06a_cache_probe.py `
    --depth 7 `
    --trials 3 `
    --optimized-only
```

This is a feasibility measurement only.

Do not change `SOLVER_DEPTH`, CX-05 policy, or Kaggle submission code during this task.

- [ ] **Step 4: Interpret depth 7 against the competition engineering objective**

Record:
- median and worst observed trial time on the empty opening;
- median and worst observed trial time on the three danger positions;
- cache hit ratio `cache_hits / nodes_visited`;
- whether depth 7 changes any of the previously UNKNOWN depth-6 root classifications.

Possible conclusions:
1. depth 7 is comfortably practical: proceed to a separately approved playing-policy experiment;
2. depth 7 is useful but too close to the move budget: CX-06B alpha-beta is justified;
3. cache gains are small: CX-06B is justified because transposition caching alone is insufficient;
4. depth 7 remains strategically unresolved: future work should address UNKNOWN evaluation, but only under a new approved design.

No conclusion in this task authorizes code changes beyond CX-06A.

---

### Task 7: Full CX-06A Acceptance Gate and Scope Audit

**Files:**
- Verify only the approved CX-06A paths plus design/plan documentation.
- Do not stage unrelated untracked material.

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: evidence-backed acceptance or a concrete blocking failure.

- [ ] **Step 1: Run fresh full repository tests**

```powershell
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run full Ruff**

```powershell
uv run ruff check .
```

Expected: all checks pass.

- [ ] **Step 3: Run full source/script Mypy**

```powershell
uv run mypy src scripts
```

Expected: success with no issues.

- [ ] **Step 4: Run whitespace gate**

```powershell
git diff --check
```

Expected: exit code 0. LF→CRLF informational warnings are not trailing-whitespace failures.

- [ ] **Step 5: Audit exact branch scope**

Run:

```powershell
Write-Host "`n=== BRANCH / HEAD ==="
git branch --show-current
git rev-parse HEAD

Write-Host "`n=== STATUS ==="
git status --short

Write-Host "`n=== CX-06A COMMITS ==="
git log --oneline --decorate -8
```

Expected CX-06A implementation paths:

```text
docs/superpowers/specs/2026-08-28-cx-06a-transposition-search-design.md
docs/superpowers/plans/2026-08-28-cx-06a-transposition-search.md
src/connect_x_agent/optimized_search.py
tests/test_optimized_search.py
tests/test_optimized_search_oracle.py
scripts/cx06a_cache_probe.py
```

`src/connect_x_agent/search.py` must not be modified by CX-06A.

`docs/training/` and `evidence/cx_gt_02/` remain operator-owned/unrelated and must not be staged, deleted, cleaned, or rewritten.

- [ ] **Step 6: Verify CX-04 itself remains green and untouched**

```powershell
uv run pytest tests/test_search.py -q

git diff -- `
    src/connect_x_agent/search.py `
    tests/test_search.py
```

Expected:
- CX-04 tests pass;
- no CX-06A diff exists in `src/connect_x_agent/search.py`.

- [ ] **Step 7: Report acceptance without automatically promoting or changing playing depth**

The acceptance report must state:
- focused test count/result;
- full pytest result;
- Ruff result;
- Mypy result;
- diff-check result;
- oracle parity result;
- mutation verification result;
- depth-6 benchmark summary;
- depth-7 feasibility summary;
- exact branch status and unrelated untracked paths.

Do not push, merge, open a PR, promote the branch, change CX-05 playing depth, or package a Kaggle submission without separate explicit authorization.

---

## Self-Review Against the Approved Spec

### 1. Spec coverage

- Fixed 7×6×4 specialization: Tasks 1–2.
- Independent structural validation: Task 2.
- Public `PositionSolution` semantics: Tasks 1 and 3.
- Bounded UNKNOWN behavior and terminal precedence: Tasks 1 and 3.
- Exact CX-04 oracle parity: Task 3.
- Forced win/draw/loss exact standard-board fixtures: Task 3.
- Mixed exact root move classifications: Task 3.
- Per-call transposition table: Task 4.
- Cache key includes board, mark, remaining depth: Task 4.
- UNKNOWN is cacheable under depth-specific key: Task 4 implementation.
- Immutable `SearchStats`: Task 4.
- Required cache statistics: Task 4.
- Fresh cache lifetime: Task 4 test.
- Depth isolation: Task 4 key test.
- Mutation-sensitive cache verification: Task 4.
- Fixed-position repeated benchmark: Task 5.
- Depth-6 evidence before depth-7 probe: Task 6.
- No timing thresholds in unit tests: Tasks 4–6.
- No alpha-beta, heuristic, ordering, symmetry, bitboards, persistent cache, or playing-policy changes: Global Constraints and all task boundaries.
- Full repository quality gates and scope audit: Task 7.

No approved spec requirement is intentionally deferred.

### 2. Placeholder scan

This plan contains no placeholder implementation steps.

### 3. Type/interface consistency

The plan consistently uses:

```python
solve_optimized_position(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> PositionSolution

solve_optimized_position_with_stats(
    board: list[int],
    mark: int,
    max_depth: int | None = None,
) -> tuple[PositionSolution, SearchStats]
```

`SearchStats` fields are consistently:

```text
nodes_visited
cache_hits
cache_misses
cache_entries
max_recursion_depth
```

The benchmark consumes the same diagnostic interface defined in Task 4.

---

## Execution Boundary

Implementation begins only after:
1. the approved design spec is committed;
2. this plan is saved and committed;
3. the branch baseline gate is green.

The first production-code action is Task 1’s failing test. No optimized-search production code should be created before that RED state is observed.
