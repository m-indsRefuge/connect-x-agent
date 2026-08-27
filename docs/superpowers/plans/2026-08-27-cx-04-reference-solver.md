# CX-04 Reference Position Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a correctness-first recursive Connect X position solver that classifies every legal root move as `win`, `draw`, `loss`, or `unknown`, with exact terminal semantics, deterministic depth bounds, and strict local-position validation.

**Architecture:** Add a single focused solver module, `src/connect_x_agent/search.py`, that reuses `legal_columns`, `drop_piece`, and `is_win` from `tactical.py`. The module owns immutable result contracts, validation, player-relative minimax recursion, root move classification, and bounded `unknown` semantics; it intentionally contains no move-selection policy, alpha-beta pruning, cache, symmetry, heuristic evaluation, timing, or Kaggle-facing integration.

**Tech Stack:** Python 3.12+, pytest 9.1+, Ruff 0.16+, Mypy 2.3+, existing `connect_x_agent.tactical` mechanics.

**Spec:** `docs/superpowers/specs/2026-08-27-cx-04-reference-solver-design.md`

## Global Constraints

- Work only on branch `build/cx-04-reference-solver` in an isolated worktree created at execution time.
- Preserve all unrelated local and untracked files; never reset, clean, stash, overwrite, or delete unrelated work.
- Implementation scope is limited to `src/connect_x_agent/search.py` and `tests/test_search.py` unless a concrete integration blocker is demonstrated before modifying another source file.
- Reuse `legal_columns(board, columns)`, `drop_piece(board, column, mark, columns, rows)`, and `is_win(board, mark, columns, rows, inarow)` from `src/connect_x_agent/tactical.py`.
- Values are always player-relative: `win`, `draw`, `loss`, `unknown`.
- Every legal root move must receive its own `MoveAnalysis`; CX-04 must not expose a `best_move()` function.
- `unknown` means valid but unproven under the current depth bound; invalid input raises `ValueError`.
- `max_depth=None` means exhaustive recursion to terminal states; `max_depth` must otherwise be a non-negative integer.
- At the public root, `max_depth=0` returns every legal root move as `unknown` without simulating any root move, unless the input position is already terminal.
- A proven root `win` may make the overall value `win` even when other root moves remain `unknown`; unresolved moves prevent an exact `draw` or `loss` claim.
- No heuristics, randomness, elapsed-time cutoff, alpha-beta pruning, caching, symmetry, bitboards, opening book, persistent tablebase, or competition action policy in CX-04.
- TDD is mandatory: RED first, then minimal GREEN, focused test, full relevant test file, and commit at every task boundary.
- Final acceptance requires focused tests, full pytest, Ruff, Mypy, `git diff --check`, and changed-file scope review.

---

## Execution Preflight

Before Task 1, use the worktree workflow to create or enter an isolated worktree for `build/cx-04-reference-solver`. Then run:

```powershell
Write-Host "`n=== BRANCH ==="
git branch --show-current

Write-Host "`n=== HEAD ==="
git rev-parse HEAD

Write-Host "`n=== STATUS ==="
git status --short

Write-Host "`n=== BASELINE TESTS ==="
uv run pytest -q

Write-Host "`n=== BASELINE RUFF ==="
uv run ruff check .

Write-Host "`n=== BASELINE MYPY ==="
uv run mypy src scripts

Write-Host "`n=== BASELINE DIFF CHECK ==="
git diff --check
```

Expected baseline before implementation:

- branch is `build/cx-04-reference-solver`;
- the design spec and this plan are already committed;
- implementation paths `src/connect_x_agent/search.py` and `tests/test_search.py` do not yet exist;
- existing repository tests, Ruff, Mypy, and diff check pass.

If the baseline is not green, stop and diagnose before Task 1.

---

### Task 1: Immutable solver contracts and value aggregation algebra

**Files:**
- Create: `tests/test_search.py`
- Create: `src/connect_x_agent/search.py`

**Interfaces:**
- Consumes: no new CX-04 interfaces; standard-library `dataclass` and `Literal` only.
- Produces:
  - `GameValue = Literal["win", "draw", "loss", "unknown"]`
  - `MoveAnalysis(column: int, value: GameValue)`
  - `PositionSolution(value: GameValue, moves: tuple[MoveAnalysis, ...], complete: bool)`
  - internal `_aggregate_values(values: tuple[GameValue, ...]) -> GameValue`

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_search.py` with:

```python
from dataclasses import FrozenInstanceError

import pytest

from connect_x_agent.search import (
    MoveAnalysis,
    PositionSolution,
    _aggregate_values,
)


def test_solver_result_contracts_are_immutable_and_structural() -> None:
    move = MoveAnalysis(column=3, value="win")
    solution = PositionSolution(
        value="win",
        moves=(move,),
        complete=True,
    )

    assert move == MoveAnalysis(column=3, value="win")
    assert solution == PositionSolution(
        value="win",
        moves=(MoveAnalysis(column=3, value="win"),),
        complete=True,
    )

    with pytest.raises(FrozenInstanceError):
        move.column = 4  # type: ignore[misc]


def test_value_aggregation_preserves_partial_proof_semantics() -> None:
    assert _aggregate_values(("loss", "win", "unknown")) == "win"
    assert _aggregate_values(("loss", "draw", "unknown")) == "unknown"
    assert _aggregate_values(("loss", "loss", "draw")) == "draw"
    assert _aggregate_values(("loss", "loss", "loss")) == "loss"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
uv run pytest tests/test_search.py -q
```

Expected: collection/import failure because `connect_x_agent.search` does not exist.

- [ ] **Step 3: Add the minimal contracts and aggregation implementation**

Create `src/connect_x_agent/search.py` with:

```python
from dataclasses import dataclass
from typing import Literal

GameValue = Literal["win", "draw", "loss", "unknown"]


@dataclass(frozen=True)
class MoveAnalysis:
    column: int
    value: GameValue


@dataclass(frozen=True)
class PositionSolution:
    value: GameValue
    moves: tuple[MoveAnalysis, ...]
    complete: bool


def _aggregate_values(values: tuple[GameValue, ...]) -> GameValue:
    if "win" in values:
        return "win"
    if "unknown" in values:
        return "unknown"
    if "draw" in values:
        return "draw"
    return "loss"
```

The helper is deliberately small and exact. It encodes the approved partial-proof order `WIN > UNKNOWN > DRAW > LOSS` rather than a numeric heuristic.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
uv run pytest tests/test_search.py -q
uv run ruff check src/connect_x_agent/search.py tests/test_search.py
uv run mypy src/connect_x_agent/search.py tests/test_search.py
```

Expected: 2 tests pass; Ruff and Mypy pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/connect_x_agent/search.py tests/test_search.py
git commit -m "feat: add CX-04 solver result contracts"
```

---

### Task 2: Structural position validation and zero-depth root semantics

**Files:**
- Modify: `src/connect_x_agent/search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes:
  - `MoveAnalysis`, `PositionSolution`, `_aggregate_values`
  - `legal_columns(board: list[int], columns: int) -> list[int]` from `connect_x_agent.tactical`
- Produces:
  - public `solve_position(board: list[int], mark: int, columns: int, rows: int, inarow: int, max_depth: int | None = None) -> PositionSolution`
  - internal structural validation helpers

- [ ] **Step 1: Add failing tests for valid zero-depth behavior and malformed structure**

Append to `tests/test_search.py`:

```python
from connect_x_agent.search import solve_position


def test_zero_depth_classifies_every_legal_root_move_as_unknown() -> None:
    solution = solve_position(
        board=[0] * 9,
        mark=1,
        columns=3,
        rows=3,
        inarow=3,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="unknown",
        moves=(
            MoveAnalysis(0, "unknown"),
            MoveAnalysis(1, "unknown"),
            MoveAnalysis(2, "unknown"),
        ),
        complete=False,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"columns": 0}, "columns"),
        ({"rows": 0}, "rows"),
        ({"inarow": 0}, "inarow"),
        ({"mark": 3}, "mark"),
        ({"max_depth": -1}, "max_depth"),
    ],
)
def test_solver_rejects_invalid_scalar_inputs(
    kwargs: dict[str, int],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "board": [0] * 9,
        "mark": 1,
        "columns": 3,
        "rows": 3,
        "inarow": 3,
        "max_depth": 0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        solve_position(**arguments)  # type: ignore[arg-type]


def test_solver_rejects_wrong_board_length() -> None:
    with pytest.raises(ValueError, match="board length"):
        solve_position([0] * 8, 1, 3, 3, 3, max_depth=0)


def test_solver_rejects_invalid_cell_value() -> None:
    board = [0] * 9
    board[-1] = 7

    with pytest.raises(ValueError, match="cell"):
        solve_position(board, 1, 3, 3, 3, max_depth=0)


def test_solver_rejects_floating_checker() -> None:
    board = [0] * 9
    board[0] = 1

    with pytest.raises(ValueError, match="gravity"):
        solve_position(board, 2, 3, 3, 3, max_depth=0)


def test_solver_rejects_turn_count_mismatch() -> None:
    board = [0] * 9
    board[6:9] = [1, 1, 0]

    with pytest.raises(ValueError, match="turn"):
        solve_position(board, 1, 3, 3, 3, max_depth=0)
```

- [ ] **Step 2: Run the new focused tests and verify RED**

Run:

```powershell
uv run pytest tests/test_search.py -q
```

Expected: failures because `solve_position` and validation do not exist.

- [ ] **Step 3: Implement structural validation and the zero-depth shell**

Extend `search.py` with imports and helpers equivalent to:

```python
from connect_x_agent.tactical import legal_columns


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _counts_match_turn(board: list[int], mark: int) -> bool:
    p1 = board.count(1)
    p2 = board.count(2)
    if mark == 1:
        return p1 == p2
    return p1 == p2 + 1


def _gravity_is_valid(board: list[int], columns: int, rows: int) -> bool:
    for column in range(columns):
        empty_seen_below = False
        for row in range(rows - 1, -1, -1):
            value = board[row * columns + column]
            if value == 0:
                empty_seen_below = True
            elif empty_seen_below:
                return False
    return True


def _validate_structure(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    max_depth: int | None,
) -> None:
    _require_positive_int("columns", columns)
    _require_positive_int("rows", rows)
    _require_positive_int("inarow", inarow)

    if type(mark) is not int or mark not in (1, 2):
        raise ValueError("mark must be 1 or 2")
    if max_depth is not None and (
        type(max_depth) is not int or max_depth < 0
    ):
        raise ValueError("max_depth must be None or a non-negative integer")
    if len(board) != columns * rows:
        raise ValueError("board length must equal columns * rows")
    if any(type(cell) is not int or cell not in (0, 1, 2) for cell in board):
        raise ValueError("board cell values must be 0, 1, or 2")
    if not _gravity_is_valid(board, columns, rows):
        raise ValueError("board violates gravity")
    if not _counts_match_turn(board, mark):
        raise ValueError("piece counts are inconsistent with side to move")


def solve_position(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    max_depth: int | None = None,
) -> PositionSolution:
    _validate_structure(board, mark, columns, rows, inarow, max_depth)
    legal = legal_columns(board, columns)

    if max_depth == 0:
        moves = tuple(MoveAnalysis(column, "unknown") for column in legal)
        return PositionSolution(
            value="unknown",
            moves=moves,
            complete=False,
        )

    raise NotImplementedError("recursive search is added in later CX-04 tasks")
```

The temporary `NotImplementedError` is permitted only inside this intermediate TDD slice and must be removed in Task 4. No final CX-04 behavior may retain it.

- [ ] **Step 4: Verify Task 2 GREEN**

Run:

```powershell
uv run pytest tests/test_search.py -q
uv run ruff check src/connect_x_agent/search.py tests/test_search.py
uv run mypy src/connect_x_agent/search.py tests/test_search.py
```

Expected: all Task 1-2 tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/connect_x_agent/search.py tests/test_search.py
git commit -m "feat: validate CX-04 solver inputs"
```

---

### Task 3: Terminal positions and local last-move consistency

**Files:**
- Modify: `src/connect_x_agent/search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes:
  - `_validate_structure`, `_counts_match_turn`, `_gravity_is_valid`
  - `is_win(...)` and `legal_columns(...)` from `connect_x_agent.tactical`
- Produces:
  - exact terminal `loss` and `draw` `PositionSolution` values
  - winner-state validation including simultaneous-winner, winner/turn, and local last-move consistency checks

- [ ] **Step 1: Add failing terminal and winner-integrity tests**

Append:

```python

def test_terminal_previous_player_win_is_exact_loss() -> None:
    board = [
        0, 0, 2, 0,
        1, 1, 1, 2,
    ]

    solution = solve_position(board, 2, 4, 2, 3, max_depth=0)

    assert solution == PositionSolution(
        value="loss",
        moves=(),
        complete=True,
    )


def test_terminal_full_board_without_winner_is_exact_draw() -> None:
    solution = solve_position(
        board=[1, 2],
        mark=1,
        columns=2,
        rows=1,
        inarow=2,
        max_depth=0,
    )

    assert solution == PositionSolution(
        value="draw",
        moves=(),
        complete=True,
    )


def test_solver_rejects_simultaneous_winners() -> None:
    board = [
        1, 2, 0, 0,
        1, 2, 0, 0,
    ]

    with pytest.raises(ValueError, match="both players"):
        solve_position(board, 1, 4, 2, 2, max_depth=0)


def test_solver_rejects_winner_that_is_not_previous_player() -> None:
    board = [
        0, 2, 2, 0,
        1, 1, 1, 2,
    ]

    with pytest.raises(ValueError, match="previous player"):
        solve_position(board, 1, 4, 2, 3, max_depth=0)


def test_solver_rejects_board_requiring_play_after_an_earlier_win() -> None:
    board = [
        1, 2, 2,
        1, 2, 2,
        1, 1, 1,
    ]

    with pytest.raises(ValueError, match="last move"):
        solve_position(board, 2, 3, 3, 3, max_depth=0)
```

The final fixture has P1 winning both vertically in column 0 and horizontally on the bottom row. The only topmost P1 checker that could be the last move is the top of column 0; removing it leaves the bottom-row win intact. Therefore the board cannot represent a legal game that stopped immediately when the first win occurred.

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: terminal positions still receive the Task 2 zero-depth shell result or winner-integrity tests do not reject invalid states.

- [ ] **Step 3: Implement winner validation and terminal handling**

Import `is_win` and add helpers equivalent to:

```python
from connect_x_agent.tactical import is_win, legal_columns


def _has_consistent_last_move(
    board: list[int],
    winner: int,
    columns: int,
    rows: int,
    inarow: int,
) -> bool:
    for column in range(columns):
        topmost: int | None = None
        for row in range(rows):
            index = row * columns + column
            if board[index] != 0:
                topmost = index
                break

        if topmost is None or board[topmost] != winner:
            continue

        predecessor = board.copy()
        predecessor[topmost] = 0

        if not _gravity_is_valid(predecessor, columns, rows):
            continue
        if not _counts_match_turn(predecessor, winner):
            continue
        if is_win(predecessor, 1, columns, rows, inarow):
            continue
        if is_win(predecessor, 2, columns, rows, inarow):
            continue
        return True

    return False


def _winner(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> int | None:
    p1_wins = is_win(board, 1, columns, rows, inarow)
    p2_wins = is_win(board, 2, columns, rows, inarow)

    if p1_wins and p2_wins:
        raise ValueError("both players cannot have winning lines")
    if not p1_wins and not p2_wins:
        return None

    winner = 1 if p1_wins else 2
    previous_player = 3 - mark
    if winner != previous_player:
        raise ValueError("winner must be the previous player")
    if not _has_consistent_last_move(board, winner, columns, rows, inarow):
        raise ValueError("winning board has no consistent last move")
    return winner
```

Then change `solve_position` ordering so terminal truth is evaluated before the `max_depth == 0` branch:

```python
    winner = _winner(board, mark, columns, rows, inarow)
    if winner is not None:
        return PositionSolution("loss", (), True)

    legal = legal_columns(board, columns)
    if not legal:
        return PositionSolution("draw", (), True)

    if max_depth == 0:
        ...
```

Do not recurse below terminal input boards.

- [ ] **Step 4: Verify Task 3 GREEN**

```powershell
uv run pytest tests/test_search.py -q
uv run ruff check src/connect_x_agent/search.py tests/test_search.py
uv run mypy src/connect_x_agent/search.py tests/test_search.py
```

Expected: all Task 1-3 tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/connect_x_agent/search.py tests/test_search.py
git commit -m "feat: enforce CX-04 terminal position integrity"
```

---

### Task 4: Exact recursive minimax and complete root move classification

**Files:**
- Modify: `src/connect_x_agent/search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes:
  - result contracts and aggregation algebra
  - validated nonterminal root positions
  - `drop_piece`, `is_win`, `legal_columns`
- Produces:
  - internal `_invert(value: GameValue) -> GameValue`
  - internal recursive `_solve_value(...) -> GameValue`
  - complete exhaustive `solve_position(...)` behavior when `max_depth=None`

- [ ] **Step 1: Add failing exhaustive-solver fixtures**

Append:

```python

def test_exhaustive_small_board_is_a_complete_draw() -> None:
    solution = solve_position(
        board=[0, 0],
        mark=1,
        columns=2,
        rows=1,
        inarow=2,
    )

    assert solution == PositionSolution(
        value="draw",
        moves=(
            MoveAnalysis(0, "draw"),
            MoveAnalysis(1, "draw"),
        ),
        complete=True,
    )


def test_exhaustive_solver_classifies_mixed_root_values_independently() -> None:
    board = [
        0, 0, 0,
        0, 2, 2,
        0, 1, 1,
    ]

    solution = solve_position(board, 1, 3, 3, 3)

    assert solution == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(0, "win"),
            MoveAnalysis(1, "loss"),
            MoveAnalysis(2, "draw"),
        ),
        complete=True,
    )


def test_exhaustive_solver_preserves_multiple_winning_moves() -> None:
    board = [
        0, 0, 0,
        0, 0, 2,
        0, 0, 1,
    ]

    solution = solve_position(board, 1, 3, 3, 3)

    assert solution == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(0, "win"),
            MoveAnalysis(1, "win"),
            MoveAnalysis(2, "draw"),
        ),
        complete=True,
    )


def test_exhaustive_solver_proves_forced_loss() -> None:
    board = [
        0, 0, 1,
        0, 1, 2,
        0, 2, 1,
    ]

    solution = solve_position(board, 2, 3, 3, 3)

    assert solution == PositionSolution(
        value="loss",
        moves=(
            MoveAnalysis(0, "loss"),
            MoveAnalysis(1, "loss"),
        ),
        complete=True,
    )
```

These fixtures are small enough for transparent exhaustive search and collectively prove draw, mixed values, multiple optimal wins, and forced loss.

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: the Task 2 temporary `NotImplementedError` is reached for positive/unbounded search.

- [ ] **Step 3: Implement the recursive player-relative solver**

Import `drop_piece` and implement:

```python
from connect_x_agent.tactical import drop_piece, is_win, legal_columns


def _invert(value: GameValue) -> GameValue:
    if value == "win":
        return "loss"
    if value == "loss":
        return "win"
    return value


def _solve_value(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    remaining_depth: int | None,
) -> GameValue:
    legal = legal_columns(board, columns)
    if not legal:
        return "draw"
    if remaining_depth == 0:
        return "unknown"

    child_depth = (
        None
        if remaining_depth is None
        else remaining_depth - 1
    )
    move_values: list[GameValue] = []

    for column in legal:
        candidate = drop_piece(board, column, mark, columns, rows)
        if is_win(candidate, mark, columns, rows, inarow):
            move_value: GameValue = "win"
        else:
            child_value = _solve_value(
                candidate,
                3 - mark,
                columns,
                rows,
                inarow,
                child_depth,
            )
            move_value = _invert(child_value)

        move_values.append(move_value)

    return _aggregate_values(tuple(move_values))
```

Replace the temporary `NotImplementedError` in `solve_position` with complete root classification:

```python
    child_depth = None if max_depth is None else max_depth - 1
    analyses: list[MoveAnalysis] = []

    for column in legal:
        candidate = drop_piece(board, column, mark, columns, rows)
        if is_win(candidate, mark, columns, rows, inarow):
            value: GameValue = "win"
        else:
            child_value = _solve_value(
                candidate,
                3 - mark,
                columns,
                rows,
                inarow,
                child_depth,
            )
            value = _invert(child_value)

        analyses.append(MoveAnalysis(column, value))

    moves = tuple(analyses)
    values = tuple(move.value for move in moves)
    return PositionSolution(
        value=_aggregate_values(values),
        moves=moves,
        complete="unknown" not in values,
    )
```

Important: even when one root move is proven `win`, continue classifying every remaining root move. Internal nodes may aggregate their child values normally because only the root contract requires preservation of every legal move classification.

- [ ] **Step 4: Verify exhaustive solver GREEN**

```powershell
uv run pytest tests/test_search.py -q
uv run ruff check src/connect_x_agent/search.py tests/test_search.py
uv run mypy src/connect_x_agent/search.py tests/test_search.py
```

Expected: all Task 1-4 tests pass; the temporary `NotImplementedError` no longer exists.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/connect_x_agent/search.py tests/test_search.py
git commit -m "feat: add CX-04 recursive reference minimax"
```

---

### Task 5: Bounded partial proofs, deterministic results, and 7x6 reference behavior

**Files:**
- Modify: `tests/test_search.py`
- Modify only if a failing test demonstrates a semantic defect: `src/connect_x_agent/search.py`

**Interfaces:**
- Consumes: complete `solve_position` implementation from Task 4.
- Produces: acceptance evidence for bounded `unknown`, proven-win-plus-unknown semantics, deterministic output, and standard-board geometry support.

- [ ] **Step 1: Add failing-or-confirming tests for bounded partial proofs**

Append:

```python

def test_bounded_search_preserves_proven_win_with_unknown_alternatives() -> None:
    board = [0] * 42
    board[35:42] = [1, 1, 1, 0, 2, 2, 2]

    solution = solve_position(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
        max_depth=1,
    )

    assert solution == PositionSolution(
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


def test_draw_plus_unknown_cannot_be_promoted_to_draw() -> None:
    assert _aggregate_values(("loss", "draw", "unknown")) == "unknown"


def test_solver_is_deterministic_for_identical_inputs() -> None:
    board = [
        0, 0, 0,
        0, 2, 2,
        0, 1, 1,
    ]

    first = solve_position(board, 1, 3, 3, 3)
    second = solve_position(board, 1, 3, 3, 3)

    assert first == second
    assert tuple(move.column for move in first.moves) == (0, 1, 2)
```

The 7×6 fixture is intentionally shallow. It proves that CX-04 supports competition geometry and can retain one mathematically proven immediate win while leaving all unsearched alternatives explicitly unknown.

- [ ] **Step 2: Run the focused file**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: PASS if Task 4 implemented the approved depth semantics exactly. If any test fails, diagnose the root cause and make the smallest correction consistent with the spec; do not add heuristics or special-case the fixture.

- [ ] **Step 3: Run mutation-style semantic checks manually through tests**

Confirm the test suite would catch the two most dangerous semantic regressions by temporarily changing one line at a time locally, without committing the mutation:

1. Change `_aggregate_values` so `unknown` is checked before `win`; verify `test_bounded_search_preserves_proven_win_with_unknown_alternatives` fails.
2. Restore it, then change `_invert("loss")` to return `"loss"`; verify at least one exhaustive fixture fails.
3. Restore the correct implementation and rerun the focused test file.

Commands after restoring:

```powershell
uv run pytest tests/test_search.py -q
```

Expected: all focused tests pass. No mutation is committed.

- [ ] **Step 4: Run the CX-04 quality gate**

```powershell
Write-Host "`n=== FOCUSED CX-04 ==="
uv run pytest tests/test_search.py -q

Write-Host "`n=== RUFF ==="
uv run ruff check src/connect_x_agent/search.py tests/test_search.py

Write-Host "`n=== MYPY ==="
uv run mypy src/connect_x_agent/search.py tests/test_search.py

Write-Host "`n=== DIFF CHECK ==="
git diff --check
```

Expected: all focused tests pass; Ruff, Mypy, and diff check are clean.

- [ ] **Step 5: Commit Task 5 if tests required any legitimate implementation/test changes**

If Task 5 only added tests:

```powershell
git add tests/test_search.py
git commit -m "test: harden CX-04 partial proof semantics"
```

If a minimal spec-consistent implementation correction was also required:

```powershell
git add src/connect_x_agent/search.py tests/test_search.py
git commit -m "fix: preserve CX-04 bounded proof semantics"
```

---

### Task 6: Full repository acceptance and scope audit

**Files:**
- No new implementation files.
- Review:
  - `src/connect_x_agent/search.py`
  - `tests/test_search.py`
  - `docs/superpowers/specs/2026-08-27-cx-04-reference-solver-design.md`
  - `docs/superpowers/plans/2026-08-27-cx-04-reference-solver.md`

**Interfaces:**
- Consumes: completed CX-04 branch.
- Produces: repository-level acceptance evidence only; no promotion to `main` without separate explicit authorization.

- [ ] **Step 1: Run the full repository gate**

```powershell
Write-Host "`n=== FOCUSED CX-04 ==="
uv run pytest tests/test_search.py -q

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

Expected: focused and full suites pass; Ruff and Mypy are clean; diff check prints no errors. Any unrelated untracked evidence must remain untouched and be reported separately rather than deleted.

- [ ] **Step 2: Audit changed-file scope against `main`**

```powershell
Write-Host "`n=== CX-04 CHANGED FILES ==="
git diff --name-only main...HEAD

Write-Host "`n=== CX-04 DIFF STAT ==="
git diff --stat main...HEAD
```

Expected implementation scope beyond the already-approved docs:

```text
src/connect_x_agent/search.py
tests/test_search.py
```

If another source path appears, stop and justify it against a concrete implementation need before accepting CX-04.

- [ ] **Step 3: Review the final solver against the non-goals**

Confirm by inspection that `search.py` contains none of the following:

```text
best_move or action-selection policy
random choice
heuristic position score
alpha-beta parameters or pruning
transposition cache
symmetry canonicalization
bitboard representation
wall-clock timing
iterative deepening
Kaggle environment calls
persistent solution database
```

Also confirm every root legal column is returned in deterministic ascending-column order.

- [ ] **Step 4: Record acceptance evidence**

Report the exact focused-test count, full-suite count, Ruff result, Mypy result, diff-check result, status output, changed paths, and current branch HEAD SHA.

Do not claim CX-04 complete if any gate is missing or stale.

- [ ] **Step 5: Stop at the promotion boundary**

Do not merge, fast-forward `main`, push `main`, delete the feature branch, or create a PR unless the user gives separate explicit authorization after seeing the acceptance evidence.
