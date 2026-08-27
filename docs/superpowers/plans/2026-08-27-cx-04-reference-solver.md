# CX-04 Reference Position Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a correctness-first recursive Connect X position solver that classifies every legal root move as `win`, `draw`, `loss`, or `unknown`, with exact terminal semantics, deterministic depth bounds, and strict local-position validation.

**Architecture:** Add one focused solver module, `src/connect_x_agent/search.py`, which reuses `legal_columns`, `drop_piece`, and `is_win` from `tactical.py`. The module owns immutable result contracts, validation, player-relative minimax recursion, root move classification, and bounded `unknown` semantics; it intentionally contains no move-selection policy, alpha-beta pruning, cache, symmetry, heuristic evaluation, timing, or Kaggle-facing integration.

**Tech Stack:** Python 3.12+, pytest 9.1+, Ruff 0.16+, Mypy 2.3+, existing `connect_x_agent.tactical` mechanics.

**Spec:** `docs/superpowers/specs/2026-08-27-cx-04-reference-solver-design.md`

## Global Constraints

- Work only on branch `build/cx-04-reference-solver` in an isolated worktree created at execution time.
- Preserve unrelated local and untracked files; never reset, clean, stash, overwrite, or delete unrelated work.
- Implementation scope is `src/connect_x_agent/search.py` and `tests/test_search.py`. If another source path appears necessary, stop and demonstrate the integration blocker before modifying it.
- Reuse `legal_columns(board, columns)`, `drop_piece(board, column, mark, columns, rows)`, and `is_win(board, mark, columns, rows, inarow)` from `src/connect_x_agent/tactical.py`.
- Values are player-relative: `win`, `draw`, `loss`, `unknown`.
- Every legal root move receives its own `MoveAnalysis`; CX-04 exposes no `best_move()` function.
- `unknown` means valid but unproven under the current depth bound; invalid input raises `ValueError`.
- `max_depth=None` means exhaustive recursion to terminal states. Otherwise `max_depth` is a non-negative integer.
- At the public root, `max_depth=0` returns every legal root move as `unknown` without simulating a root move, unless the input position is already terminal.
- A proven root `win` makes the overall value `win` even when other root moves remain `unknown`; unresolved moves prevent an exact `draw` or `loss` claim.
- No heuristics, randomness, elapsed-time cutoff, alpha-beta pruning, caching, symmetry, bitboards, opening books, persistent tablebases, or competition action policy in CX-04.
- TDD is mandatory: RED first, minimal GREEN, focused verification, then commit at each task boundary.
- Final acceptance requires focused tests, full pytest, Ruff, Mypy, `git diff --check`, and changed-file scope review.

---

## Execution Preflight

Before Task 1, create or enter an isolated worktree for `build/cx-04-reference-solver` using the repository worktree workflow. Then run:

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

Expected baseline:

- branch `build/cx-04-reference-solver`;
- approved design spec and this plan already committed;
- no `src/connect_x_agent/search.py` or `tests/test_search.py` yet;
- existing repository tests, Ruff, Mypy, and diff check green.

If any baseline gate fails, stop and diagnose before implementation.

---

### Task 1: Immutable solver contracts and value aggregation algebra

**Files:**
- Create: `tests/test_search.py`
- Create: `src/connect_x_agent/search.py`

**Interfaces:**
- Produces `GameValue = Literal["win", "draw", "loss", "unknown"]`.
- Produces `MoveAnalysis(column: int, value: GameValue)`.
- Produces `PositionSolution(value: GameValue, moves: tuple[MoveAnalysis, ...], complete: bool)`.
- Produces internal `_aggregate_values(values: tuple[GameValue, ...]) -> GameValue`.

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_search.py`:

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

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: import/collection failure because `connect_x_agent.search` does not exist.

- [ ] **Step 3: Add the minimal contracts and aggregation implementation**

Create `src/connect_x_agent/search.py`:

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

This encodes the approved proof order `WIN > UNKNOWN > DRAW > LOSS`; it is not a numeric heuristic.

- [ ] **Step 4: Verify GREEN**

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

### Task 2: Structural validation and zero-depth root behavior

**Files:**
- Modify: `src/connect_x_agent/search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes `legal_columns(board: list[int], columns: int) -> list[int]` from `connect_x_agent.tactical`.
- Produces public `solve_position(board: list[int], mark: int, columns: int, rows: int, inarow: int, max_depth: int | None = None) -> PositionSolution`.
- Produces internal structural validation helpers.

- [ ] **Step 1: Add failing zero-depth and malformed-input tests**

Extend the existing import in `tests/test_search.py` to include `solve_position`, then append:

```python

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

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: failures because `solve_position` and validation do not exist.

- [ ] **Step 3: Implement structural validation and a conservative nonterminal shell**

Add:

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


def _unknown_root_solution(board: list[int], columns: int) -> PositionSolution:
    moves = tuple(
        MoveAnalysis(column, "unknown")
        for column in legal_columns(board, columns)
    )
    return PositionSolution(
        value="unknown",
        moves=moves,
        complete=False,
    )


def solve_position(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    max_depth: int | None = None,
) -> PositionSolution:
    _validate_structure(board, mark, columns, rows, inarow, max_depth)
    return _unknown_root_solution(board, columns)
```

At this slice the solver is deliberately conservative for every valid nonterminal position. It asserts only `unknown`, which is mathematically safe. Subsequent RED tests require exact terminal and recursive behavior and replace this conservative result incrementally.

- [ ] **Step 4: Verify GREEN**

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
- Consumes `_validate_structure`, `_counts_match_turn`, `_gravity_is_valid`.
- Consumes `is_win(...)` and `legal_columns(...)` from `connect_x_agent.tactical`.
- Produces exact terminal `loss` and `draw` values and winner-state validation.

- [ ] **Step 1: Add failing terminal and winner-integrity tests**

Append:

```python

def test_terminal_previous_player_win_is_exact_loss() -> None:
    board = [
        0, 0, 2, 0,
        1, 1, 1, 2,
    ]

    solution = solve_position(board, 2, 4, 2, 3, max_depth=0)

    assert solution == PositionSolution("loss", (), True)


def test_terminal_full_board_without_winner_is_exact_draw() -> None:
    solution = solve_position(
        board=[1, 2],
        mark=1,
        columns=2,
        rows=1,
        inarow=2,
        max_depth=0,
    )

    assert solution == PositionSolution("draw", (), True)


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

The final fixture has P1 winning vertically in column 0 and horizontally on the bottom row. The only topmost P1 checker that can be the last move is the top of column 0; removing it leaves the bottom-row win, proving legal play would already have terminated.

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: terminal boards still return the conservative `unknown` result, and invalid winner states are not fully rejected.

- [ ] **Step 3: Implement winner validation and terminal handling**

Change the tactical import to include `is_win`, then add:

```python

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

Then make `solve_position` evaluate terminal truth before returning the conservative nonterminal result:

```python
    _validate_structure(board, mark, columns, rows, inarow, max_depth)

    if _winner(board, mark, columns, rows, inarow) is not None:
        return PositionSolution("loss", (), True)

    legal = legal_columns(board, columns)
    if not legal:
        return PositionSolution("draw", (), True)

    return _unknown_root_solution(board, columns)
```

- [ ] **Step 4: Verify GREEN**

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

### Task 4: Exhaustive recursive minimax and complete root classification

**Files:**
- Modify: `src/connect_x_agent/search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes `drop_piece`, `is_win`, and `legal_columns` from `connect_x_agent.tactical`.
- Produces internal `_invert(value: GameValue) -> GameValue`.
- Produces internal exhaustive `_solve_exact_value(...) -> GameValue`.
- Produces exact `solve_position(..., max_depth=None)` behavior.

- [ ] **Step 1: Add failing exhaustive fixtures**

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


def test_exhaustive_solver_solves_late_standard_7x6_position_exactly() -> None:
    board = [
        2, 2, 1, 2, 0, 0, 1,
        2, 1, 2, 1, 1, 1, 2,
        1, 1, 1, 2, 2, 1, 2,
        2, 2, 2, 1, 1, 1, 2,
        1, 2, 1, 2, 2, 2, 1,
        1, 1, 2, 1, 2, 2, 1,
    ]

    solution = solve_position(board, 1, 7, 6, 4)

    assert solution == PositionSolution(
        value="win",
        moves=(
            MoveAnalysis(4, "draw"),
            MoveAnalysis(5, "win"),
        ),
        complete=True,
    )
```

The 7×6 fixture is a legal 40-ply nonterminal position with only columns 4 and 5 available. Exhaustive CX-04 search must prove the two remaining continuations exactly: column 4 preserves a draw, while column 5 forces a win for P1.

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: exhaustive positions still return root moves as `unknown`.

- [ ] **Step 3: Implement exact player-relative recursion**

Change the tactical import to include `drop_piece`, then add:

```python

def _invert(value: GameValue) -> GameValue:
    if value == "win":
        return "loss"
    if value == "loss":
        return "win"
    return value


def _solve_exact_value(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> GameValue:
    legal = legal_columns(board, columns)
    if not legal:
        return "draw"

    values: list[GameValue] = []
    for column in legal:
        candidate = drop_piece(board, column, mark, columns, rows)
        if is_win(candidate, mark, columns, rows, inarow):
            value: GameValue = "win"
        else:
            value = _invert(
                _solve_exact_value(
                    candidate,
                    3 - mark,
                    columns,
                    rows,
                    inarow,
                )
            )
        values.append(value)

    return _aggregate_values(tuple(values))


def _exact_root_solution(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> PositionSolution:
    analyses: list[MoveAnalysis] = []

    for column in legal_columns(board, columns):
        candidate = drop_piece(board, column, mark, columns, rows)
        if is_win(candidate, mark, columns, rows, inarow):
            value: GameValue = "win"
        else:
            value = _invert(
                _solve_exact_value(
                    candidate,
                    3 - mark,
                    columns,
                    rows,
                    inarow,
                )
            )
        analyses.append(MoveAnalysis(column, value))

    moves = tuple(analyses)
    values = tuple(move.value for move in moves)
    return PositionSolution(
        value=_aggregate_values(values),
        moves=moves,
        complete=True,
    )
```

Update the final branch of `solve_position`:

```python
    if max_depth is None:
        return _exact_root_solution(board, mark, columns, rows, inarow)

    return _unknown_root_solution(board, columns)
```

This slice solves exhaustive mode exactly. Positive finite depth remains conservatively `unknown` until Task 5 so bounded semantics receive an independent RED/GREEN cycle.

- [ ] **Step 4: Verify GREEN**

```powershell
uv run pytest tests/test_search.py -q
uv run ruff check src/connect_x_agent/search.py tests/test_search.py
uv run mypy src/connect_x_agent/search.py tests/test_search.py
```

Expected: all Task 1-4 tests pass, including the exact late-game 7×6 fixture.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/connect_x_agent/search.py tests/test_search.py
git commit -m "feat: add CX-04 exhaustive reference minimax"
```

---

### Task 5: Deterministic bounded search and partial proofs

**Files:**
- Modify: `src/connect_x_agent/search.py`
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes `_invert`, `_aggregate_values`, root validation, and tactical mechanics.
- Produces depth-aware internal `_solve_bounded_value(...) -> GameValue`.
- Completes public finite-depth semantics, including `WIN + complete=False` when a winning root move is proven while alternatives remain unresolved.

- [ ] **Step 1: Add failing bounded-search tests**

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


def test_solver_is_deterministic_for_identical_inputs() -> None:
    board = [
        0, 0, 0,
        0, 2, 2,
        0, 1, 1,
    ]

    first = solve_position(board, 1, 3, 3, 3, max_depth=4)
    second = solve_position(board, 1, 3, 3, 3, max_depth=4)

    assert first == second
    assert tuple(move.column for move in first.moves) == (0, 1, 2)
```

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_search.py -q
```

Expected: the finite-depth conservative shell marks column 3 `unknown`, so the proven-win test fails.

- [ ] **Step 3: Implement bounded recursion**

Add:

```python

def _solve_bounded_value(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    remaining_depth: int,
) -> GameValue:
    legal = legal_columns(board, columns)
    if not legal:
        return "draw"
    if remaining_depth == 0:
        return "unknown"

    values: list[GameValue] = []
    for column in legal:
        candidate = drop_piece(board, column, mark, columns, rows)
        if is_win(candidate, mark, columns, rows, inarow):
            value: GameValue = "win"
        else:
            value = _invert(
                _solve_bounded_value(
                    candidate,
                    3 - mark,
                    columns,
                    rows,
                    inarow,
                    remaining_depth - 1,
                )
            )
        values.append(value)

    return _aggregate_values(tuple(values))


def _bounded_root_solution(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
    max_depth: int,
) -> PositionSolution:
    if max_depth == 0:
        return _unknown_root_solution(board, columns)

    analyses: list[MoveAnalysis] = []
    for column in legal_columns(board, columns):
        candidate = drop_piece(board, column, mark, columns, rows)
        if is_win(candidate, mark, columns, rows, inarow):
            value: GameValue = "win"
        else:
            value = _invert(
                _solve_bounded_value(
                    candidate,
                    3 - mark,
                    columns,
                    rows,
                    inarow,
                    max_depth - 1,
                )
            )
        analyses.append(MoveAnalysis(column, value))

    moves = tuple(analyses)
    values = tuple(move.value for move in moves)
    return PositionSolution(
        value=_aggregate_values(values),
        moves=moves,
        complete="unknown" not in values,
    )
```

Replace the finite-depth conservative branch in `solve_position` with:

```python
    if max_depth is None:
        return _exact_root_solution(board, mark, columns, rows, inarow)

    return _bounded_root_solution(
        board,
        mark,
        columns,
        rows,
        inarow,
        max_depth,
    )
```

- [ ] **Step 4: Verify GREEN and semantic sensitivity**

```powershell
uv run pytest tests/test_search.py -q
uv run ruff check src/connect_x_agent/search.py tests/test_search.py
uv run mypy src/connect_x_agent/search.py tests/test_search.py
```

Then perform two local mutation checks without committing either mutation:

1. Change `_aggregate_values` so `unknown` is checked before `win`; rerun the focused suite and confirm the proven-win partial-proof test fails. Restore the correct order.
2. Change `_invert("loss")` to return `"loss"`; rerun the focused suite and confirm an exhaustive fixture fails. Restore the correct inversion.

Finally rerun:

```powershell
uv run pytest tests/test_search.py -q
```

Expected: focused suite passes after both correct restorations.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/connect_x_agent/search.py tests/test_search.py
git commit -m "feat: add CX-04 bounded partial proofs"
```

---

### Task 6: Acceptance hardening and repository gate

**Files:**
- Modify: `tests/test_search.py`
- Review: `src/connect_x_agent/search.py`, `tests/test_search.py`, design spec, and this plan.

**Interfaces:**
- Produces repository-level acceptance evidence only.
- Does not promote `main`; promotion requires separate explicit authorization.

- [ ] **Step 1: Add the final terminal-precedence regression test**

Append:

```python

def test_terminal_truth_takes_precedence_over_zero_depth_boundary() -> None:
    board = [
        0, 0, 2, 0,
        1, 1, 1, 2,
    ]

    zero_depth = solve_position(board, 2, 4, 2, 3, max_depth=0)
    exhaustive = solve_position(board, 2, 4, 2, 3)

    assert zero_depth == PositionSolution("loss", (), True)
    assert zero_depth == exhaustive
```

This should pass if Task 3 ordering remains intact. If it fails, correct only the terminal/depth ordering defect.

- [ ] **Step 2: Run the focused CX-04 gate**

```powershell
Write-Host "`n=== FOCUSED CX-04 ==="
uv run pytest tests/test_search.py -q
Write-Host "`n=== FOCUSED RUFF ==="
uv run ruff check src/connect_x_agent/search.py tests/test_search.py
Write-Host "`n=== FOCUSED MYPY ==="
uv run mypy src/connect_x_agent/search.py tests/test_search.py
```

Expected: focused tests pass; Ruff and Mypy clean.

- [ ] **Step 3: Commit the final regression test**

```powershell
git add tests/test_search.py
git commit -m "test: lock CX-04 terminal precedence"
```

- [ ] **Step 4: Run the full repository acceptance gate**

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

Expected: focused and full suites pass; Ruff and Mypy clean; diff check has no errors. Report unrelated untracked files rather than deleting or changing them.

- [ ] **Step 5: Audit scope and non-goals**

```powershell
Write-Host "`n=== CX-04 CHANGED FILES ==="
git diff --name-only main...HEAD
Write-Host "`n=== CX-04 DIFF STAT ==="
git diff --stat main...HEAD
Write-Host "`n=== CX-04 HEAD ==="
git rev-parse HEAD
```

Beyond the already-approved spec and plan, implementation scope must be exactly:

```text
src/connect_x_agent/search.py
tests/test_search.py
```

By inspection, confirm `search.py` contains none of:

```text
best-move or action-selection policy
random choice
heuristic position scoring
alpha-beta parameters or pruning
transposition cache
symmetry canonicalization
bitboard representation
wall-clock timing
iterative deepening
Kaggle environment calls
persistent solution database
```

Confirm every root legal column is returned in deterministic ascending-column order.

Record the exact focused-test count, full-suite count, Ruff result, Mypy result, diff-check result, status output, changed paths, and branch HEAD SHA. Do not claim CX-04 complete if any gate is missing or stale.

Stop at the promotion boundary. Do not merge, fast-forward `main`, push `main`, delete the feature branch, or create a PR without separate explicit authorization.
