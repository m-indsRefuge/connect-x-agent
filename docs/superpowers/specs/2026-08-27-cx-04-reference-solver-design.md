# CX-04 Reference Position Solver Design

Date: 2026-08-27
Status: Approved design, pending implementation plan
Branch: `build/cx-04-reference-solver`

## Purpose

CX-04 introduces the first exact adversarial search layer for the Connect X agent.

Its purpose is not to become the final Kaggle submission engine. CX-04 is a correctness-first mathematical reference implementation that can establish exact game-theoretic truth on positions that can be searched to terminal completion, and explicitly report unresolved branches when a deliberate search-depth boundary prevents proof.

The reference solver will become the oracle against which later search optimizations are checked. Alpha-beta pruning, transposition tables, symmetry reduction, compact board encodings, iterative deepening, time control, and heuristic frontier evaluation must preserve CX-04's exact results wherever CX-04 produces complete proofs.

The central design principle is:

> The solver answers what is mathematically true about every legal continuation it can prove. It does not choose a competition action.

## Scope

CX-04 will add an exact recursive position solver with:

- complete legal-move analysis at the root position;
- `WIN`, `DRAW`, `LOSS`, and `UNKNOWN` values;
- player-to-move-relative game values;
- explicit partial-proof semantics;
- exhaustive solving when no depth limit is supplied and the position is computationally tractable;
- deterministic depth-bounded solving when a maximum depth is supplied;
- strict validation of local Connect X position invariants;
- deterministic, reproducible results;
- no move-selection policy.

CX-04 will not include:

- alpha-beta pruning;
- transposition tables;
- symmetry canonicalization;
- bitboards;
- heuristic position evaluation;
- learned models;
- opening books;
- persistent solved-position databases;
- randomization;
- wall-clock timing;
- iterative deepening;
- Kaggle action selection or champion promotion.

## Existing foundations

CX-04 will reuse the existing game mechanics rather than duplicate them.

`src/connect_x_agent/tactical.py` already provides:

- legal column generation;
- gravity-respecting piece drops;
- terminal win detection.

`src/connect_x_agent/game_theory.py` already provides richer geometric concepts such as:

- winning windows;
- viable windows;
- landing cells;
- threats;
- immediate winning columns;
- fork moves;
- surviving replies.

Those game-theory concepts remain intentionally outside CX-04's value calculation. CX-04 must establish exact values only by terminal game outcomes and recursive adversarial search. Strategic features may later support move ordering or frontier evaluation, but they must not contaminate the reference oracle.

## Architectural role

The project will separate the exact reference solver from the future competition engine.

```text
DEVELOPMENT / ORACLE

position
   ↓
CX-04 reference solver
   ↓
exact move classification where proven
   ↓
trusted fixtures and regression truth


FUTURE KAGGLE ENGINE

position
   ↓
optimized search
   ├─ alpha-beta
   ├─ transposition cache
   ├─ symmetry
   ├─ compact board representation
   ├─ iterative deepening
   └─ time management
   ↓
competition policy
   ↓
legal action
```

CX-04 may be slow. That is acceptable. Its value is transparency and correctness.

## Core data model

### GameValue

```python
GameValue = Literal["win", "draw", "loss", "unknown"]
```

Values are always expressed from the perspective of the player whose turn it is in the position being described, or from the perspective of the player making a root move in `MoveAnalysis`.

Semantics:

- `win`: the player can force a win against perfect opposition;
- `draw`: the player can force at least a draw, but not a win, and the result is fully proven;
- `loss`: every legal continuation loses against perfect opposition;
- `unknown`: the exact optimal game-theoretic result has not been proven under the current search boundary.

`unknown` is not a heuristic score, draw assumption, or malformed-input state.

### MoveAnalysis

```python
@dataclass(frozen=True)
class MoveAnalysis:
    column: int
    value: GameValue
```

A `MoveAnalysis` classifies one legal move from the current position.

Example:

```text
column 0 → loss
column 1 → win
column 2 → draw
column 3 → unknown
```

The classification is from the perspective of the player making that move.

### PositionSolution

```python
@dataclass(frozen=True)
class PositionSolution:
    value: GameValue
    moves: tuple[MoveAnalysis, ...]
    complete: bool
```

`moves` contains one entry for every legal root move, in deterministic legal-column order.

`complete` is true exactly when every legal root move has a non-`unknown` value. Terminal positions have no legal continuation analysis and are explicitly complete.

`value` describes the strongest game-theoretic result currently proven for the player to move, subject to the partial-proof rules below.

CX-04 intentionally does not expose `best_move()` and does not embed child `PositionSolution` objects recursively.

## Why the solver returns every legal move

The solver must preserve the complete root decision set rather than collapse the result to one move.

Example:

```text
PositionSolution
├─ value = win
├─ complete = true
└─ moves
   ├─ 0 → loss
   ├─ 1 → win
   ├─ 2 → draw
   ├─ 3 → win
   ├─ 4 → loss
   └─ 5 → draw
```

This supports several later requirements:

- distinguishing game-theoretically equivalent winning moves;
- separating mathematical truth from move-selection policy;
- detecting opponent deviations from perfect play;
- testing future optimized search against complete root classifications;
- adding secondary competition policies later without modifying the solver.

The future competition agent will consume a `PositionSolution` and decide how to act. That policy may eventually prefer fast wins, resilient draws, difficult-to-convert losses, or randomized choices among genuinely equivalent alternatives. None of those policies belong in CX-04.

## Recursive game semantics

CX-04 uses player-to-move-relative values.

After the current player makes a legal nonterminal move, the resulting child position belongs to the opponent.

The value inversion rule is:

```text
child position = loss for opponent
→ current move = win

child position = draw for opponent
→ current move = draw

child position = win for opponent
→ current move = loss

child position = unknown
→ current move = unknown
```

An immediate winning move is classified directly as `win` without recursing into an impossible post-win continuation.

## Search modes

The public solver contract is conceptually:

```python
solve_position(
    board,
    mark,
    columns,
    rows,
    inarow,
    max_depth=None,
)
```

`mark` is the player whose turn it is.

`max_depth` must be `None` or a non-negative integer. Negative values are invalid and raise `ValueError`.

### Exhaustive mode

`max_depth=None` means the solver will recurse until terminal states are reached for every explored branch.

This mode is intended for:

- small Connect X configurations that can be fully solved cheaply;
- deliberately chosen late-game standard 7×6 positions;
- exact reference fixtures.

CX-04 is not required to solve the empty standard 7×6 starting board in practical time.

### Depth-bounded mode

A non-negative integer `max_depth` limits the number of future plies searched from the current position.

When the solver reaches depth zero on a nonterminal recursive node, that node is `unknown`.

At the public root, `max_depth=0` still returns one `MoveAnalysis` for every legal root column, but every such move is `unknown`; the overall position is `unknown` and `complete=False`. No root move is simulated because no future ply is inside the search boundary.

The depth boundary is deterministic. CX-04 does not use elapsed wall-clock time.

This makes repeated calls with the same position and depth reproducible across runs and provides a stable oracle for later optimized implementations.

## Root completeness and partial proofs

CX-04 preserves proven mathematical facts even when some root moves remain unresolved.

`complete` is true only when no root `MoveAnalysis` has value `unknown`, except that a terminal position with `moves=()` is explicitly complete.

The position-value aggregation rule is:

```text
if any move is WIN:
    position value = WIN

else if any move is UNKNOWN:
    position value = UNKNOWN

else if any move is DRAW:
    position value = DRAW

else:
    position value = LOSS
```

Examples:

```text
LOSS, WIN, UNKNOWN
→ value = WIN
→ complete = false
```

A forced win is already proven; an unresolved alternative cannot invalidate the existence of that winning continuation.

```text
LOSS, DRAW, UNKNOWN
→ value = UNKNOWN
→ complete = false
```

The unresolved move might prove to be a win, so the true optimal value is not yet known.

```text
LOSS, LOSS, DRAW
→ value = DRAW
→ complete = true
```

```text
LOSS, LOSS, LOSS
→ value = LOSS
→ complete = true
```

The governing principle is:

> CX-04 must never claim a stronger optimal result than it has proven.

## Terminal-state semantics

Terminal positions are exact and complete.

### Previous player has won

If the board already contains a valid winning line for the player who made the previous move, then the player nominally to move has already lost:

```text
value = LOSS
complete = True
moves = ()
```

No search continues below a won board.

A won input board must also pass the last-move consistency check described under position validation; CX-04 must reject a board that can only exist if play continued after an earlier terminal win.

### Full board without a winner

If the board is full and neither player has a winning line:

```text
value = DRAW
complete = True
moves = ()
```

### Immediate win after a simulated move

If a root or recursive move itself creates Connect X, that move is `WIN` for the mover and recursion stops on that branch.

## Position validation

Malformed positions must be rejected rather than represented as `unknown`.

CX-04 will validate strong local invariants before search:

1. `columns`, `rows`, and `inarow` are positive integers;
2. board length equals `columns * rows`;
3. every cell is one of `0`, `1`, or `2`;
4. `mark` is `1` or `2`;
5. occupied cells obey gravity within each column;
6. piece counts are consistent with alternating play and the supplied side to move;
7. both players cannot simultaneously have winning lines;
8. if a winner is present, the winner must be the player who moved immediately before the supplied side to move;
9. winner/count relationships must be consistent with normal alternating play;
10. a won board must have at least one topmost checker belonging to the winner whose removal produces a gravity-valid predecessor with no winner and with piece counts consistent with the turn before the winning move.

Rule 10 is a local last-move consistency proof. It prevents the reference solver from accepting boards that require moves to have been played after the game should already have ended.

Invalid positions raise `ValueError`.

CX-04 will not attempt full historical reachability proof for every arbitrary board. It validates the local invariants required for trustworthy recursive solving. Proving that every arbitrary board can be generated by some exact legal move history is outside this task.

## Determinism

CX-04 must be deterministic.

For identical inputs:

- legal moves appear in the same column order;
- move classifications are identical;
- `PositionSolution` equality is stable;
- no randomness or timing affects results.

Determinism is necessary because CX-04 will be used as a regression oracle for later search implementations.

## Strategy generation without storing the whole game tree

CX-04 does not persist a giant strategy tree or solved-position database.

The future live-agent model is:

```text
actual board
   ↓
solve/analyze current position
   ↓
policy chooses an action
   ↓
opponent makes any legal response
   ↓
new actual board
   ↓
solve/analyze that board
```

The agent therefore carries the ability to derive strategy from the current state rather than attempting to store every possible future response.

Later transposition caching may retain a bounded set of useful solved nodes in memory, but that is an optimization layer and not part of the `PositionSolution` contract.

## Kaggle size and runtime implications

CX-04 itself does not need to meet the final live-action performance target. It establishes the reference semantics from which the optimized engine will be derived.

The final Kaggle architecture will remain compact because it stores algorithms rather than a complete tablebase. Large neural models, giant opening books, or multi-gigabyte solved-position databases are not part of the current plan.

The project treats final submission size as an explicit later acceptance gate. The intended live engine will be composed of compact Python game logic, search algorithms, bounded in-memory caches, and only small derived data if later justified.

The harder final constraint is expected to be the action-time budget rather than source size. CX-05 through CX-07 will address search efficiency and time-bounded execution while preserving CX-04 reference truth.

## Testing strategy

CX-04 will be implemented TDD-first.

The reference test suite must establish game semantics, not merely code coverage.

Required behavior includes:

1. immediate winning move is classified `WIN`;
2. forced-loss positions are classified `LOSS` when exhaustively solved;
3. forced-draw positions are classified `DRAW` when exhaustively solved;
4. multiple winning legal moves are all preserved;
5. mixed `WIN`, `DRAW`, and `LOSS` root continuations are classified independently;
6. child `LOSS` correctly inverts to current-player `WIN`;
7. child `WIN` correctly inverts to current-player `LOSS`;
8. a live recursive node at the depth boundary becomes `UNKNOWN`;
9. root `max_depth=0` returns every legal root move as `UNKNOWN`, with overall `UNKNOWN` and `complete=False`;
10. negative `max_depth` is rejected;
11. a proven winning move can coexist with unresolved alternatives as `WIN` plus `complete=False`;
12. `DRAW` plus an unresolved alternative yields overall `UNKNOWN`;
13. a fully exhaustive solution contains no `UNKNOWN` root moves;
14. a terminal won board returns exact `LOSS`, `complete=True`, and no legal continuation analysis;
15. a terminal full-board draw returns exact `DRAW`, `complete=True`, and no moves;
16. invalid board dimensions are rejected;
17. invalid cell values are rejected;
18. gravity violations are rejected;
19. impossible piece counts or side-to-move relationships are rejected;
20. simultaneous winners are rejected;
21. winner/turn inconsistencies are rejected;
22. a won board that requires play after an earlier win is rejected by last-move consistency validation;
23. repeated calls with identical inputs return identical results.

Small configurations will provide fully exhaustive fixtures because their complete game trees are cheap enough to solve directly.

Selected late-game 7×6 positions will demonstrate exact solving on the competition board geometry without requiring empty-board full solution.

Early standard 7×6 positions may be used specifically to verify bounded `UNKNOWN` behavior.

## Proposed implementation boundary

The intended implementation is deliberately narrow:

```text
src/connect_x_agent/search.py
tests/test_search.py
```

The implementation plan may split the work into multiple TDD slices, but unrelated repository files should not be modified unless a proven integration need appears during planning.

`search.py` will consume the existing mechanics in `tactical.py` rather than creating a second independent representation of legal moves, piece dropping, or winning logic.

No Kaggle-facing agent entry point changes are part of CX-04.

## Acceptance criteria

CX-04 is accepted when:

- the approved data contracts exist;
- valid terminal positions receive exact values;
- every legal root move is independently classified;
- partial-proof semantics match this specification;
- invalid local positions are rejected;
- small-board exhaustive fixtures produce exact complete solutions;
- bounded searches report `UNKNOWN` rather than heuristic estimates;
- results are deterministic;
- no move-selection policy or later optimization layer has leaked into the reference solver;
- focused search tests pass;
- the full repository test suite passes;
- Ruff passes;
- Mypy passes;
- `git diff --check` is clean;
- changed-file scope matches the approved implementation plan.

## Future phases

CX-04 establishes semantics that later phases must preserve.

The current intended progression is:

```text
CX-04  reference minimax position solver
   ↓
CX-05  alpha-beta pruning
   ↓
CX-06  compact state representation + transpositions + symmetry
   ↓
CX-07  iterative deepening + competition time management
   ↓
CX-08  strategic frontier evaluation / solver-informed competition policy
```

A future learned component, if justified, would guide or evaluate search rather than replace legality, tactical truth, or adversarial calculation.

## Explicit design decisions

The following decisions are locked for CX-04:

- correctness-first reference solver, not immediate Kaggle optimization;
- calculate every legal root move rather than expose only one best move;
- solver and move-selection policy remain separate;
- values are relative to the player to move;
- exact outcomes are `WIN`, `DRAW`, and `LOSS`;
- unresolved bounded branches are explicitly `UNKNOWN`;
- a proven `WIN` may be reported even if other root moves remain unknown;
- unresolved alternatives prevent a `DRAW` or `LOSS` claim from being treated as exact optimal value;
- exhaustive and deterministic depth-bounded modes are supported;
- `max_depth=0` has explicit all-root-unknown semantics;
- no wall-clock cutoff is used in the reference solver;
- no heuristics influence exact values;
- won input states require local last-move consistency;
- no child strategy tree is embedded in `PositionSolution`;
- no persistent complete-game tablebase is part of the architecture;
- final submission compactness comes from algorithms, not memorizing every Connect Four position.
