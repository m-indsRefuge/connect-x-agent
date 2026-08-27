from typing import Any

from connect_x_agent.search import PositionSolution, solve_position
from connect_x_agent.tactical import tactical_agent

SOLVER_DEPTH = 4

_VALUE_PRIORITY = (
    "win",
    "draw",
    "unknown",
    "loss",
)


def _choose_solver_move(
    solution: PositionSolution,
) -> int:
    for value in _VALUE_PRIORITY:
        for move in solution.moves:
            if move.value == value:
                return move.column

    raise RuntimeError(
        "Solver returned no legal root moves"
    )


def _choose_hybrid_move(
    solution: PositionSolution,
    tactical_column: int,
) -> int:
    for move in solution.moves:
        if move.value == "win":
            return move.column

    for move in solution.moves:
        if move.value == "draw":
            return move.column

    tactical_value = next(
        (
            move.value
            for move in solution.moves
            if move.column == tactical_column
        ),
        None,
    )

    if tactical_value != "loss":
        return tactical_column

    for move in solution.moves:
        if move.value == "unknown":
            return move.column

    for move in solution.moves:
        if move.value == "loss":
            return move.column

    raise RuntimeError(
        "Solver returned no legal root moves"
    )


def solver_agent(
    observation: Any,
    configuration: Any,
) -> int:
    solution = solve_position(
        board=list(observation.board),
        mark=int(observation.mark),
        columns=int(configuration.columns),
        rows=int(configuration.rows),
        inarow=int(configuration.inarow),
        max_depth=SOLVER_DEPTH,
    )

    return _choose_solver_move(
        solution
    )


def hybrid_solver_agent(
    observation: Any,
    configuration: Any,
) -> int:
    solution = solve_position(
        board=list(observation.board),
        mark=int(observation.mark),
        columns=int(configuration.columns),
        rows=int(configuration.rows),
        inarow=int(configuration.inarow),
        max_depth=SOLVER_DEPTH,
    )

    tactical_column = tactical_agent(
        observation,
        configuration,
    )

    return _choose_hybrid_move(
        solution,
        tactical_column,
    )
