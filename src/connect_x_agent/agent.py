from typing import Any


def agent(observation: Any, configuration: Any) -> int:
    """Return the leftmost legal Connect X column."""
    for column in range(configuration.columns):
        if observation.board[column] == 0:
            return column

    raise RuntimeError("No legal moves available")