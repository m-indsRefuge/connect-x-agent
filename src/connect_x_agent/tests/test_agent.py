from types import SimpleNamespace

from connect_x_agent.agent import agent


def make_configuration(columns: int = 7) -> SimpleNamespace:
    return SimpleNamespace(columns=columns)


def make_observation(board: list[int]) -> SimpleNamespace:
    return SimpleNamespace(board=board)


def test_agent_selects_leftmost_legal_column() -> None:
    observation = make_observation([0] * 42)
    configuration = make_configuration()

    assert agent(observation, configuration) == 0


def test_agent_skips_full_columns() -> None:
    board = [0] * 42

    # Top cells for columns 0 and 1 are occupied.
    board[0] = 1
    board[1] = 2

    observation = make_observation(board)
    configuration = make_configuration()

    assert agent(observation, configuration) == 2


def test_agent_selects_only_legal_column() -> None:
    board = [1, 1, 1, 0, 2, 2, 2] + [0] * 35

    observation = make_observation(board)
    configuration = make_configuration()

    assert agent(observation, configuration) == 3