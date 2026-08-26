from types import SimpleNamespace

from connect_x_agent.tactical import tactical_agent


def observation(board: list[int], mark: int = 1) -> SimpleNamespace:
    return SimpleNamespace(board=board, mark=mark)


def configuration() -> SimpleNamespace:
    return SimpleNamespace(
        columns=7,
        rows=6,
        inarow=4,
    )


def empty_board() -> list[int]:
    return [0] * 42


def test_tactical_agent_takes_immediate_horizontal_win() -> None:
    board = empty_board()
    board[35:42] = [1, 1, 1, 0, 0, 0, 0]

    move = tactical_agent(
        observation(board),
        configuration(),
    )

    assert move == 3


def test_tactical_agent_takes_immediate_vertical_win() -> None:
    board = empty_board()
    board[35] = 1
    board[28] = 1
    board[21] = 1

    move = tactical_agent(
        observation(board),
        configuration(),
    )

    assert move == 0


def test_tactical_agent_blocks_immediate_opponent_win() -> None:
    board = empty_board()
    board[35:42] = [2, 2, 2, 0, 0, 0, 0]

    move = tactical_agent(
        observation(board),
        configuration(),
    )

    assert move == 3


def test_tactical_agent_preserves_cx01_fallback() -> None:
    move = tactical_agent(
        observation(empty_board()),
        configuration(),
    )

    assert move == 0
