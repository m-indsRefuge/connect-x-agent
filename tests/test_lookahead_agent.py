from types import SimpleNamespace

from connect_x_agent.lookahead import lookahead_agent


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


def test_lookahead_agent_takes_immediate_win() -> None:
    board = empty_board()
    board[35:42] = [1, 1, 1, 0, 0, 0, 0]

    assert lookahead_agent(
        observation(board),
        configuration(),
    ) == 3


def test_lookahead_agent_blocks_immediate_loss() -> None:
    board = empty_board()
    board[35:42] = [2, 2, 2, 0, 0, 0, 0]

    assert lookahead_agent(
        observation(board),
        configuration(),
    ) == 3


def test_lookahead_agent_avoids_enabling_opponent_win() -> None:
    board = empty_board()

    board[28:35] = [0, 2, 2, 2, 0, 0, 0]
    board[35:42] = [0, 1, 2, 1, 0, 0, 0]

    move = lookahead_agent(
        observation(board),
        configuration(),
    )

    assert move != 0


def test_lookahead_agent_preserves_leftmost_safe_fallback() -> None:
    assert lookahead_agent(
        observation(empty_board()),
        configuration(),
    ) == 0
