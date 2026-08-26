from typing import Any

from kaggle_environments import make  # type: ignore[import-untyped]

from connect_x_agent.agent import agent


def assert_completed_game(env: Any) -> None:
    statuses = [state.status for state in env.state]

    assert statuses == ["DONE", "DONE"]
    assert all(state.reward is not None for state in env.state)


def test_connectx_runtime_contract() -> None:
    env = make("connectx", debug=True)

    assert env.configuration.columns == 7
    assert env.configuration.rows == 6
    assert env.configuration.inarow == 4
    assert env.configuration.actTimeout == 2

    assert len(env.state[0].observation.board) == 42

    assert "random" in env.agents
    assert "negamax" in env.agents


def test_agent_completes_game_as_player_one() -> None:
    env = make("connectx", debug=True)

    env.run([agent, "random"])

    assert_completed_game(env)


def test_agent_completes_game_as_player_two() -> None:
    env = make("connectx", debug=True)

    env.run(["random", agent])

    assert_completed_game(env)
