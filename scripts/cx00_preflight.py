from importlib.metadata import version
from typing import Any

from kaggle_environments import make  # type: ignore[import-untyped]


def baseline_agent(observation: Any, configuration: Any) -> int:
    """Choose the leftmost legal column."""
    for column in range(configuration.columns):
        if observation.board[column] == 0:
            return column

    raise RuntimeError("No legal moves available")


def run_match(agent_a: Any, agent_b: Any) -> tuple[list[str], list[float | None]]:
    env = make("connectx", debug=True)
    steps = env.run([agent_a, agent_b])
    final_state = steps[-1]

    statuses = [state.status for state in final_state]
    rewards = [state.reward for state in final_state]

    return statuses, rewards


def main() -> None:
    env = make("connectx", debug=True)

    print("=== CONNECT X CX-00 PREFLIGHT ===")
    print(f"PACKAGE: {version('kaggle-environments')}")
    print(f"ENVIRONMENT: {env.specification.name}")
    print(f"ENV VERSION: {env.specification.version}")
    print(f"COLUMNS: {env.configuration.columns}")
    print(f"ROWS: {env.configuration.rows}")
    print(f"INAROW: {env.configuration.inarow}")
    print(f"ACT TIMEOUT: {env.configuration.actTimeout}")
    print(f"AGENTS: {sorted(env.agents.keys())}")

    observation = env.state[0].observation

    print(f"BOARD CELLS: {len(observation.board)}")
    print(f"P1 MARK: {env.state[0].observation.mark}")
    print(f"P2 MARK: {env.state[1].observation.mark}")
    print(
        "REMAINING OVERAGE TIME:",
        observation.remainingOverageTime,
    )

    assert env.configuration.columns == 7
    assert env.configuration.rows == 6
    assert env.configuration.inarow == 4
    assert env.configuration.actTimeout == 2
    assert len(observation.board) == 42
    assert observation.mark == 1
    assert env.state[1].observation.mark == 2
    assert "random" in env.agents
    assert "negamax" in env.agents

    print("\n=== BASELINE AS PLAYER 1 ===")
    statuses, rewards = run_match(baseline_agent, "random")
    print("STATUSES:", statuses)
    print("REWARDS:", rewards)
    assert statuses == ["DONE", "DONE"]

    print("\n=== BASELINE AS PLAYER 2 ===")
    statuses, rewards = run_match("random", baseline_agent)
    print("STATUSES:", statuses)
    print("REWARDS:", rewards)
    assert statuses == ["DONE", "DONE"]

    print("\nCX-00 PREFLIGHT PASS")


if __name__ == "__main__":
    main()