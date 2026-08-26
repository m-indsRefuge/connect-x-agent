from connect_x_agent.arena import ArenaReport, summarize_rewards


def test_summarize_rewards_for_player_one() -> None:
    rewards = [
        [1, -1],
        [0, 0],
        [-1, 1],
        [1, -1],
    ]

    result = summarize_rewards(rewards, candidate_index=0)

    assert result.games == 4
    assert result.wins == 2
    assert result.draws == 1
    assert result.losses == 1
    assert result.failures == 0


def test_summarize_rewards_for_player_two() -> None:
    rewards = [
        [1, -1],
        [0, 0],
        [-1, 1],
    ]

    result = summarize_rewards(rewards, candidate_index=1)

    assert result.games == 3
    assert result.wins == 1
    assert result.draws == 1
    assert result.losses == 1
    assert result.failures == 0


def test_unknown_reward_is_recorded_as_failure() -> None:
    rewards = [
        [None, 0],
    ]

    result = summarize_rewards(rewards, candidate_index=0)

    assert result.games == 1
    assert result.failures == 1


def test_arena_report_combines_both_seats() -> None:
    report = ArenaReport.from_rewards(
        player_one_rewards=[
            [1, -1],
            [-1, 1],
        ],
        player_two_rewards=[
            [-1, 1],
            [0, 0],
        ],
    )

    assert report.games == 4
    assert report.wins == 2
    assert report.draws == 1
    assert report.losses == 1
    assert report.failures == 0


def test_arena_runs_candidate_in_both_seats() -> None:
    from connect_x_agent.agent import agent
    from connect_x_agent.arena import run_arena

    report = run_arena(
        candidate=agent,
        opponent="random",
        games_per_seat=2,
    )

    assert report.games == 4
    assert report.player_one.games == 2
    assert report.player_two.games == 2
    assert report.failures == 0
