from connect_x_agent.study.instrumentation import position_features


def test_empty_standard_board_has_symmetric_first_principles_features() -> None:
    features = position_features(
        board=[0] * 42,
        columns=7,
        rows=6,
        inarow=4,
    )

    assert features.legal_columns == (0, 1, 2, 3, 4, 5, 6)
    assert features.p1_viable_windows == 69
    assert features.p2_viable_windows == 69
    assert features.p1_playable_threats == 0
    assert features.p2_playable_threats == 0
    assert features.p1_latent_threats == 0
    assert features.p2_latent_threats == 0
    assert features.p1_winning_columns == ()
    assert features.p2_winning_columns == ()
    assert features.p1_fork_moves == ()
    assert features.p2_fork_moves == ()
    assert features.p1_surviving_replies == (0, 1, 2, 3, 4, 5, 6)
    assert features.p2_surviving_replies == (0, 1, 2, 3, 4, 5, 6)


def test_playable_threat_position_records_expected_counts_and_forced_reply() -> None:
    board = [0] * 42
    board[35:38] = [1, 1, 1]

    features = position_features(
        board,
        columns=7,
        rows=6,
        inarow=4,
    )

    assert features.p1_playable_threats == 1
    assert features.p1_latent_threats == 0
    assert features.p2_playable_threats == 0
    assert features.p2_latent_threats == 0
    assert features.p1_winning_columns == (3,)
    assert features.p2_winning_columns == ()
    assert features.p2_surviving_replies == (3,)


def test_genuine_nonstandard_fork_position_records_its_candidate_column() -> None:
    board = [
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        1,
        2,
        0,
    ]

    features = position_features(
        board,
        columns=4,
        rows=3,
        inarow=3,
    )

    assert features.legal_columns == (0, 1, 2, 3)
    assert features.p1_fork_moves == (1,)
    assert features.p2_fork_moves == ()


def test_position_instrumentation_does_not_mutate_its_board() -> None:
    board = [
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        1,
        2,
        0,
    ]
    original_board = board.copy()

    position_features(
        board,
        columns=4,
        rows=3,
        inarow=3,
    )

    assert board == original_board


def test_empty_nonstandard_board_uses_its_own_geometry() -> None:
    features = position_features(
        board=[0] * 12,
        columns=4,
        rows=3,
        inarow=3,
    )

    assert features.legal_columns == (0, 1, 2, 3)
    assert features.p1_viable_windows == 14
    assert features.p2_viable_windows == 14
