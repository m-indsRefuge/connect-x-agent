from connect_x_agent import game_theory
from connect_x_agent.game_theory import winning_windows

STANDARD_CONNECT_FOUR_INFLUENCE = [
    3,
    4,
    5,
    7,
    5,
    4,
    3,
    4,
    6,
    8,
    10,
    8,
    6,
    4,
    5,
    8,
    11,
    13,
    11,
    8,
    5,
    5,
    8,
    11,
    13,
    11,
    8,
    5,
    4,
    6,
    8,
    10,
    8,
    6,
    4,
    3,
    4,
    5,
    7,
    5,
    4,
    3,
]


def _landing_cell_for_test(
    board: list[int],
    column: int,
    columns: int,
    rows: int,
) -> int | None:
    for row in range(rows - 1, -1, -1):
        cell = row * columns + column

        if board[cell] == 0:
            return cell

    return None


def _board_after_legal_drop(
    board: list[int],
    mark: int,
    column: int,
    columns: int,
    rows: int,
) -> list[int]:
    target = _landing_cell_for_test(
        board,
        column,
        columns,
        rows,
    )
    assert target is not None

    board_after_drop = board.copy()
    board_after_drop[target] = mark
    return board_after_drop


def _has_winning_window(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> bool:
    return any(
        all(board[cell] == mark for cell in window)
        for window in winning_windows(
            columns=columns,
            rows=rows,
            inarow=inarow,
        )
    )


def _immediate_winning_columns_from_geometry(
    board: list[int],
    mark: int,
    columns: int,
    rows: int,
    inarow: int,
) -> tuple[int, ...]:
    winning_columns: list[int] = []

    for column in range(columns):
        if _landing_cell_for_test(board, column, columns, rows) is None:
            continue

        board_after_drop = _board_after_legal_drop(
            board,
            mark,
            column,
            columns,
            rows,
        )

        if _has_winning_window(
            board_after_drop,
            mark,
            columns,
            rows,
            inarow,
        ):
            winning_columns.append(column)

    return tuple(winning_columns)


def test_standard_connect_four_has_69_winning_windows() -> None:
    windows = winning_windows(
        columns=7,
        rows=6,
        inarow=4,
    )

    assert len(windows) == 69
    assert len(set(windows)) == 69


def test_winning_windows_include_all_directions() -> None:
    windows = set(
        winning_windows(
            columns=7,
            rows=6,
            inarow=4,
        )
    )

    assert (0, 1, 2, 3) in windows
    assert (0, 7, 14, 21) in windows
    assert (0, 8, 16, 24) in windows
    assert (3, 9, 15, 21) in windows


def test_every_winning_window_contains_four_valid_cells() -> None:
    windows = winning_windows(
        columns=7,
        rows=6,
        inarow=4,
    )

    for window in windows:
        assert len(window) == 4
        assert len(set(window)) == 4
        assert all(0 <= cell < 42 for cell in window)


def test_winning_windows_support_nonstandard_positive_dimensions() -> None:
    windows = winning_windows(
        columns=4,
        rows=3,
        inarow=3,
    )

    assert len(windows) == 14
    assert len(set(windows)) == 14
    assert (0, 1, 2) in windows
    assert (0, 4, 8) in windows
    assert (0, 5, 10) in windows
    assert (2, 5, 8) in windows


def test_one_piece_wins_are_unique() -> None:
    assert winning_windows(
        columns=2,
        rows=2,
        inarow=1,
    ) == [(0,), (1,), (2,), (3,)]


def test_standard_connect_four_cell_influence_equals_expected_map() -> None:
    assert game_theory.cell_influence(
        columns=7,
        rows=6,
        inarow=4,
    ) == STANDARD_CONNECT_FOUR_INFLUENCE


def test_standard_connect_four_cell_influence_maximum_is_thirteen() -> None:
    influence = game_theory.cell_influence(
        columns=7,
        rows=6,
        inarow=4,
    )

    assert max(influence) == 13


def test_standard_connect_four_corner_cell_influence_is_three() -> None:
    influence = game_theory.cell_influence(
        columns=7,
        rows=6,
        inarow=4,
    )

    assert influence[0] == influence[6] == influence[35] == influence[41] == 3


def test_standard_connect_four_cell_influence_has_horizontal_symmetry() -> None:
    influence = game_theory.cell_influence(
        columns=7,
        rows=6,
        inarow=4,
    )
    influence_rows = [
        influence[row * 7 : (row + 1) * 7]
        for row in range(6)
    ]

    assert influence_rows == [list(reversed(row)) for row in influence_rows]


def test_standard_connect_four_cell_influence_has_vertical_symmetry() -> None:
    influence = game_theory.cell_influence(
        columns=7,
        rows=6,
        inarow=4,
    )
    influence_rows = [
        influence[row * 7 : (row + 1) * 7]
        for row in range(6)
    ]

    assert influence_rows == list(reversed(influence_rows))


def test_cell_influence_supports_smaller_nonstandard_geometry() -> None:
    assert game_theory.cell_influence(
        columns=4,
        rows=3,
        inarow=3,
    ) == [
        3,
        4,
        4,
        3,
        2,
        5,
        5,
        2,
        3,
        4,
        4,
        3,
    ]


def test_empty_board_keeps_all_winning_windows_viable_for_both_players() -> None:
    board = [0] * 42
    expected_windows = winning_windows(
        columns=7,
        rows=6,
        inarow=4,
    )

    assert game_theory.viable_windows(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == expected_windows
    assert game_theory.viable_windows(
        board,
        mark=2,
        columns=7,
        rows=6,
        inarow=4,
    ) == expected_windows


def test_window_with_only_current_player_checkers_and_empty_cells_is_viable() -> None:
    board = [1, 0, 1, 0]

    assert game_theory.viable_windows(
        board,
        mark=1,
        columns=4,
        rows=1,
        inarow=4,
    ) == [(0, 1, 2, 3)]


def test_window_with_both_marks_is_nonviable_for_both_players() -> None:
    board = [1, 0, 2, 0]

    assert game_theory.viable_windows(
        board,
        mark=1,
        columns=4,
        rows=1,
        inarow=4,
    ) == []
    assert game_theory.viable_windows(
        board,
        mark=2,
        columns=4,
        rows=1,
        inarow=4,
    ) == []


def test_viable_windows_does_not_mutate_the_input_board() -> None:
    board = [1, 0, 1, 0]
    original_board = board.copy()

    game_theory.viable_windows(
        board,
        mark=1,
        columns=4,
        rows=1,
        inarow=4,
    )

    assert board == original_board


def test_empty_standard_board_landing_cells_are_all_in_the_bottom_row() -> None:
    assert game_theory.landing_cells(
        board=[0] * 42,
        columns=7,
        rows=6,
    ) == {
        0: 35,
        1: 36,
        2: 37,
        3: 38,
        4: 39,
        5: 40,
        6: 41,
    }


def test_landing_cell_moves_up_for_each_supported_checker() -> None:
    board = [0] * 42
    board[38] = 1

    assert game_theory.landing_cells(
        board,
        columns=7,
        rows=6,
    )[3] == 31

    board[31] = 2

    assert game_theory.landing_cells(
        board,
        columns=7,
        rows=6,
    )[3] == 24

    board[24] = 1

    assert game_theory.landing_cells(
        board,
        columns=7,
        rows=6,
    )[3] == 17


def test_full_column_has_no_landing_cell() -> None:
    board = [0] * 42

    for row in range(6):
        board[row * 7 + 2] = 1 if row % 2 == 0 else 2

    landing = game_theory.landing_cells(
        board,
        columns=7,
        rows=6,
    )

    assert 2 not in landing
    assert set(landing) == {0, 1, 3, 4, 5, 6}


def test_landing_cells_does_not_mutate_the_input_board() -> None:
    board = [0] * 42
    board[35] = 1
    original_board = board.copy()

    game_theory.landing_cells(
        board,
        columns=7,
        rows=6,
    )

    assert board == original_board


def test_player_one_supported_horizontal_threat_is_playable() -> None:
    board = [0] * 42
    board[35:38] = [1, 1, 1]

    assert game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == [
        game_theory.Threat(
            window=(35, 36, 37, 38),
            target=38,
            playable=True,
        )
    ]


def test_supported_horizontal_one_away_structure_is_latent_when_target_is_above_empty_cell() -> None:
    board = [0] * 42
    board[35:38] = [2, 2, 2]
    board[28:31] = [1, 1, 1]

    assert game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == [
        game_theory.Threat(
            window=(28, 29, 30, 31),
            target=31,
            playable=False,
        )
    ]


def test_supported_vertical_three_of_four_is_playable() -> None:
    board = [0] * 42
    board[35] = 1
    board[28] = 1
    board[21] = 1
    board[40] = 2
    board[41] = 2

    assert game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == [
        game_theory.Threat(
            window=(14, 21, 28, 35),
            target=14,
            playable=True,
        )
    ]


def test_supported_diagonal_three_of_four_is_playable() -> None:
    board = [0] * 42
    for cell in (14, 22, 28, 29, 30):
        board[cell] = 1
    for cell in (21, 35, 36, 37, 41):
        board[cell] = 2

    assert game_theory.Threat(
        window=(14, 22, 30, 38),
        target=38,
        playable=True,
    ) in game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )


def test_diagonal_one_away_structure_is_latent_when_target_is_unsupported() -> None:
    board = [0] * 42
    for cell in (22, 30, 38):
        board[cell] = 1
    for cell in (29, 36, 37):
        board[cell] = 2

    assert game_theory.Threat(
        window=(14, 22, 30, 38),
        target=14,
        playable=False,
    ) in game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )


def test_player_two_supported_horizontal_threat_is_playable() -> None:
    board = [0] * 42
    board[35:38] = [2, 2, 2]
    board[27] = 1
    board[34] = 1
    board[41] = 1

    assert game_theory.threats(
        board,
        mark=2,
        columns=7,
        rows=6,
        inarow=4,
    ) == [
        game_theory.Threat(
            window=(35, 36, 37, 38),
            target=38,
            playable=True,
        )
    ]


def test_mixed_window_is_excluded_for_both_players() -> None:
    board = [1, 1, 2, 0]

    assert game_theory.threats(
        board,
        mark=1,
        columns=4,
        rows=1,
        inarow=4,
    ) == []
    assert game_theory.threats(
        board,
        mark=2,
        columns=4,
        rows=1,
        inarow=4,
    ) == []


def test_completed_winning_window_is_not_a_threat() -> None:
    assert game_theory.threats(
        board=[1, 1, 1, 1],
        mark=1,
        columns=4,
        rows=1,
        inarow=4,
    ) == []


def test_two_of_four_structure_is_not_a_threat() -> None:
    assert game_theory.threats(
        board=[1, 1, 0, 0],
        mark=1,
        columns=4,
        rows=1,
        inarow=4,
    ) == []


def test_playable_threats_complete_their_windows_after_legal_drop() -> None:
    # Deliberately artificial board isolating four overlapping threats.
    board = [0] * 42
    for cell in (35, 36, 37, 39, 40, 41):
        board[cell] = 1
    expected_threats = [
        game_theory.Threat(
            window=(35, 36, 37, 38),
            target=38,
            playable=True,
        ),
        game_theory.Threat(
            window=(36, 37, 38, 39),
            target=38,
            playable=True,
        ),
        game_theory.Threat(
            window=(37, 38, 39, 40),
            target=38,
            playable=True,
        ),
        game_theory.Threat(
            window=(38, 39, 40, 41),
            target=38,
            playable=True,
        ),
    ]

    found_threats = game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )

    assert found_threats == expected_threats
    assert found_threats == game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )
    assert game_theory.landing_cells(
        board,
        columns=7,
        rows=6,
    )[3] == 38

    for threat in found_threats:
        board_after_drop = board.copy()
        board_after_drop[threat.target] = 1

        assert all(board_after_drop[cell] == 1 for cell in threat.window)


def test_threats_does_not_mutate_the_input_board() -> None:
    board = [0] * 42
    board[35:38] = [1, 1, 1]
    original_board = board.copy()

    game_theory.threats(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )

    assert board == original_board


def test_threats_support_smaller_nonstandard_geometry() -> None:
    board = [0] * 12
    board[8] = 1
    board[9] = 1

    assert game_theory.threats(
        board,
        mark=1,
        columns=4,
        rows=3,
        inarow=3,
    ) == [
        game_theory.Threat(
            window=(8, 9, 10),
            target=10,
            playable=True,
        )
    ]


def test_winning_columns_returns_one_playable_winning_column_for_player_one() -> None:
    board = [0] * 42
    board[35:38] = [1, 1, 1]

    assert game_theory.winning_columns(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == (3,)


def test_winning_columns_returns_multiple_distinct_playable_targets() -> None:
    board = [0] * 42
    for cell in (27, 34, 35, 36, 37, 41):
        board[cell] = 1
    for row in range(6):
        board[row * 7 + 4] = 2

    assert game_theory.winning_columns(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == (3, 6)


def test_winning_columns_deduplicates_multiple_windows_with_one_target() -> None:
    # Deliberately artificial board isolating four windows sharing target 38.
    board = [0] * 42
    for cell in (35, 36, 37, 39, 40, 41):
        board[cell] = 1

    assert game_theory.winning_columns(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == (3,)


def test_winning_columns_excludes_latent_threats() -> None:
    board = [0] * 42
    board[35:38] = [2, 2, 2]
    board[28:31] = [1, 1, 1]

    assert game_theory.winning_columns(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == ()


def test_winning_columns_excludes_full_columns() -> None:
    board = [0] * 42

    for row in range(6):
        board[row * 7 + 3] = 1 if row % 2 == 0 else 2

    assert game_theory.winning_columns(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == ()


def test_winning_columns_supports_player_two() -> None:
    board = [0] * 42
    board[35:38] = [2, 2, 2]
    board[27] = 1
    board[34] = 1
    board[41] = 1

    assert game_theory.winning_columns(
        board,
        mark=2,
        columns=7,
        rows=6,
        inarow=4,
    ) == (3,)


def test_fork_moves_finds_two_distinct_playable_targets_on_smaller_board() -> None:
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
    expected_threats = (
        game_theory.Threat(
            window=(1, 5, 9),
            target=1,
            playable=True,
        ),
        game_theory.Threat(
            window=(4, 5, 6),
            target=6,
            playable=True,
        ),
    )

    found_forks = game_theory.fork_moves(
        board,
        mark=1,
        columns=4,
        rows=3,
        inarow=3,
    )

    assert found_forks == {1: expected_threats}

    for candidate_column, resulting_threats in found_forks.items():
        board_after_drop = _board_after_legal_drop(
            board,
            1,
            candidate_column,
            columns=4,
            rows=3,
        )

        assert not _has_winning_window(
            board_after_drop,
            mark=1,
            columns=4,
            rows=3,
            inarow=3,
        )
        assert all(threat.playable for threat in resulting_threats)
        assert len({threat.target for threat in resulting_threats}) == 2


def test_fork_moves_rejects_multiple_windows_with_one_target_cell() -> None:
    board = [
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
        1,
    ]
    board_after_drop = _board_after_legal_drop(
        board,
        1,
        column=0,
        columns=4,
        rows=3,
    )
    playable_targets = [
        threat.target
        for threat in game_theory.threats(
            board_after_drop,
            mark=1,
            columns=4,
            rows=3,
            inarow=3,
        )
        if threat.playable
    ]

    assert playable_targets == [9, 9]
    assert 0 not in game_theory.fork_moves(
        board,
        mark=1,
        columns=4,
        rows=3,
        inarow=3,
    )


def test_fork_moves_rejects_one_playable_and_one_latent_threat() -> None:
    board = [
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        2,
    ]
    board_after_drop = _board_after_legal_drop(
        board,
        1,
        column=2,
        columns=4,
        rows=3,
    )
    resulting_threats = game_theory.threats(
        board_after_drop,
        mark=1,
        columns=4,
        rows=3,
        inarow=3,
    )

    assert [threat.playable for threat in resulting_threats] == [True, False]
    assert 2 not in game_theory.fork_moves(
        board,
        mark=1,
        columns=4,
        rows=3,
        inarow=3,
    )


def test_fork_moves_excludes_terminal_immediate_wins_and_non_forks() -> None:
    board = [0] * 42
    board[35:38] = [1, 1, 1]

    assert 3 not in game_theory.fork_moves(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )
    assert game_theory.fork_moves(
        [0] * 42,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    ) == {}


def test_fork_moves_supports_player_two() -> None:
    board = [
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        2,
        1,
        0,
    ]

    assert 1 in game_theory.fork_moves(
        board,
        mark=2,
        columns=4,
        rows=3,
        inarow=3,
    )


def test_surviving_replies_returns_one_forced_blocking_column() -> None:
    board = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
        1,
    ]

    surviving = game_theory.surviving_replies(
        board,
        defender_mark=2,
        columns=4,
        rows=3,
        inarow=3,
    )

    assert surviving == (1,)

    for column in range(4):
        board_after_drop = _board_after_legal_drop(
            board,
            2,
            column,
            columns=4,
            rows=3,
        )
        opponent_wins = _immediate_winning_columns_from_geometry(
            board_after_drop,
            mark=1,
            columns=4,
            rows=3,
            inarow=3,
        )

        if column in surviving:
            assert opponent_wins == ()
        else:
            assert opponent_wins


def test_surviving_replies_returns_none_against_a_genuine_double_threat() -> None:
    board = [
        0,
        0,
        0,
        2,
        0,
        0,
        1,
        2,
        0,
        0,
        1,
        1,
    ]

    assert _immediate_winning_columns_from_geometry(
        board,
        mark=1,
        columns=4,
        rows=3,
        inarow=3,
    ) == (1, 2)
    assert game_theory.surviving_replies(
        board,
        defender_mark=2,
        columns=4,
        rows=3,
        inarow=3,
    ) == ()

    for column in range(4):
        if _landing_cell_for_test(board, column, columns=4, rows=3) is None:
            continue

        board_after_drop = _board_after_legal_drop(
            board,
            2,
            column,
            columns=4,
            rows=3,
        )

        assert _immediate_winning_columns_from_geometry(
            board_after_drop,
            mark=1,
            columns=4,
            rows=3,
            inarow=3,
        )


def test_surviving_replies_keeps_defenders_immediate_win_without_blocking() -> None:
    board = [
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        1,
        0,
        1,
        2,
        1,
    ]

    assert game_theory.surviving_replies(
        board,
        defender_mark=2,
        columns=4,
        rows=3,
        inarow=3,
    ) == (2, 3)

    board_after_win = _board_after_legal_drop(
        board,
        2,
        column=2,
        columns=4,
        rows=3,
    )

    assert _has_winning_window(
        board_after_win,
        mark=2,
        columns=4,
        rows=3,
        inarow=3,
    )
    assert _immediate_winning_columns_from_geometry(
        board_after_win,
        mark=1,
        columns=4,
        rows=3,
        inarow=3,
    ) == (3,)


def test_surviving_replies_can_return_multiple_legal_columns() -> None:
    assert game_theory.surviving_replies(
        board=[0] * 12,
        defender_mark=1,
        columns=4,
        rows=3,
        inarow=3,
    ) == (0, 1, 2, 3)


def test_surviving_replies_excludes_full_columns() -> None:
    board = [0] * 12

    for row in range(3):
        board[row * 4] = 1 if row % 2 == 0 else 2

    assert 0 not in game_theory.surviving_replies(
        board,
        defender_mark=1,
        columns=4,
        rows=3,
        inarow=3,
    )


def test_tactical_diagnostics_do_not_mutate_the_input_board() -> None:
    board = [0] * 42
    board[35:38] = [1, 1, 1]
    original_board = board.copy()

    game_theory.winning_columns(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )
    game_theory.fork_moves(
        board,
        mark=1,
        columns=7,
        rows=6,
        inarow=4,
    )
    game_theory.surviving_replies(
        board,
        defender_mark=2,
        columns=7,
        rows=6,
        inarow=4,
    )

    assert board == original_board
