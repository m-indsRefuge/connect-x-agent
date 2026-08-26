from connect_x_agent.game_theory import (
    fork_moves,
    landing_cells,
    surviving_replies,
    threats,
    viable_windows,
    winning_columns,
)

from .records import PositionFeatures


def position_features(
    board: list[int],
    columns: int,
    rows: int,
    inarow: int,
) -> PositionFeatures:
    """Return immutable tactical structure for both marks on ``board``.

    Surviving-reply fields are counterfactual: each is calculated as if the
    named mark moves next, regardless of the actual game turn.
    """
    p1_threats = threats(board, 1, columns, rows, inarow)
    p2_threats = threats(board, 2, columns, rows, inarow)

    return PositionFeatures(
        legal_columns=tuple(landing_cells(board, columns, rows)),
        p1_viable_windows=len(viable_windows(board, 1, columns, rows, inarow)),
        p2_viable_windows=len(viable_windows(board, 2, columns, rows, inarow)),
        p1_playable_threats=sum(threat.playable for threat in p1_threats),
        p2_playable_threats=sum(threat.playable for threat in p2_threats),
        p1_latent_threats=sum(not threat.playable for threat in p1_threats),
        p2_latent_threats=sum(not threat.playable for threat in p2_threats),
        p1_winning_columns=winning_columns(board, 1, columns, rows, inarow),
        p2_winning_columns=winning_columns(board, 2, columns, rows, inarow),
        p1_fork_moves=tuple(fork_moves(board, 1, columns, rows, inarow)),
        p2_fork_moves=tuple(fork_moves(board, 2, columns, rows, inarow)),
        p1_surviving_replies=surviving_replies(board, 1, columns, rows, inarow),
        p2_surviving_replies=surviving_replies(board, 2, columns, rows, inarow),
    )
