"""Position and move encoding for the chess neural network.

Move encoding follows the Leela Chess Zero / AlphaZero convention:
- 73 move types per source square (56 queen-like + 8 knight + 9 underpromotions)
- Board always oriented from side-to-move perspective (rank 0 = our back rank)
- Total valid moves: 1858 out of 4672 (64 × 73)
"""

import numpy as np

POLICY_SIZE = 1858

# Square utilities (matching C++ types.h: A1=0, B1=1, ..., H8=63)
def rank_of(sq: int) -> int:
    return sq >> 3

def file_of(sq: int) -> int:
    return sq & 7

def make_square(file: int, rank: int) -> int:
    return rank * 8 + file

def mirror_move(sq: int) -> int:
    """Flip square vertically (rank 0↔7, 1↔6, etc.)."""
    return make_square(file_of(sq), 7 - rank_of(sq))


# Direction vectors: (file_delta, rank_delta)
_QUEEN_DIRS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Underpromotion directions: (file_delta, rank_delta) for pawn on rank 6 → rank 7
_PROMO_DIRS = [
    (-1, 1),  # capture left
    (0, 1),   # forward
    (1, 1),   # capture right
]

_PROMO_PIECES = ['n', 'b', 'r']  # knight, bishop, rook (queen uses normal encoding)


def _build_tables():
    """Build bidirectional mapping between (from, to, promo) and flat policy index."""
    move_to_idx = {}  # (from_sq, to_sq, promo) → index
    idx_to_move = []  # index → (from_sq, to_sq, promo)

    idx = 0

    for from_sq in range(64):
        f, r = file_of(from_sq), rank_of(from_sq)

        # Queen-like moves: 8 directions × 7 distances
        for dir_idx, (df, dr) in enumerate(_QUEEN_DIRS):
            for dist in range(1, 8):
                nf, nr = f + df * dist, r + dr * dist
                if 0 <= nf < 8 and 0 <= nr < 8:
                    to_sq = make_square(nf, nr)
                    move_to_idx[(from_sq, to_sq, None)] = idx
                    idx_to_move.append((from_sq, to_sq, None))
                    idx += 1

        # Knight moves
        for df, dr in _KNIGHT_DELTAS:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                to_sq = make_square(nf, nr)
                move_to_idx[(from_sq, to_sq, None)] = idx
                idx_to_move.append((from_sq, to_sq, None))
                idx += 1

        # Underpromotions: only from rank 6 (7th rank in 0-indexed)
        if r == 6:
            for dir_idx, (df, dr) in enumerate(_PROMO_DIRS):
                nf, nr = f + df, r + dr
                if 0 <= nf < 8 and nr == 7:
                    to_sq = make_square(nf, nr)
                    for piece in _PROMO_PIECES:
                        move_to_idx[(from_sq, to_sq, piece)] = idx
                        idx_to_move.append((from_sq, to_sq, piece))
                        idx += 1

    return move_to_idx, idx_to_move


_MOVE_TO_IDX, _IDX_TO_MOVE = _build_tables()

assert len(_IDX_TO_MOVE) == POLICY_SIZE, f"Expected {POLICY_SIZE} moves, got {len(_IDX_TO_MOVE)}"


def move_to_index(from_sq: int, to_sq: int, promo: str | None) -> int | None:
    """Convert a chess move to a policy index.

    Args:
        from_sq: Source square (0=A1, 63=H8), from side-to-move perspective.
        to_sq: Destination square, from side-to-move perspective.
        promo: Underpromotion piece ('n', 'b', 'r') or None for queen/non-promotion.

    Returns:
        Policy index in [0, 1857], or None if move is not in the encoding.
    """
    return _MOVE_TO_IDX.get((from_sq, to_sq, promo))


def index_to_move(idx: int) -> tuple[int, int, str | None]:
    """Convert a policy index back to a chess move.

    Returns:
        (from_sq, to_sq, promo) tuple.
    """
    return _IDX_TO_MOVE[idx]
