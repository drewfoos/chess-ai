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


# Piece type indices (matching FEN chars)
_PIECE_TO_PLANE = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}


def _parse_fen(fen: str):
    """Parse FEN string into components."""
    parts = fen.split()
    board_str = parts[0]
    color = parts[1]
    castling = parts[2] if len(parts) > 2 else '-'
    ep = parts[3] if len(parts) > 3 else '-'
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    return board_str, color, castling, ep, halfmove, fullmove


def _board_to_pieces(board_str: str):
    """Parse FEN board string into a dict of {square: (color, piece_type)}."""
    pieces = {}
    rank = 7  # FEN starts from rank 8 (index 7)
    file = 0
    for ch in board_str:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = make_square(file, rank)
            is_white = ch.isupper()
            pieces[sq] = (is_white, _PIECE_TO_PLANE[ch])
            file += 1
    return pieces


def encode_position(fen: str) -> np.ndarray:
    """Encode a chess position as a 112×8×8 float32 tensor.

    .. deprecated::
        This FEN-only variant has no move history, so it fills all 8 time steps
        with duplicates of the current position — the distribution the network
        does NOT see during self-play / inference (where the C++ encoder always
        has real history). Using this on real training positions produces
        out-of-distribution inputs and will corrupt training.

        Use ``encode_board(python_chess.Board)`` with an intact move_stack, or
        call ``chess_mcts.encode_packed(start_fen, uci_moves)`` — both produce
        identical tensors (see ``test_encoder_parity_*``). Kept only for
        isolated unit tests that encode a lone FEN with no history context.

    The board is always oriented from the side-to-move's perspective.
    When Black is to move, the board is flipped vertically.

    Plane layout (112 total):
        0-12:   Time step 0 (current position) — 13 planes
        13-25:  Time step 1 (repeated current) — 13 planes
        ...     (8 time steps total, all identical without history)
        104:    Color to move (1 = white, 0 = black)
        105:    Total move count (normalized: fullmove / 200)
        106:    Our kingside castling
        107:    Our queenside castling
        108:    Opponent kingside castling
        109:    Opponent queenside castling
        110:    Halfmove clock (normalized: halfmove / 100)
        111:    All-ones bias plane

    Each time step (13 planes):
        0-5:    Our pieces (pawn, knight, bishop, rook, queen, king)
        6-11:   Opponent pieces (pawn, knight, bishop, rook, queen, king)
        12:     Repetition count (0 for now — no history tracking)
    """
    board_str, color, castling, ep, halfmove, fullmove = _parse_fen(fen)
    pieces = _board_to_pieces(board_str)

    is_white = (color == 'w')
    planes = np.zeros((112, 8, 8), dtype=np.float32)

    # Fill piece planes for one time step
    step_planes = np.zeros((13, 8, 8), dtype=np.float32)

    for sq, (piece_is_white, piece_type) in pieces.items():
        # Flip square if black to move
        actual_sq = sq if is_white else mirror_move(sq)
        r, f = rank_of(actual_sq), file_of(actual_sq)

        if piece_is_white == is_white:
            # Our piece → planes 0-5
            step_planes[piece_type, r, f] = 1.0
        else:
            # Opponent piece → planes 6-11
            step_planes[6 + piece_type, r, f] = 1.0

    # Plane 12: repetition count (0 for now)

    # Copy to all 8 time steps (no history available)
    for t in range(8):
        planes[t * 13:(t + 1) * 13] = step_planes

    # Constant planes (104-111)
    # 104: color to move
    if is_white:
        planes[104] = 1.0

    # 105: total move count (normalized)
    planes[105] = fullmove / 200.0

    # 106-109: castling rights (from side-to-move perspective)
    if is_white:
        if 'K' in castling: planes[106] = 1.0
        if 'Q' in castling: planes[107] = 1.0
        if 'k' in castling: planes[108] = 1.0
        if 'q' in castling: planes[109] = 1.0
    else:
        if 'k' in castling: planes[106] = 1.0
        if 'q' in castling: planes[107] = 1.0
        if 'K' in castling: planes[108] = 1.0
        if 'Q' in castling: planes[109] = 1.0

    # 110: halfmove clock (normalized)
    planes[110] = halfmove / 100.0

    # 111: all-ones bias
    planes[111] = 1.0

    return planes


def encode_board(board) -> np.ndarray:
    """Encode a python-chess Board as a 112x8x8 float32 tensor with position history.

    Unlike encode_position(fen), this uses the board's move stack to fill
    8 time steps with actual history positions instead of repeating the current one.

    Args:
        board: A python-chess Board object with move history.

    Returns:
        np.ndarray of shape (112, 8, 8), dtype float32.
    """
    import chess

    is_white = (board.turn == chess.WHITE)
    planes = np.zeros((112, 8, 8), dtype=np.float32)

    # We need to encode 8 time steps: current position + 7 history positions.
    # Walk backwards through the move stack to reconstruct earlier positions.
    # We work on a copy so we don't mutate the original board.
    history_board = board.copy()
    for t in range(8):
        step_planes = np.zeros((13, 8, 8), dtype=np.float32)

        # Encode pieces on the board
        for sq in range(64):
            piece = history_board.piece_at(sq)
            if piece is None:
                continue
            piece_is_white = (piece.color == chess.WHITE)
            piece_type = piece.piece_type - 1  # chess.PAWN=1..KING=6 -> 0..5

            # Flip square if side-to-move is Black (always orient from STM perspective)
            actual_sq = sq if is_white else mirror_move(sq)
            r, f = rank_of(actual_sq), file_of(actual_sq)

            if piece_is_white == is_white:
                step_planes[piece_type, r, f] = 1.0  # Our piece
            else:
                step_planes[6 + piece_type, r, f] = 1.0  # Opponent piece

        # Repetition plane (plane 12): 1 if position has occurred at least twice
        if history_board.is_repetition(2):
            step_planes[12] = 1.0

        planes[t * 13:(t + 1) * 13] = step_planes

        # Step back one move for the next time step
        if history_board.move_stack:
            history_board.pop()
        # If no more history, remaining time steps will repeat this earliest position

    # Constant planes (104-111)
    if is_white:
        planes[104] = 1.0  # Color to move

    # Total move count (normalized)
    planes[105] = board.fullmove_number / 200.0

    # Castling rights (from side-to-move perspective)
    if is_white:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[106] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[107] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[108] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[109] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[106] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[107] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[108] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[109] = 1.0

    # Halfmove clock (normalized)
    planes[110] = board.halfmove_clock / 100.0

    # All-ones bias
    planes[111] = 1.0

    return planes
