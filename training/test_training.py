import pytest
from training.config import NetworkConfig
from training.encoder import (
    move_to_index,
    index_to_move,
    POLICY_SIZE,
    mirror_move,
)


def test_network_config_defaults():
    cfg = NetworkConfig()
    assert cfg.num_blocks == 10
    assert cfg.num_filters == 128
    assert cfg.se_ratio == 4
    assert cfg.input_planes == 112
    assert cfg.policy_size == 1858
    assert cfg.value_size == 3


def test_network_config_custom():
    cfg = NetworkConfig(num_blocks=20, num_filters=256)
    assert cfg.num_blocks == 20
    assert cfg.num_filters == 256
    assert cfg.input_planes == 112  # Unchanged default


def test_policy_size():
    assert POLICY_SIZE == 1858


def test_move_encoding_roundtrip_e2e4():
    # e2e4: from=E2(12), to=E4(28), promo=None
    # From white's perspective, this is a pawn double push
    # E2 is file=4, rank=1 → square index 12
    # E4 is file=4, rank=3 → square index 28
    # Direction: North (index 0), distance 2
    idx = move_to_index(12, 28, None)
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 12
    assert to_sq == 28
    assert promo is None


def test_move_encoding_knight():
    # Ng1-f3: from=G1(6), to=F3(21)
    # Knight move: delta = (21-6) = 15 → (-1, +2) in (file, rank)
    idx = move_to_index(6, 21, None)
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 6
    assert to_sq == 21
    assert promo is None


def test_move_encoding_queen_promotion():
    # e7e8=Q: from=E7(52), to=E8(60), promo=queen
    # Queen promotion uses the normal queen-move encoding (N, distance 1)
    idx = move_to_index(52, 60, None)  # No promo flag for queen promo
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 52
    assert to_sq == 60


def test_move_encoding_underpromotion():
    # e7e8=N: from=E7(52), to=E8(60), promo='n'
    idx = move_to_index(52, 60, 'n')
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 52
    assert to_sq == 60
    assert promo == 'n'


def test_move_encoding_capture_underpromotion():
    # e7d8=R: from=E7(52), to=D8(59), promo='r'
    idx = move_to_index(52, 59, 'r')
    assert idx is not None
    from_sq, to_sq, promo = index_to_move(idx)
    assert from_sq == 52
    assert to_sq == 59
    assert promo == 'r'


def test_all_indices_unique():
    # Collect all valid move indices and ensure no duplicates
    seen = set()
    for idx in range(POLICY_SIZE):
        from_sq, to_sq, promo = index_to_move(idx)
        key = (from_sq, to_sq, promo)
        assert key not in seen, f"Duplicate move at index {idx}: {key}"
        seen.add(key)


def test_mirror_move():
    # E2(12) mirrored = E7(52) — flip rank: rank 1 → rank 6
    assert mirror_move(12) == 52
    # A1(0) mirrored = A8(56)
    assert mirror_move(0) == 56
    # H8(63) mirrored = H1(7)
    assert mirror_move(63) == 7


import numpy as np
from training.encoder import encode_position


def test_encode_starting_position_shape():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert planes.shape == (112, 8, 8)
    assert planes.dtype == np.float32


def test_encode_starting_position_white_pawns():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # White to move, so "our" pieces = white
    # Plane 0 = our pawns. White pawns on rank 1 (index 1 in 0-indexed)
    # In the tensor: planes[0] should have 1s on rank 1 (row index 1)
    pawn_plane = planes[0]
    assert pawn_plane.sum() == 8  # 8 white pawns
    # All on rank 1
    for file in range(8):
        assert pawn_plane[1, file] == 1.0


def test_encode_starting_position_opponent_pawns():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Plane 6 = opponent pawns (black). Black pawns on rank 6
    opp_pawn_plane = planes[6]
    assert opp_pawn_plane.sum() == 8
    for file in range(8):
        assert opp_pawn_plane[6, file] == 1.0


def test_encode_black_to_move_flips():
    # Same position but black to move — board should be flipped
    planes = encode_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # Black to move: "our" pieces = black, board flipped vertically
    # Black pawns were on rank 6, after flip they're on rank 1
    our_pawn_plane = planes[0]
    assert our_pawn_plane.sum() == 8
    for file in range(8):
        assert our_pawn_plane[1, file] == 1.0


def test_encode_castling_planes():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Constant planes start at index 104
    # Plane 106 = our kingside castling (K for white) → all 1s
    assert planes[106].sum() == 64  # All 1s
    # Plane 107 = our queenside castling (Q for white) → all 1s
    assert planes[107].sum() == 64


def test_encode_no_castling():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
    # No castling rights → planes 106-109 all zeros
    for i in range(106, 110):
        assert planes[i].sum() == 0


def test_encode_color_plane():
    # White to move: color plane (104) = all 1s
    planes_w = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert planes_w[104].sum() == 64

    # Black to move: color plane (104) = all 0s
    planes_b = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert planes_b[104].sum() == 0


def test_encode_bias_plane():
    planes = encode_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Plane 111 = all-ones bias
    assert planes[111].sum() == 64
