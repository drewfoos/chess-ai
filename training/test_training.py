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
