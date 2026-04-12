"""Generate reference data for C++ encoder/policy_map cross-validation.

Run: python tests/generate_test_data.py

This script outputs plane sums and specific cell values for known FEN strings,
which can be compared against C++ encoder output to verify they match exactly.
"""
from training.encoder import encode_position, move_to_index

fens = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 50 100",
]

print("=== Encoder Reference Data ===\n")
for fen in fens:
    planes = encode_position(fen)
    print(f"FEN: {fen}")
    # Print sums for piece planes (0-12) and constant planes (104-111)
    for p in list(range(13)) + list(range(104, 112)):
        s = float(planes[p].sum())
        if s != 0.0:
            print(f"  Plane {p:3d}: sum = {s:.1f}")
    print()

print("=== Policy Index Reference Data ===\n")
moves = [
    (12, 28, None, "e2e4"),
    (6, 21, None, "Ng1f3"),
    (52, 60, None, "e7e8Q"),
    (52, 60, 'n', "e7e8N"),
    (52, 59, 'r', "e7d8R"),
    (8, 16, None, "a2a3"),
    (1, 18, None, "b1c3"),
    (3, 39, None, "d1h5"),
]

for from_sq, to_sq, promo, name in moves:
    idx = move_to_index(from_sq, to_sq, promo)
    print(f"  {name:8s}: move_to_index({from_sq}, {to_sq}, {promo!r}) = {idx}")
