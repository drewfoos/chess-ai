# Neural Network & Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the neural network architecture (residual tower + SE layers + policy/value heads) and training pipeline in Python/PyTorch, with TorchScript export for C++ inference.

**Architecture:** A residual CNN with Squeeze-and-Excitation layers, classical policy head (1858 move logits), and WDL value head (win/draw/loss probabilities). Input is 112 planes × 8×8 encoding the board state from the side-to-move's perspective. Training uses policy cross-entropy + value cross-entropy + L2 regularization. The network exports to TorchScript for Plan 5 (C++ inference).

**Tech Stack:** Python 3.11, PyTorch 2.11.0+cu126, numpy, pytest

---

## File Structure

```
training/
├── config.py            NetworkConfig dataclass (blocks, filters, SE ratio)
├── encoder.py           Position → 112×8×8 tensor, Move ↔ policy index (1858)
├── model.py             ChessNetwork: residual tower + SE + policy/value heads
├── dataset.py           ChessDataset: loads .npz training data
├── train.py             Training loop CLI: policy CE + value CE + L2 reg
├── export.py            TorchScript export + verification
├── generate_data.py     Synthetic data generator (testing pipeline without self-play)
└── test_training.py     All pytest tests
```

**Interfaces with existing code:**
- Move encoding uses the same square/move conventions as `src/core/types.h` (A1=0, H8=63, file-major ordering)
- TorchScript `.pt` output consumed by Plan 5 (C++ inference via `src/neural/network.h`)
- Training data `.npz` format produced by Plan 4 (self-play pipeline)
- Policy output shape (1858) maps to Leela-style move encoding (64 squares × 73 move types, filtered to valid moves)

---

### Task 1: Project setup and NetworkConfig

**Files:**
- Create: `training/config.py`
- Create: `training/__init__.py`
- Create: `training/test_training.py` (initial scaffold)

- [ ] **Step 1: Create training/__init__.py**

Create an empty `training/__init__.py` to make the directory a Python package:

```python
```

(Empty file.)

- [ ] **Step 2: Create training/config.py with NetworkConfig**

```python
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    num_blocks: int = 10
    num_filters: int = 128
    se_ratio: int = 4
    input_planes: int = 112
    policy_size: int = 1858
    value_size: int = 3  # WDL: win, draw, loss
    policy_conv_filters: int = 80
    value_conv_filters: int = 32
    value_fc_size: int = 128
```

- [ ] **Step 3: Create training/test_training.py with first test**

```python
import pytest
from training.config import NetworkConfig


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
```

- [ ] **Step 4: Run tests**

Run from project root: `cd E:\dev\chess-ai && python -m pytest training/test_training.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/__init__.py training/config.py training/test_training.py
git commit -m "feat(training): add NetworkConfig dataclass and project scaffold"
```

---

### Task 2: Move encoding — policy index ↔ chess move

**Files:**
- Create: `training/encoder.py`
- Modify: `training/test_training.py`

The policy head outputs 1858 logits. Each corresponds to a valid chess move encoded as (from_square, move_type). The 73 move types per square are:

- **Queen-like moves (56):** 8 directions × 7 distances. Directions: N, NE, E, SE, S, SW, W, NW (indices 0–7). Distance 1–7 within each direction.
- **Knight moves (8):** 8 L-shaped jumps (indices 56–63).
- **Underpromotions (9):** 3 directions (forward-left, forward, forward-right) × 3 piece types (knight, bishop, rook). Indices 64–72. Queen promotion uses the normal queen-move encoding.

Board is always oriented from the side-to-move's perspective: rank 0 = our back rank, rank 7 = opponent's back rank. For Black, the board is flipped vertically.

- [ ] **Step 1: Write move encoding tests**

Append to `training/test_training.py`:

```python
from training.encoder import (
    move_to_index,
    index_to_move,
    POLICY_SIZE,
    mirror_move,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py -v -k "move_encoding or policy_size or mirror_move or all_indices"`
Expected: FAIL — `encoder.py` doesn't exist.

- [ ] **Step 3: Implement move encoding in encoder.py**

Create `training/encoder.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/test_training.py -v -k "move_encoding or policy_size or mirror_move or all_indices"`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/encoder.py training/test_training.py
git commit -m "feat(training): add move encoding — 1858-dim policy index ↔ chess move mapping"
```

---

### Task 3: Position encoding — FEN to 112×8×8 tensor

**Files:**
- Modify: `training/encoder.py`
- Modify: `training/test_training.py`

The input to the neural network is a 112×8×8 float tensor:
- **104 planes (13 per time step × 8 time steps):** 6 planes for our pieces, 6 for opponent's, 1 for repetition. For now (no history), repeat the current position for all 8 time steps.
- **8 constant planes:** color to move (all 1s if white, all 0s), total move count (normalized), 4 castling rights, halfmove clock, all-ones bias.

Board is always from side-to-move perspective. For Black, flip vertically.

- [ ] **Step 1: Write position encoding tests**

Append to `training/test_training.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py -v -k "encode_"`
Expected: FAIL — `encode_position` not yet defined.

- [ ] **Step 3: Implement position encoding**

Append to `training/encoder.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/test_training.py -v -k "encode_"`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/encoder.py training/test_training.py
git commit -m "feat(training): add position encoding — FEN to 112×8×8 tensor with board flipping"
```

---

### Task 4: Network architecture — residual tower + SE + policy/value heads

**Files:**
- Create: `training/model.py`
- Modify: `training/test_training.py`

- [ ] **Step 1: Write model architecture tests**

Append to `training/test_training.py`:

```python
import torch
from training.model import ChessNetwork
from training.config import NetworkConfig


def test_model_output_shapes():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)  # Small for testing
    model = ChessNetwork(cfg)
    x = torch.randn(4, 112, 8, 8)  # Batch of 4
    policy, value = model(x)
    assert policy.shape == (4, 1858)
    assert value.shape == (4, 3)


def test_model_policy_logits():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    x = torch.randn(1, 112, 8, 8)
    policy, _ = model(x)
    # Policy should be raw logits (not softmaxed) — can be any real number
    assert policy.dtype == torch.float32


def test_model_value_probabilities():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    x = torch.randn(1, 112, 8, 8)
    _, value = model(x)
    # Value head should output probabilities that sum to 1 (WDL softmax)
    assert torch.allclose(value.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
    # All values non-negative
    assert (value >= 0).all()


def test_model_default_config():
    cfg = NetworkConfig()  # 10 blocks, 128 filters
    model = ChessNetwork(cfg)
    # Count parameters — should be roughly 1.5M
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1_000_000  # At least 1M
    assert total_params < 3_000_000  # Less than 3M


def test_model_batch_independence():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    model.eval()
    x = torch.randn(2, 112, 8, 8)
    policy_batch, value_batch = model(x)
    policy_0, value_0 = model(x[0:1])
    policy_1, value_1 = model(x[1:2])
    assert torch.allclose(policy_batch[0], policy_0[0], atol=1e-5)
    assert torch.allclose(value_batch[0], value_0[0], atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py -v -k "model_"`
Expected: FAIL — `model.py` doesn't exist.

- [ ] **Step 3: Implement ChessNetwork**

Create `training/model.py`:

```python
"""Chess neural network: residual tower with SE layers, policy and value heads.

Architecture (matching Leela Chess Zero):
    Input: 112 × 8 × 8
    → Initial Conv(3×3, filters) + BN + ReLU
    → N × ResidualBlock(3×3 conv + BN + ReLU + 3×3 conv + BN + SE + skip + ReLU)
    → Policy Head: Conv(1×1, 80) + BN + ReLU + Flatten + FC(5120 → 1858)
    → Value Head: Conv(1×1, 32) + BN + ReLU + Flatten + FC(2048 → 128) + ReLU + FC(128 → 3) + Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.config import NetworkConfig


class SqueezeExcitation(nn.Module):
    """Leela-style SE: produces both multiplicative weights and additive biases."""

    def __init__(self, channels: int, ratio: int):
        super().__init__()
        mid = channels // ratio
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, 2 * channels)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Global average pooling
        z = x.mean(dim=(2, 3))  # (B, C)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)  # (B, 2C)
        # Split into weights and biases
        weights, biases = z.split(self.channels, dim=1)
        weights = torch.sigmoid(weights).view(b, c, 1, 1)
        biases = biases.view(b, c, 1, 1)
        return weights * x + biases


class ResidualBlock(nn.Module):
    """Residual block with SE layer."""

    def __init__(self, channels: int, se_ratio: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class ChessNetwork(nn.Module):
    """Full chess neural network with residual tower, policy head, and WDL value head."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        nf = config.num_filters

        # Initial convolution
        self.input_conv = nn.Conv2d(config.input_planes, nf, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(nf)

        # Residual tower
        self.blocks = nn.ModuleList([
            ResidualBlock(nf, config.se_ratio)
            for _ in range(config.num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(nf, config.policy_conv_filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(config.policy_conv_filters)
        self.policy_fc = nn.Linear(config.policy_conv_filters * 64, config.policy_size)

        # Value head (WDL)
        self.value_conv = nn.Conv2d(nf, config.value_conv_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_conv_filters)
        self.value_fc1 = nn.Linear(config.value_conv_filters * 64, config.value_fc_size)
        self.value_fc2 = nn.Linear(config.value_fc_size, config.value_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Initial conv
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)  # Raw logits

        # Value head (WDL)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = F.softmax(self.value_fc2(v), dim=1)  # WDL probabilities

        return p, v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/test_training.py -v -k "model_"`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/model.py training/test_training.py
git commit -m "feat(training): add ChessNetwork with residual tower, SE layers, policy/value heads"
```

---

### Task 5: Synthetic data generator and dataset

**Files:**
- Create: `training/generate_data.py`
- Create: `training/dataset.py`
- Modify: `training/test_training.py`

Since self-play data doesn't exist yet (Plan 4), we need synthetic data to test the training pipeline. The generator creates random positions with random policy targets and random WDL outcomes.

Training data format (`.npz`):
- `planes`: float32 array of shape `[N, 112, 8, 8]`
- `policies`: float32 array of shape `[N, 1858]` — normalized probability distribution
- `values`: float32 array of shape `[N, 3]` — WDL target (one-hot or soft)

- [ ] **Step 1: Write dataset and generation tests**

Append to `training/test_training.py`:

```python
import tempfile
import os
from training.generate_data import generate_synthetic_data
from training.dataset import ChessDataset


def test_generate_synthetic_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.npz")
        generate_synthetic_data(path, num_positions=50)

        data = np.load(path)
        assert data['planes'].shape == (50, 112, 8, 8)
        assert data['policies'].shape == (50, 1858)
        assert data['values'].shape == (50, 3)

        # Policies should be valid probability distributions
        policy_sums = data['policies'].sum(axis=1)
        np.testing.assert_allclose(policy_sums, 1.0, atol=1e-5)

        # Values should be valid WDL distributions
        value_sums = data['values'].sum(axis=1)
        np.testing.assert_allclose(value_sums, 1.0, atol=1e-5)


def test_chess_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.npz")
        generate_synthetic_data(path, num_positions=20)

        dataset = ChessDataset([path])
        assert len(dataset) == 20

        planes, policy, value = dataset[0]
        assert planes.shape == (112, 8, 8)
        assert policy.shape == (1858,)
        assert value.shape == (3,)
        assert isinstance(planes, torch.Tensor)


def test_chess_dataset_multiple_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "data1.npz")
        path2 = os.path.join(tmpdir, "data2.npz")
        generate_synthetic_data(path1, num_positions=10)
        generate_synthetic_data(path2, num_positions=15)

        dataset = ChessDataset([path1, path2])
        assert len(dataset) == 25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py -v -k "generate_ or chess_dataset"`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement generate_data.py**

Create `training/generate_data.py`:

```python
"""Generate synthetic training data for testing the training pipeline.

This produces random positions with random policy and value targets.
Real training data comes from self-play (Plan 4).
"""

import argparse
import numpy as np
from training.encoder import encode_position, POLICY_SIZE


# Standard FENs for generating diverse synthetic positions
_SEED_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pp2pppp/5n2/2pp4/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq d6 0 3",
]


def generate_synthetic_data(output_path: str, num_positions: int = 1000):
    """Generate synthetic training data and save as .npz file.

    Each sample has:
    - planes: encoded position (112, 8, 8)
    - policy: random probability distribution over 1858 moves
    - value: random WDL target (one of [1,0,0], [0,1,0], [0,0,1])
    """
    rng = np.random.default_rng(42)

    planes_list = []
    policies_list = []
    values_list = []

    for i in range(num_positions):
        # Cycle through seed FENs
        fen = _SEED_FENS[i % len(_SEED_FENS)]
        planes = encode_position(fen)
        planes_list.append(planes)

        # Random policy: Dirichlet-distributed over a random subset of moves
        policy = rng.dirichlet(np.ones(POLICY_SIZE) * 0.03)
        policies_list.append(policy.astype(np.float32))

        # Random WDL outcome
        outcome = rng.choice(3)
        value = np.zeros(3, dtype=np.float32)
        value[outcome] = 1.0
        values_list.append(value)

    np.savez(
        output_path,
        planes=np.array(planes_list, dtype=np.float32),
        policies=np.array(policies_list, dtype=np.float32),
        values=np.array(values_list, dtype=np.float32),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--output', type=str, default='data/synthetic.npz')
    parser.add_argument('--num-positions', type=int, default=1000)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    generate_synthetic_data(args.output, args.num_positions)
    print(f"Generated {args.num_positions} positions → {args.output}")
```

- [ ] **Step 4: Implement dataset.py**

Create `training/dataset.py`:

```python
"""Chess training dataset: loads .npz files into a PyTorch Dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """Dataset that loads chess training positions from .npz files.

    Each .npz file contains:
        planes:   float32[N, 112, 8, 8]  — encoded board position
        policies: float32[N, 1858]        — MCTS visit distribution target
        values:   float32[N, 3]           — WDL target (win, draw, loss)
    """

    def __init__(self, npz_paths: list[str]):
        planes_list = []
        policies_list = []
        values_list = []

        for path in npz_paths:
            data = np.load(path)
            planes_list.append(data['planes'])
            policies_list.append(data['policies'])
            values_list.append(data['values'])

        self.planes = torch.from_numpy(np.concatenate(planes_list, axis=0))
        self.policies = torch.from_numpy(np.concatenate(policies_list, axis=0))
        self.values = torch.from_numpy(np.concatenate(values_list, axis=0))

    def __len__(self) -> int:
        return len(self.planes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.planes[idx], self.policies[idx], self.values[idx]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest training/test_training.py -v -k "generate_ or chess_dataset"`
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add training/generate_data.py training/dataset.py training/test_training.py
git commit -m "feat(training): add synthetic data generator and ChessDataset for .npz files"
```

---

### Task 6: Training loop

**Files:**
- Create: `training/train.py`
- Modify: `training/test_training.py`

The training loop:
1. Load data files into ChessDataset
2. Create DataLoader with shuffling and batching
3. For each batch: forward pass → compute loss → backward → optimizer step
4. Loss = policy_CE + value_CE + L2_reg (weight decay via optimizer)
5. Save checkpoints periodically
6. Log metrics to console

- [ ] **Step 1: Write training loop test**

Append to `training/test_training.py`:

```python
from training.train import train_step, create_optimizer


def test_train_step_reduces_loss():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    optimizer = create_optimizer(model)

    # Create a batch of synthetic data
    batch_size = 8
    planes = torch.randn(batch_size, 112, 8, 8)
    policies = torch.softmax(torch.randn(batch_size, 1858), dim=1)
    values = torch.zeros(batch_size, 3)
    values[:, 0] = 1.0  # All wins

    # Run multiple steps and check loss decreases
    losses = []
    for _ in range(10):
        loss = train_step(model, optimizer, planes, policies, values)
        losses.append(loss)

    # Loss should decrease over 10 steps
    assert losses[-1] < losses[0]


def test_train_step_loss_components():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)
    optimizer = create_optimizer(model)

    planes = torch.randn(4, 112, 8, 8)
    policies = torch.softmax(torch.randn(4, 1858), dim=1)
    values = torch.zeros(4, 3)
    values[:, 1] = 1.0  # All draws

    loss = train_step(model, optimizer, planes, policies, values)
    # Loss should be a positive number
    assert loss > 0
    assert not np.isnan(loss)
    assert not np.isinf(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py -v -k "train_step"`
Expected: FAIL — `train.py` doesn't exist.

- [ ] **Step 3: Implement train.py**

Create `training/train.py`:

```python
"""Training loop for the chess neural network.

Loss function:
    L = L_policy + L_value
    L_policy = cross_entropy(policy_target, policy_output)
    L_value  = cross_entropy(wdl_target, wdl_output)
    L2 regularization is handled by the optimizer (weight_decay).

Usage:
    python -m training.train --data data/synthetic.npz --epochs 10
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.config import NetworkConfig
from training.model import ChessNetwork
from training.dataset import ChessDataset


def compute_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> torch.Tensor:
    """Compute combined policy + value loss.

    Args:
        policy_logits: Raw logits from policy head (B, 1858)
        value_pred: WDL probabilities from value head (B, 3)
        policy_target: MCTS visit distribution target (B, 1858)
        value_target: WDL target (B, 3)

    Returns:
        Scalar loss tensor.
    """
    # Policy loss: cross-entropy with soft targets
    # = -sum(target * log_softmax(logits))
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

    # Value loss: cross-entropy between WDL distributions
    # value_pred is already softmaxed, so use -sum(target * log(pred))
    value_loss = -(value_target * torch.log(value_pred + 1e-8)).sum(dim=1).mean()

    return policy_loss + value_loss


def create_optimizer(model: ChessNetwork, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Create Adam optimizer with L2 weight decay."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(
    model: ChessNetwork,
    optimizer: torch.optim.Optimizer,
    planes: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> float:
    """Execute one training step. Returns the loss value."""
    model.train()
    optimizer.zero_grad()

    policy_logits, value_pred = model(planes)
    loss = compute_loss(policy_logits, value_pred, policy_target, value_target)

    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    data_paths: list[str],
    config: NetworkConfig = NetworkConfig(),
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = 'checkpoints',
    device: str = 'auto',
):
    """Full training loop."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Config: {config.num_blocks} blocks, {config.num_filters} filters")

    dataset = ChessDataset(data_paths)
    print(f"Training data: {len(dataset)} positions from {len(data_paths)} files")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ChessNetwork(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        start = time.time()

        for planes, policies, values in loader:
            planes = planes.to(device)
            policies = policies.to(device)
            values = values.to(device)

            loss = train_step(model, optimizer, planes, policies, values)
            epoch_loss += loss
            num_batches += 1

        elapsed = time.time() - start
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': avg_loss,
            }, path)
            print(f"  Saved checkpoint: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train chess neural network')
    parser.add_argument('--data', nargs='+', required=True, help='Training .npz files')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--filters', type=int, default=128)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    config = NetworkConfig(num_blocks=args.blocks, num_filters=args.filters)
    train(
        data_paths=args.data,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/test_training.py -v -k "train_step"`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/train.py training/test_training.py
git commit -m "feat(training): add training loop with policy CE + value CE + L2 weight decay"
```

---

### Task 7: TorchScript export

**Files:**
- Create: `training/export.py`
- Modify: `training/test_training.py`

Export the trained PyTorch model to TorchScript format for C++ inference (Plan 5). TorchScript models can be loaded by LibTorch without any Python runtime.

- [ ] **Step 1: Write export tests**

Append to `training/test_training.py`:

```python
from training.export import export_torchscript


def test_export_torchscript():
    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        export_torchscript(model, path)

        # Load the exported model
        loaded = torch.jit.load(path)

        # Verify it produces the same output
        model.eval()
        x = torch.randn(1, 112, 8, 8)
        with torch.no_grad():
            orig_policy, orig_value = model(x)
            loaded_policy, loaded_value = loaded(x)

        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
        assert torch.allclose(orig_value, loaded_value, atol=1e-5)


def test_export_torchscript_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg).cuda()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model_gpu.pt")
        export_torchscript(model, path, device='cpu')  # Always export as CPU

        # Load on CPU and verify
        loaded = torch.jit.load(path)
        x = torch.randn(1, 112, 8, 8)
        model_cpu = model.cpu()
        model_cpu.eval()
        with torch.no_grad():
            orig_policy, orig_value = model_cpu(x)
            loaded_policy, loaded_value = loaded(x)

        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
        assert torch.allclose(orig_value, loaded_value, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/test_training.py -v -k "export_"`
Expected: FAIL — `export.py` doesn't exist.

- [ ] **Step 3: Implement export.py**

Create `training/export.py`:

```python
"""Export trained chess network to TorchScript for C++ inference.

Usage:
    python -m training.export --checkpoint checkpoints/model_epoch_10.pt --output model.pt
"""

import argparse

import torch

from training.config import NetworkConfig
from training.model import ChessNetwork


def export_torchscript(
    model: ChessNetwork,
    output_path: str,
    device: str = 'cpu',
):
    """Export a ChessNetwork to TorchScript via tracing.

    Args:
        model: Trained ChessNetwork instance.
        output_path: Path to save the .pt TorchScript file.
        device: Device to export on ('cpu' recommended for portability).
    """
    model = model.to(device)
    model.eval()

    # Trace with example input
    example = torch.randn(1, model.config.input_planes, 8, 8, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    traced.save(output_path)
    print(f"Exported TorchScript model to {output_path}")


def export_from_checkpoint(checkpoint_path: str, output_path: str):
    """Load a training checkpoint and export to TorchScript."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = ChessNetwork(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    export_torchscript(model, output_path)
    print(f"  Config: {config.num_blocks} blocks, {config.num_filters} filters")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export chess network to TorchScript')
    parser.add_argument('--checkpoint', required=True, help='Training checkpoint .pt file')
    parser.add_argument('--output', required=True, help='Output TorchScript .pt file')
    args = parser.parse_args()

    export_from_checkpoint(args.checkpoint, args.output)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/test_training.py -v -k "export_"`
Expected: 2 tests PASS (1 may skip if no CUDA).

- [ ] **Step 5: Commit**

```bash
git add training/export.py training/test_training.py
git commit -m "feat(training): add TorchScript export for C++ inference"
```

---

### Task 8: End-to-end smoke test and documentation

**Files:**
- Modify: `training/test_training.py`
- Modify: `docs/changelog.md`
- Modify: `docs/architecture.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write end-to-end test**

Append to `training/test_training.py`:

```python
from torch.utils.data import DataLoader


def test_end_to_end_pipeline():
    """Full pipeline: generate data → train → export → verify."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Generate synthetic data
        data_path = os.path.join(tmpdir, "train.npz")
        generate_synthetic_data(data_path, num_positions=64)

        # 2. Load into dataset
        dataset = ChessDataset([data_path])
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 3. Create model (tiny for speed)
        cfg = NetworkConfig(num_blocks=1, num_filters=16)
        model = ChessNetwork(cfg)
        optimizer = create_optimizer(model, lr=1e-3)

        # 4. Train for a few steps
        model.train()
        losses = []
        for planes, policies, values in loader:
            loss = train_step(model, optimizer, planes, policies, values)
            losses.append(loss)

        assert len(losses) > 0
        assert all(not np.isnan(l) for l in losses)

        # 5. Export to TorchScript
        export_path = os.path.join(tmpdir, "model.pt")
        export_torchscript(model, export_path)
        assert os.path.exists(export_path)

        # 6. Load exported model and verify
        loaded = torch.jit.load(export_path)
        model.eval()
        x = torch.randn(1, 112, 8, 8)
        with torch.no_grad():
            orig_p, orig_v = model(x)
            load_p, load_v = loaded(x)
        assert torch.allclose(orig_p, load_p, atol=1e-5)
        assert torch.allclose(orig_v, load_v, atol=1e-5)


def test_gpu_training():
    """Verify training works on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = NetworkConfig(num_blocks=2, num_filters=32)
    model = ChessNetwork(cfg).cuda()
    optimizer = create_optimizer(model)

    planes = torch.randn(4, 112, 8, 8).cuda()
    policies = torch.softmax(torch.randn(4, 1858), dim=1).cuda()
    values = torch.zeros(4, 3).cuda()
    values[:, 0] = 1.0

    loss = train_step(model, optimizer, planes, policies, values)
    assert loss > 0
    assert not np.isnan(loss)
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest training/test_training.py -v`
Expected: All tests PASS (~25 tests total).

- [ ] **Step 3: Run the full pipeline manually**

```bash
# Generate synthetic data
python -m training.generate_data --output data/synthetic.npz --num-positions 1000

# Train for a few epochs (tiny network for speed)
python -m training.train --data data/synthetic.npz --epochs 5 --blocks 2 --filters 32 --batch-size 64

# Export to TorchScript
python -m training.export --checkpoint checkpoints/model_epoch_5.pt --output checkpoints/model.pt
```

Verify each command runs without errors and outputs reasonable logs.

- [ ] **Step 4: Update docs/changelog.md**

Add under `[Unreleased]`:

```markdown
## [Unreleased]

### Added
- Neural network architecture: residual tower with SE layers, policy and WDL value heads (`training/model.py`)
- Position encoder: FEN → 112×8×8 tensor with board flipping for black (`training/encoder.py`)
- Move encoder: 1858-dim policy index ↔ chess move mapping (`training/encoder.py`)
- Training loop: policy cross-entropy + value cross-entropy + L2 regularization (`training/train.py`)
- TorchScript export for C++ inference (`training/export.py`)
- Synthetic data generator for pipeline testing (`training/generate_data.py`)
- Chess training dataset loader for .npz files (`training/dataset.py`)
- NetworkConfig dataclass for model hyperparameters (`training/config.py`)
- Training test suite: ~25 tests covering encoding, model, training, export, end-to-end
```

- [ ] **Step 5: Update CLAUDE.md**

Update "Current Milestone" to Phase 3, "Key Features" to include training pipeline, Plan Roadmap to mark Plan 2 complete and Plan 3 current, add Plan 3 to Documentation section.

- [ ] **Step 6: Update docs/architecture.md**

Under "### 3. Neural Network & Training (Python/PyTorch) — Plan 3", add:

```markdown
**Implementation status:** Complete. Network architecture, encoder, training loop, and TorchScript export all working. Uses synthetic data for validation; real self-play data comes in Plan 4.
```

- [ ] **Step 7: Commit**

```bash
git add training/test_training.py docs/changelog.md docs/architecture.md CLAUDE.md
git commit -m "feat(training): add end-to-end smoke test, update docs for Plan 3"
```

---

Plan complete and saved to `docs/superpowers/plans/2026-04-11-neural-network-training.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
