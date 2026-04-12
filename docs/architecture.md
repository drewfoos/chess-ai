# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Chess AI System                        │
│                                                             │
│  ┌───────────┐    ┌───────────┐    ┌─────────────────────┐  │
│  │  Engine    │    │ Self-Play │    │   Visualization     │  │
│  │  (C++17)  │◄──►│ Pipeline  │    │   Dashboard         │  │
│  │           │    │  (C++)    │    │   (React + Flask)    │  │
│  └─────┬─────┘    └─────┬─────┘    └──────────┬──────────┘  │
│        │                │                      │             │
│        │          training data (.gz)           │             │
│        │                │                      │             │
│        │          ┌─────▼─────┐          WebSocket           │
│        │          │ Training  │                │             │
│        │          │ Pipeline  │────────────────┘             │
│        │          │ (PyTorch) │                              │
│        │          └─────┬─────┘                              │
│        │                │                                    │
│        │          new weights                                │
│        │          (.pt / TorchScript)                        │
│        │                │                                    │
│        └────────────────┘                                    │
│         load weights for                                     │
│         inference + self-play                                │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Chess Engine Core (C++17) — Plan 1

The foundation. Provides board representation and legal move generation used by all other components.

```
src/core/
├── types.h          Square, Move, Bitboard, Color, PieceType, CastlingRight
├── bitboard.h/cpp   Bitboard constants, popcount, lsb, shifts
├── attacks.h/cpp    Precomputed attack tables (knight, king, pawn)
│                    Ray-based sliding attacks (bishop, rook, queen)
├── position.h/cpp   Board state, FEN I/O, make/unmake, check detection
└── movegen.h/cpp    Pseudo-legal generation → legality filter
```

**Data flow:**
```
FEN string → Position::set_fen() → Position object
Position → generate_legal_moves() → Move array
Position + Move → make_move() → updated Position + UndoInfo
Position + Move + UndoInfo → unmake_move() → restored Position
```

**Key design decisions:**
- Bitboard representation (not mailbox) for performance
- Ray-based sliding attacks initially (magic bitboards deferred)
- Position copy for legality check (simple, correct, optimize later)
- Perft-validated: correctness proven against known results

### 2. MCTS Search Engine (C++17) — Plan 2

Monte Carlo Tree Search guided by neural network policy and value outputs.

```
src/mcts/
├── node.h/cpp       Tree node: visit count, value, prior, children
└── search.h/cpp     MCTS loop: select → expand → evaluate → backprop
```

**Data flow:**
```
Position → MCTS Search (N iterations)
  Each iteration:
    Select: root → leaf via PUCT score
    Expand: leaf → create children for legal moves
    Evaluate: leaf position → neural network → (policy, value)
    Backprop: value → update all nodes on path to root
Result: visit distribution over root children → best move
```

**Key parameters:**
- PUCT constant: c_puct = 2.5
- FPU reduction: 0.25 (non-root), 0.44 (root)
- Virtual loss: 1 (for multi-threaded search)
- Dirichlet noise: alpha = 0.3, epsilon = 0.25 (root only)

**Implementation status:** Complete with stub evaluator. Uses `RandomEvaluator` (uniform policy + material value) until neural network is available (Plan 3/5).

### 3. Neural Network & Training (Python/PyTorch) — Plan 3

Residual CNN with policy and value heads, trained on self-play data.

```
training/
├── config.py        NetworkConfig dataclass for model hyperparameters
├── encoder.py       Position encoder (FEN → 112×8×8) and move index mapping
├── model.py         Network architecture (residual tower + SE + heads)
├── dataset.py       Training data loading from .npz files
├── train.py         Training loop (policy CE + value CE loss (WDL))
├── export.py        Export to TorchScript for C++ inference
├── generate_data.py Synthetic data generator for pipeline testing
└── test_training.py Training test suite (~32 tests)
```

**Network architecture:**
```
Input: 112 planes × 8×8 (position encoding)
  ↓
Initial Conv: 3×3, 128 filters, batch norm, ReLU
  ↓
Residual Tower: 10 blocks × (3×3 conv → BN → ReLU → 3×3 conv → BN → SE → skip → ReLU)
  ↓
  ├── Policy Head: 1×1 conv → BN → ReLU → flatten → FC → 1858 move logits → softmax
  │
  └── Value Head: 1×1 conv → BN → ReLU → flatten → FC 128 → ReLU → FC 3 → softmax (WDL)
```

**Starting config:** 10 blocks, 128 filters (~13M params). Scale to 20 blocks, 256 filters after pipeline validation.

**Implementation status:** Complete. Network architecture, encoder, training loop, and TorchScript export all working. Uses synthetic data for validation; real self-play data comes in Plan 4.

**Training data format (per position):**
```
input_planes:   float[112][8][8]    Board encoding (side-to-move perspective)
policy_target:  float[1858]         MCTS visit distribution (normalized)
value_target:   float[3]            Game result as WDL: [P(win), P(draw), P(loss)]
```

### 4. Self-Play & Data Pipeline (Python) — Plan 4

Generates training games using Python MCTS + PyTorch model directly. Written entirely in Python to prove the RL loop before optimizing in C++ (Plan 5).

```
training/
├── mcts.py         Python MCTS: Node, PUCT, search, move-to-policy bridge
├── selfplay.py     Self-play game generation + full RL training loop
├── encoder.py      (modified) Added encode_board() with 8-step history
├── model.py        (modified) Value head returns raw logits
├── train.py        (modified) log_softmax loss, LR scheduler, split logging
└── export.py       (modified) Softmax wrapper for TorchScript export
```

**Data flow:**
```
Current PyTorch model
  ↓
Self-play game loop (python-chess + MCTS):
  Position → encode_board() → model → (policy, value)
  MCTS search (N simulations) → move selection (with temperature)
  Record: (encoded position, visit distribution, side to move)
  ↓
Game result: W/D/L
  ↓
Label all positions with WDL
  ↓
Save training data to .npz
  ↓
Train on sliding window of recent generations
  ↓
Export TorchScript → loop back
```

**Key design decisions:**
- Python-only (no C++ needed) — uses python-chess for move generation
- Single-position inference (no batching) — simple, correct, fast enough for validation
- Small network (2 blocks, 32 filters) for development speed (~10-30s per game)
- Value head returns raw logits for numerical stability (softmax applied at inference/export)
- En passant not encoded as separate input plane (matches Lc0; inferable from position history)

**Implementation status:** Complete. Full RL loop running: self-play → train → export → repeat.

### 5. C++ Neural Net Inference — Plan 5

Loads trained network into C++ for use during MCTS and self-play.

```
src/neural/
├── encoder.h/cpp           Position → input tensor (112 × 8 × 8)
├── policy_map.h/cpp        Move → policy index mapping (1858-dim)
└── neural_evaluator.h/cpp  TorchScript model loading, Evaluator interface
```

**Data flow:**
```
Position → Encoder → float[112][8][8] tensor
  ↓
TorchScript model → single-position inference
  ↓
Policy logits[1858] → mask illegal moves → softmax → move probabilities
Value output[3] → WDL → v = P(win) - P(loss) ∈ [-1, +1]
```

**Implementation status:** Complete. Three components bridge the Python training pipeline and C++ MCTS:

1. **Position Encoder** (`encoder.h`): Converts a `Position` to a 112×8×8 float32 tensor matching the Python encoder. 8 time steps (duplicated current position — no history in C++ MCTS), 13 planes per step (6 own + 6 opponent + repetition), 8 constant planes (color, move count, castling, halfmove, bias). Board flipped for Black to move.

2. **Policy Map** (`policy_map.h`): Maps chess moves to indices in the 1858-dim policy vector. Replicates the Leela Chess Zero encoding: 56 queen-like + 8 knight + 9 underpromotion per source square. Handles Black mirroring and promotion flag extraction.

3. **NeuralEvaluator** (`neural_evaluator.h`): Implements the `Evaluator` interface from `mcts/search.h`. Loads TorchScript models via LibTorch, runs single-position inference, extracts WDL value (win - loss → [-1, +1]), applies softmax over legal move logits for policy output. Supports CPU and CUDA devices.

**Build:** Optional via `-DENABLE_NEURAL=ON -DCMAKE_PREFIX_PATH=<torch_cmake_path>`. Requires LibTorch (included in PyTorch installation).

**CLI:** `chess_engine search_nn <model_path> [fen] [iterations] [device]`

### Neural Inference (Plan 5)

**Files:** `src/neural/encoder.h/.cpp`, `src/neural/policy_map.h/.cpp`, `src/neural/neural_evaluator.h/.cpp`

Three components bridge the Python training pipeline and C++ MCTS:

1. **Position Encoder** (`encoder.h`): Converts a `Position` to a 112×8×8 float32 tensor matching the Python encoder. 8 time steps (duplicated current position — no history in C++ MCTS), 13 planes per step (6 own + 6 opponent + repetition), 8 constant planes (color, move count, castling, halfmove, bias). Board flipped for Black to move.

2. **Policy Map** (`policy_map.h`): Maps chess moves to indices in the 1858-dim policy vector. Replicates the Leela Chess Zero encoding: 56 queen-like + 8 knight + 9 underpromotion per source square. Handles Black mirroring and promotion flag extraction.

3. **NeuralEvaluator** (`neural_evaluator.h`): Implements the `Evaluator` interface from `mcts/search.h`. Loads TorchScript models via LibTorch, runs single-position inference, extracts WDL value (win - loss → [-1, +1]), applies softmax over legal move logits for policy output. Supports CPU and CUDA devices.

**Build:** Optional via `-DENABLE_NEURAL=ON -DCMAKE_PREFIX_PATH=<torch_cmake_path>`. Requires LibTorch (included in PyTorch installation).

**CLI:** `chess_engine search_nn <model_path> [fen] [iterations] [device]`

### 6. Visualization Dashboard — Plan 6

Real-time monitoring of training progress and game playback.

```
visualization/
├── server/          Python Flask + WebSocket server
└── client/          React + chessboard.jsx + Recharts
```

**Data flow:**
```
Self-play engine ──→ WebSocket ──→ Flask server ──→ WebSocket ──→ React client
Training pipeline ──→ Metrics file ──→ Flask server ──→ WebSocket ──→ React client
```

**Dashboard features (MVP):**
- Live game board (current self-play position)
- Training loss curves (policy + value)
- Games counter / positions generated
- Speed metrics (games/hour, positions/second)

## The Training Loop (End-to-End)

```
┌──────────────────────────────────────────────────┐
│                                                  │
│  1. SELF-PLAY                                    │
│     Load current weights into C++ engine         │
│     Generate N games using MCTS                  │
│     Save training data (.gz)                     │
│              │                                   │
│              ▼                                   │
│  2. TRAINING                                     │
│     Load training data into PyTorch              │
│     Sample batches from sliding window           │
│     Minimize: policy CE + value CE + L2 reg      │
│     Run 1000–2000 gradient steps                 │
│              │                                   │
│              ▼                                   │
│  3. EXPORT                                       │
│     Export PyTorch model → TorchScript (.pt)     │
│              │                                   │
│              ▼                                   │
│  4. EVALUATE (optional)                          │
│     New network vs. old network (400 games)      │
│     If win rate > 55%: accept new network        │
│              │                                   │
│              ▼                                   │
│  5. UPDATE                                       │
│     Replace self-play weights with new network   │
│     Loop back to step 1                          │
│                                                  │
└──────────────────────────────────────────────────┘
```

**Sliding window:** Training uses only the most recent 500K–2M positions. Older data becomes stale as the network improves.

## Directory Structure (Full Project)

```
chess-ai/
├── CMakeLists.txt              Root build config
├── README.md
├── src/
│   ├── core/                   Board, moves, attacks (Plan 1)
│   ├── mcts/                   MCTS search (Plan 2)
│   ├── neural/                 C++ inference (Plan 5)
│   ├── selfplay/               Game generation (Plan 4)
│   └── main.cpp                CLI entry point
├── training/                   Python/PyTorch (Plan 3)
│   ├── config.py
│   ├── encoder.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── export.py
│   ├── generate_data.py
│   └── test_training.py
├── visualization/              Dashboard (Plan 6)
│   ├── server/
│   └── client/
├── tests/                      Google Test suite
├── data/                       Training data (.gz files)
├── checkpoints/                Network weights
└── docs/
    ├── architecture.md         This file
    ├── changelog.md            Version history
    └── how-alphazero-engines-work.md
```

## Hardware Utilization Strategy

```
Ryzen 7 5800X (8C/16T)          RTX 3080 (10GB VRAM)
┌────────────────────┐           ┌──────────────────┐
│ MCTS threads: 4–8  │──batch──→│ NN inference      │
│ Self-play games: 2–4│          │ Batch size: 64–256│
│ Data I/O: 1–2      │          │ FP16 precision    │
│ Training: 2–4      │          │ cuDNN convolutions│
└────────────────────┘           └──────────────────┘

During self-play:  CPU generates positions, GPU evaluates them
During training:   GPU trains network, CPU handles data loading
Both can run concurrently (self-play uses inference, training uses backprop)
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Engine | C++17, CMake | Board representation, MCTS, self-play |
| Training | Python 3.11, PyTorch 2.11+cu126 | Neural network training |
| Inference | TorchScript / ONNX Runtime | C++ neural network inference |
| GPU | CUDA 12.6, cuDNN | Accelerated training + inference |
| Testing | Google Test | C++ unit tests + perft validation |
| Visualization | Flask, React, WebSocket | Real-time training dashboard |
| Data format | Protocol Buffers or custom binary | Training data serialization |
