# How AlphaZero-Style Chess Engines Work

A technical reference for building a Leela Chess Zero-style engine from scratch.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Neural Network](#neural-network)
3. [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
4. [Self-Play Pipeline](#self-play-pipeline)
5. [Training Loop](#training-loop)
6. [Differences from Traditional Engines](#differences-from-traditional-engines)
7. [Performance Optimizations](#performance-optimizations)
8. [Recommended Starting Parameters](#recommended-starting-parameters)
9. [References](#references)

---

## High-Level Architecture

An AlphaZero-style engine has three components in a continuous loop:

```
┌──────────────┐
│  Self-Play   │──── training data ────┐
│  (C++/CUDA)  │                       │
└──────┬───────┘                       v
       │                       ┌──────────────┐
   new weights                 │   Training   │
       │                       │  (PyTorch)   │
       │                       └──────┬───────┘
       └───────────────────────────────┘
```

1. **Self-play** generates games using MCTS guided by a neural network
2. **Training** improves the neural network from self-play data
3. The improved network produces better self-play games, creating a virtuous cycle

There is no handcrafted evaluation function. The only chess knowledge provided is the rules of the game. Everything else — piece values, positional understanding, tactics, strategy — is learned from scratch through self-play.

---

## Neural Network

The network takes a chess position as input and outputs two things:
- **Policy**: a probability distribution over legal moves (which moves look promising)
- **Value**: an evaluation of who's winning (scalar in [-1, +1])

### Input Encoding

The position is encoded as a stack of 8x8 binary planes. Leela uses **112 input planes**:

**Per time-step (13 planes x 8 time steps = 104 planes):**

Each time step encodes one position (current + 7 most recent for history):
- 6 planes for current player's pieces (pawn, knight, bishop, rook, queen, king) — binary masks
- 6 planes for opponent's pieces
- 1 plane for repetition count (has this position occurred before?)

The 8 time steps give the network a sense of move history and repetition detection.

**Constant planes (8 planes):**
- Color to move (all 1s if white, all 0s if black)
- Total move count (normalized)
- 4 castling rights (one plane each, all 1s or all 0s)
- Halfmove clock (for 50-move rule)
- All ones (bias plane)

**Total input shape: `(112, 8, 8)`**

**Board orientation:** The board is always oriented from the perspective of the side to move. When it's Black's turn, the board is flipped vertically. This means the network always "sees" from the current player's perspective.

### Network Body: Residual Tower

The core architecture is a residual convolutional network:

1. **Initial convolution**: 3x3 conv, input_channels → filters, batch norm, ReLU
2. **Residual blocks** (repeated N times): Each block contains:
   ```
   input → 3x3 conv → batch norm → ReLU → 3x3 conv → batch norm → (+input) → ReLU
   ```
   The skip connection (adding the input back) is the key innovation from ResNets — it allows training very deep networks without vanishing gradients.

**Network sizes used in Leela:**

| Config | Blocks | Filters | Parameters | Approximate Strength |
|--------|--------|---------|------------|---------------------|
| Tiny | 6 | 64 | ~300K | Development / testing |
| Small | 10 | 128 | ~1.5M | Beginner–intermediate |
| Medium | 15 | 192 | ~5M | Strong club player |
| Large | 20 | 256 | ~15M | Master level |
| XL | 20 | 384 | ~35M | Super GM |
| Frontier | 40 | 384+ | ~100M+ | Top engine strength |

**For an RTX 3080 (10GB VRAM):** 20 blocks / 256 filters is the sweet spot. Start with 10 blocks / 128 filters for fast iteration during development.

### Squeeze-and-Excitation (SE) Layers

Leela adds SE layers to each residual block. This was a significant improvement over vanilla AlphaZero. SE learns to weight channels dynamically based on global position features:

1. **Global average pooling**: Pool each channel across 8x8 → vector of size `filters`
2. **FC down**: `filters → filters/4` (compress)
3. **ReLU**
4. **FC up**: `filters/4 → 2*filters` (expand to produce weights AND biases)
5. **Split** output into `w` (weights) and `b` (biases), each size `filters`
6. **Sigmoid** on `w` → per-channel scaling factors in [0, 1]
7. **Apply**: `output = sigmoid(w) * block_output + b`

This is Leela's modification of standard SE-Net — it produces both multiplicative gates and additive biases, giving the network more expressiveness per block.

### Policy Head

The policy head outputs move probabilities. Leela encodes the move space as:

**Move encoding (1858 possible moves):**

Moves are represented as an 8x8x73 tensor (from the perspective of the "from" square):
- **56 planes** for "queen-like" moves: 7 distances × 8 directions (N, NE, E, SE, S, SW, W, NW)
- **8 planes** for knight moves (8 possible L-shapes)
- **9 planes** for underpromotions: 3 piece types (knight/bishop/rook) × 3 directions (straight, capture-left, capture-right). Queen promotion uses the normal move encoding.

Of the 4672 total slots (64 × 73), only ~1858 correspond to valid chess moves. Illegal moves for the current position are masked to `-∞` before softmax.

**Classical policy head:**
1. 1x1 conv: filters → 80 channels
2. Batch norm + ReLU
3. Flatten to 80 × 64 = 5120, map to 1858 legal move slots

**Attention policy head (newer, stronger):**
Uses query/key attention projections from the encoder output. More parameter-efficient.

### Value Head

The value head evaluates the position:

1. 1x1 conv: filters → 32 channels
2. Batch norm + ReLU
3. Flatten: 32 × 64 = 2048
4. FC: 2048 → 128, ReLU
5. FC: 128 → 1, **tanh** → output in [-1, +1]

Where +1 = current player wins, -1 = current player loses.

**WDL (Win/Draw/Loss) head (used in modern Leela):**
- Final FC: 128 → 3 outputs
- Softmax → probabilities of (win, draw, loss)
- Value derived as: `v = P(win) - P(loss)`

WDL provides a richer training signal and helps the network understand drawn positions.

### Moves Left Head (MLH)

Modern Leela also predicts how many moves remain in the game. This helps with:
- Time management during play
- MCTS can prioritize shorter wins
- Training signal is simply the actual remaining move count

---

## Monte Carlo Tree Search (MCTS)

Unlike traditional alpha-beta search (Stockfish searches ~100M+ nodes/sec), AlphaZero uses MCTS guided by the neural network (~10K-60K nodes/sec on GPU). The key insight: instead of random rollouts, the neural network provides both a policy prior and a value estimate, eliminating rollouts entirely.

### Tree Structure

Each node in the MCTS tree stores:
- `N(s,a)` — visit count (how many times this edge was traversed)
- `W(s,a)` — total accumulated value
- `Q(s,a)` — mean value = W/N
- `P(s,a)` — prior probability from the neural network policy head

### The Four Phases

Each MCTS iteration ("playout" or "visit"):

#### 1. SELECT

Starting from the root, descend by picking the child with the highest **PUCT score** at each node:

```
PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
```

Where:
- `Q(s,a)` = exploitation term (how good is this move on average?)
- `c_puct` = exploration constant (typically **2.5**)
- `P(s,a)` = neural network prior (which moves does the NN think are good?)
- `N_parent` = total visits to the parent node
- `N(s,a)` = visits to this specific child

The formula balances exploitation (high Q) with exploration (high P, low N). Moves the neural network likes but hasn't tried much get a bonus.

**Leela's dynamic c_puct:**
```
c_puct = c_base + log((1 + N_parent + c_init) / c_init)
```
With `c_base = 2.5`, `c_init = 19652`. This increases exploration as the parent gets more visits.

#### 2. EXPAND

At a leaf node, create child nodes for all legal moves. Initialize each with N=0, W=0, and the prior P(s,a) from the neural network.

#### 3. EVALUATE

Run the neural network on the leaf position to get:
- **p** = policy vector (prior probabilities for each legal move)
- **v** = value estimate

#### 4. BACKPROPAGATE

Walk back up from the leaf to the root, updating each node:
```
N(s,a) += 1
W(s,a) += v    (negated at each level — players alternate)
Q(s,a) = W(s,a) / N(s,a)
```

The value is negated at each level because what's good for one player is bad for the other.

### First Play Urgency (FPU)

When a child has never been visited (N=0), what should Q be?

**FPU reduction:** Set unvisited Q to `parent_Q - fpu_reduction`
- Root: `fpu_reduction = 0.44`
- Non-root: `fpu_reduction = 0.25`

This encourages visiting moves the network thinks are good (high P) before trying moves it thinks are bad.

### Virtual Loss (Parallelism)

To run MCTS across multiple CPU threads:

1. When a thread selects a path, add a **virtual loss** to each node on the path:
   ```
   N(s,a) += 1     // temporarily inflate visit count
   W(s,a) -= 1     // temporarily decrease value
   ```
2. This makes Q(s,a) appear worse, discouraging other threads from exploring the same path
3. When the NN evaluation returns, remove virtual loss and apply real update:
   ```
   N(s,a) -= 1     // undo virtual
   W(s,a) += 1     // undo virtual
   N(s,a) += 1     // real update
   W(s,a) += v     // real update
   ```

Leela uses virtual_loss = 1 in current versions.

### Move Selection at Root

After search completes (e.g., 800 nodes):

- **Self-play (exploration):** Sample proportional to `N(s,a)^(1/τ)` where τ is temperature
- **Competitive play:** Pick the move with the highest visit count (temperature → 0)

---

## Self-Play Pipeline

### Game Generation

For each game:
1. Start from the initial position
2. For each move:
   a. Run MCTS (e.g., 800 nodes per move)
   b. Select a move using temperature-based sampling
   c. Record: position encoding, MCTS visit distribution, side to move
   d. Play the selected move
3. Game ends by checkmate, stalemate, draw rules, or adjudication
4. Label all positions with the game result: +1 (win), 0 (draw), -1 (loss)

### Temperature Schedule

Temperature controls exploration vs exploitation in move selection:

| Phase | Temperature | Effect |
|-------|-------------|--------|
| Moves 1–30 | τ = 1.0 | Proportional to visit counts — high diversity |
| After move 30 | τ → 0 (e.g., 0.1) | Select best move — higher quality endgame play |

Higher temperature early produces diverse openings. Lower temperature later produces better play in critical positions.

### Dirichlet Noise

At the root node only, noise is added to prior probabilities to ensure exploration:

```
P'(s,a) = (1 - ε) * P(s,a) + ε * η_a
```

Where:
- `ε = 0.25` (fraction of noise)
- `η ~ Dir(α)` where **α = 0.3 for chess**
- General formula: `α ≈ 10 / average_number_of_legal_moves`

This prevents the network from getting stuck playing the same openings. Even if the network strongly prefers e4, there's always a chance it'll try d4, c4, Nf3, etc.

### Training Data Format

Each position produces one training sample:

```
{
    input_planes:    float[112][8][8]  // Board encoding
    policy_target:   float[1858]       // MCTS visit distribution (normalized)
    value_target:    float             // Game result: {-1, 0, +1}
    wdl_target:      float[3]          // [P(win), P(draw), P(loss)]
}
```

**Critical insight:** The policy target is the **MCTS visit distribution**, NOT the raw network output. MCTS improves upon the network's policy through search, and the network trains to match this improved policy. This is the core "knowledge distillation" mechanism:

```
policy_target[a] = N(root, a) / sum(N(root, all_moves))
```

Leela stores data in gzipped protocol buffer format. A typical game of ~80 plies produces ~80 training positions.

### Throughput Estimates (RTX 3080)

| Network Size | Nodes/Move | Games/Minute | Positions/Minute |
|-------------|------------|--------------|------------------|
| 10b × 128f | 800 | ~5–10 | ~400–800 |
| 20b × 256f | 800 | ~1–3 | ~80–240 |
| 10b × 128f | 400 | ~10–20 | ~800–1600 |

At 5 games/minute → ~7,200 games/day → ~576,000 training positions/day with the small network.

---

## Training Loop

### Loss Function

```
L = L_policy + L_value + c_reg * L_regularization
```

**Policy loss** — cross-entropy between MCTS visits and network output:
```
L_policy = -Σ π(a) * log(p(a))
```
Where `π` is the MCTS visit distribution (target) and `p` is the network's softmax output.

**Value loss** — MSE between game outcome and network value:
```
L_value = (z - v)²
```
Where `z` is the game result and `v` is the network's tanh output.

Or for WDL: cross-entropy over {win, draw, loss}.

**Regularization** — L2 weight decay:
```
L_reg = c * Σ θ²     (c = 1e-4)
```

### Training Procedure

1. Collect self-play games into a **sliding window** of recent positions (e.g., last 500K–2M positions)
2. Sample random mini-batches from the window
3. Train for ~1000–2000 gradient steps
4. Export new weights
5. Begin using new weights for self-play
6. Repeat

The sliding window ensures the network trains on data generated by recent (similar-strength) versions of itself, not on stale data from much weaker past versions.

### Network Gating (Optional)

To ensure the network is improving:
1. New network plays ~400 games against the current best
2. If it wins >55%, it becomes the new best
3. Otherwise, keep the old one

In practice, Leela relaxed gating over time because it slows the pipeline. For your project, skip gating initially and always use the latest network.

### Learning Rate Schedule

**With SGD + momentum 0.9 (AlphaZero-style):**
- Start at 0.02
- Drop to 0.002 after ~100K steps
- Drop to 0.0002 after ~200K steps
- Drop to 0.00002 after ~400K steps

**With Adam (simpler, recommended to start):**
- Start at 1e-3 or 2e-4
- Cosine annealing or step decay

---

## Differences from Traditional Engines

### AlphaZero/Leela vs. Stockfish

| Aspect | AlphaZero/Leela | Stockfish (NNUE) |
|--------|----------------|------------------|
| **Search algorithm** | MCTS (~10K–60K nodes/sec) | Alpha-beta (~100M+ nodes/sec) |
| **Evaluation** | Full neural network (slow, accurate) | NNUE — incrementally updated (fast) |
| **Knowledge source** | Learned entirely from self-play | Handcrafted + NN trained on engine evals |
| **Move ordering** | Neural network prior | Killer moves, history heuristic, MVV-LVA |
| **Pruning** | Implicit (low-prior moves get few visits) | Alpha-beta, null move, LMR, futility |
| **Search depth** | No explicit depth; node budget | Typically 30–50+ ply |
| **Opening book** | None (learns openings) | Optional |
| **Endgame tables** | Optional Syzygy support | Syzygy tablebases |

### Why MCTS + NN Works

Alpha-beta needs an extremely fast evaluation because it evaluates millions of positions. MCTS works with a slow but accurate neural network because:
- Each evaluation is expensive (~0.1–1ms with batching) but highly informative
- MCTS naturally focuses on promising lines (guided by the policy prior)
- 800 well-targeted evaluations can beat millions of shallow ones

### What NNUE Is (for contrast)

Stockfish's NNUE is a fundamentally different approach:
- Small network (~10M params) for position evaluation only
- Special architecture where the first layer can be **incrementally updated** when a piece moves (avoids full recomputation)
- Runs on CPU at millions of evaluations/second
- Still uses alpha-beta with traditional pruning
- Trained on Stockfish's own evaluations (supervised learning), not self-play

---

## Performance Optimizations

### Batch Inference (Most Critical)

Instead of evaluating one position at a time on the GPU:

1. MCTS threads generate multiple leaf positions simultaneously (virtual loss diversifies paths)
2. Positions are collected into a **batch** (typically 64–256)
3. The batch goes to the GPU for parallel inference
4. Results are distributed back to requesting threads

```
Thread 1 ──┐
Thread 2 ──┤
Thread 3 ──┼── Batch Queue ──→ GPU Thread ──→ cuDNN ──→ Results
Thread 4 ──┤
...        ──┘
```

**RTX 3080 optimal batch size:** 64–256 for inference.

### Mixed Precision (FP16)

- **Training:** PyTorch AMP — FP16 forward pass, FP32 for loss and weight updates
- **Inference:** FP16 throughout — half the memory, ~2x throughput
- RTX 3080: ~29.8 TFLOPS FP16 vs ~14.9 TFLOPS FP32 (Tensor Cores)

### CUDA Acceleration

- **cuDNN** for convolutions (the dominant operation)
- **Winograd-transformed** 3x3 convolutions (significant speedup)
- Custom CUDA kernels for SE layers, batch norm, activation
- Fused operations (conv + bias + batch norm + ReLU)

For C++ inference: use **TorchScript** or **ONNX Runtime with CUDA** execution provider. TensorRT is fastest but harder to set up.

### Multi-Threading Strategy (Ryzen 7 5800X)

- 2–4 MCTS threads per game
- 2–4 concurrent self-play games
- 1 dedicated GPU inference thread (batches from all games)
- Total: 8–16 threads keeping the GPU fed
- Target: >90% GPU utilization

### Data Augmentation

Chess has left-right **mirror symmetry** (flip a↔h files). Each position generates 2 training samples for free.

Note: Unlike Go (8-fold rotational symmetry), chess only has 1 axis of symmetry due to castling conventions.

### Tree Reuse

After a move is played, reuse the subtree rooted at the chosen move. Discard the rest. This avoids rebuilding the tree from scratch every move.

---

## Recommended Starting Parameters

For initial development and validation on RTX 3080 + Ryzen 7 5800X:

### Network
| Parameter | Dev Value | Production Value |
|-----------|-----------|-----------------|
| Residual blocks | 10 | 20 |
| Filters | 128 | 256 |
| SE ratio | 4 | 4 |
| Policy head | Classical (conv) | Attention |
| Value head | WDL (3 outputs) | WDL |
| History planes | 8 time steps | 8 time steps |

### MCTS
| Parameter | Value | Notes |
|-----------|-------|-------|
| c_puct | 2.5 | Exploration constant |
| Dirichlet alpha | 0.3 | Root noise (10 / avg legal moves) |
| Dirichlet epsilon | 0.25 | Noise fraction |
| FPU reduction (root) | 0.44 | First play urgency |
| FPU reduction (non-root) | 0.25 | |
| Virtual loss | 1 | For parallel MCTS |
| Nodes per move (self-play) | 800 | Can start with 400 for speed |

### Self-Play
| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature (moves 1–30) | 1.0 | High exploration |
| Temperature (after move 30) | 0.1 | Near-greedy |
| Resign threshold | -0.95 (WDL) | Resign clearly lost games |
| Concurrent games | 2–4 | Fill GPU batches |
| Threads per game | 2–4 | |

### Training
| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | Simpler than SGD for starting |
| Learning rate | 1e-3 | With cosine annealing |
| Batch size | 256 | Fits in 10GB VRAM easily |
| Weight decay | 1e-4 | L2 regularization |
| Training window | 500K positions | Sliding window of recent data |
| Steps per generation | 1000–2000 | Per batch of new games |
| Mixed precision | Yes | PyTorch AMP |

### Validation Milestones
| Games Generated | Expected Behavior |
|----------------|-------------------|
| 1,000 | Learns piece values, avoids hanging pieces |
| 10,000 | Basic tactics (forks, pins), coherent openings |
| 50,000 | Intermediate play, some strategic understanding |
| 200,000+ | Strong club level, tactical accuracy |

---

## References

### Papers
- **AlphaZero**: Silver et al., "A general reinforcement learning algorithm that masters chess, shogi and Go through self-play" (Science, 2018)
- **AlphaGo Zero**: Silver et al., "Mastering the game of Go without human knowledge" (Nature, 2017)
- **MuZero**: Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Nature, 2020)
- **Squeeze-and-Excitation Networks**: Hu et al., "Squeeze-and-Excitation Networks" (CVPR, 2018)

### Leela Chess Zero
- Website: https://lczero.org/
- Source (C++ engine): https://github.com/LeelaChessZero/lc0
- Source (training): https://github.com/LeelaChessZero/lczero-training
- Wiki: https://lczero.org/dev/wiki/
- Discord: https://discord.gg/pKujYxD

### Chess Programming
- Chess Programming Wiki: https://www.chessprogramming.org/
- MCTS: https://www.chessprogramming.org/Monte-Carlo_Tree_Search
- Perft results: https://www.chessprogramming.org/Perft_Results

### Tools
- python-chess: https://python-chess.readthedocs.io/
- PyTorch: https://pytorch.org/docs/stable/
- ONNX Runtime: https://onnxruntime.ai/
- cuDNN: https://developer.nvidia.com/cudnn
