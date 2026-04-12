# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

---

## [Unreleased]

### Added — Performance
- Batched MCTS inference: gather-batch-scatter loop collects N leaf positions per iteration for a single GPU forward pass (configurable `batch_size`, default 16)
- Virtual loss: unscored virtual visits prevent traversal collisions within a batch (Lc0-style)
- NN evaluation cache: caches up to 20K position evaluations by FEN to avoid redundant inference
- Playout cap randomization: alternates full/quick search (25% full by default), trains policy only on full-search positions (~2x data efficiency)
- KLD-adaptive visit count: scales simulations (100-800) based on KL divergence between raw policy and search policy
- Syzygy tablebase rescoring: optional post-processing replaces endgame WDL with tablebase-correct results (`--syzygy` flag)
- Updated search parameters to Lc0 current defaults: CPuct=3.0, PST=2.2, FPU=1.2
- Two-fold repetition detection: treats 2-fold repetitions as draws in search tree (prevents repetition-seeking)
- WDL rescale/contempt: configurable draw aversion shifts value away from 0 (for evaluation play)
- Shaped Dirichlet noise (KataGo): concentrates exploration noise on plausible moves
- Uncertainty boosting (Ceres): exploration bonus proportional to child value variance
- Variance-scaled cPUCT (KataGo): dynamic exploration coefficient scales with position complexity
- Node value variance tracking: `sum_sq_value` enables per-node uncertainty estimation
- Smart pruning: stops MCTS early when best move has insurmountable visit lead (factor 1.33)
- Opening randomization: 5% of games play 2-8 random legal moves from starting position for diversity
- Auxiliary soft policy target (KataGo-style): raises search distribution to power 1/T (T=4), weighted 2x, forces network to learn about non-obvious moves
- Temperature decay schedule: smooth decay with 0.4 floor (replaces hard cutoff)
- Badgame split: forks games when temperature causes blunders, replays with greedy play
- Sibling blending (Ceres): uses visited sibling values for first-visit FPU
- Policy edge sorting: children sorted by prior descending for cache locality
- Prior compression: Node priors stored as float16 (halves per-node memory)
- `python -m training` shortcut for launching the training loop with sensible defaults

### Changed
- Training batch size default increased from 256 to 2048 (matching Lc0)

### Added — Lc0 Optimizations (Tier 1 + Tier 2)
- Dynamic c_puct: logarithmic growth with parent visit count (AlphaZero defaults: c_init=2.5, c_base=19652, c_factor=2.0) in Python MCTS and C++ search
- Policy softmax temperature (1.61): widens NN prior distribution before PUCT in Python MCTS and C++ NeuralEvaluator
- Gradient clipping: `clip_grad_norm_(max_norm=10.0)` in training loop
- LR warmup: 250-step linear warmup via `SequentialLR(LinearLR + MultiStepLR)`
- Mish activation: replaced all ReLU in SE-ResNet, policy head, and value head
- Moves-left head (MLH): auxiliary network output predicting remaining plies, Huber loss (δ=10, scaled 1/20)
- Q-value blending: configurable `q_ratio` blends game result WDL with search root Q-value
- MCTS-solver: terminal node certainty propagation (proven wins/losses/draws skip unnecessary search)
- Attention policy head: Q@K^T scaled dot-product attention over 64 square tokens, promotion offsets, gather-based mapping to 1858 (replaces classical FC policy head)
- SGD + Nesterov momentum: `create_optimizer(optimizer_type='sgd')` option alongside AdamW
- Tree reuse: `MCTS.reuse_tree()` extracts subtree for play mode; `search(root=)` accepts existing root
- Diff-focus sampling: records policy/value surprise per position, `WeightedRandomSampler` oversamples informative positions
- Stochastic Weight Averaging (SWA): `AveragedModel` for smoother self-play policy
- Glorot normal initialization: `xavier_normal_` on all conv/FC weights, proper BN init
- `docs/lc0-optimizations.md`: comprehensive research document with 14 optimizations in 3 tiers (all implemented)

### Changed
- `ChessNetwork` now returns 3 outputs: `(policy, value, moves_left)` 
- Default policy head is now attention-based (`use_attention_policy=True`); classical FC head available via config
- `MCTSConfig` expanded: `c_puct_init`, `c_puct_base`, `c_puct_factor`, `fpu_reduction_root`, `policy_softmax_temperature`
- `SelfPlayConfig` expanded: `q_ratio` for Q-value blending
- `GameRecord` now includes `moves_left` field
- MCTS `Node` has `terminal_status` for solver propagation
- `_ExportWrapper` handles 3-output model (MLH excluded from TorchScript export)

### Fixed
- Halfmove clock not reset on non-capture promotions (`position.cpp`)

---

## [0.6.0] - 2026-04-11

**Visualization Dashboard — Plan 6 complete.**

### Added
- MetricsLogger: writes per-generation JSON metrics for training monitoring (`training/metrics.py`)
- Flask visualization server: REST API serving training metrics (`visualization/server.py`)
- Single-page training dashboard: loss curves (Chart.js), game replay (chessboard.js), speed metrics (`visualization/static/index.html`)
- Self-play loop instrumented with metrics logging (games, training loss, timing)
- UCI move recording in GameRecord for game replay
- End-to-end integration test (self-play -> metrics -> Flask API)

### Changed
- `GameRecord` now includes `moves_uci` field
- `generate_games()` accepts optional `metrics_logger` parameter
- `training_loop()` automatically writes metrics to `<output_dir>/metrics/`

### Dependencies
- Added: `flask` for visualization server

---

## [0.5.0] - 2026-04-11

### Added
- C++ position encoder: converts Position to 112×8×8 float32 tensor (matches Python encoder exactly)
- C++ policy map: move-to-policy-index mapping for 1858-dim neural network output
- NeuralEvaluator: loads TorchScript models via LibTorch, implements Evaluator interface for MCTS
- `search_nn` CLI command: MCTS search using neural network evaluation
- Optional `ENABLE_NEURAL` CMake integration for LibTorch dependency
- Cross-validation script for verifying C++/Python encoder parity
- MCTS + neural network integration test

---

## [0.4.0] - 2026-04-11

**Self-Play & Data Pipeline — Plan 4 complete.**

### Added
- Python MCTS implementation: PUCT selection, Dirichlet noise, temperature selection (`training/mcts.py`)
- Self-play game generation: temperature schedule, resign logic, WDL labeling (`training/selfplay.py`)
- Full RL training loop: generate -> train -> export -> repeat with sliding window (`training/selfplay.py`)
- Position history encoding: `encode_board()` fills 8 time steps from move stack (`training/encoder.py`)
- Move bridge: `chess_move_to_policy_index()` maps python-chess moves to 1858-dim encoding
- LR scheduling: `MultiStepLR` support in training loop
- CLI: `python -m training.selfplay generate` and `python -m training.selfplay loop`
- Integration tests: self-play -> train end-to-end, GPU self-play

### Changed
- Value head returns raw logits (previously applied softmax in model)
- Value loss uses `log_softmax` for numerical stability (previously `log(softmax + eps)`)
- `compute_loss` returns `(total, policy, value)` tuple for split loss logging
- TorchScript export applies softmax wrapper for value head
- Training loop logs policy and value loss separately

### Dependencies
- Added: `python-chess` for self-play move generation

---

## [0.3.0] - 2026-04-11

**Neural Network & Training Pipeline — Plan 3 complete.**

### Added
- Neural network architecture: SE-ResNet with 10 residual blocks, 128 filters, ~13M params (`training/model.py`)
- Squeeze-and-Excitation blocks: Leela-style with multiplicative weights and additive biases
- Policy head: 1×1 conv → 80 channels → FC → 1858 move logits
- WDL value head: 1×1 conv → 32 channels → FC(128) → FC(3) → softmax
- Position encoder: FEN → 112×8×8 tensor with board flipping for black (`training/encoder.py`)
- Move encoder: 1858-dim policy index ↔ chess move mapping, Leela-style 64×73 (`training/encoder.py`)
- Training loop: policy cross-entropy + value cross-entropy + AdamW weight decay (`training/train.py`)
- TorchScript export for C++ inference via torch.jit.trace (`training/export.py`)
- Synthetic data generator for pipeline testing (`training/generate_data.py`)
- Chess training dataset loader for .npz files (`training/dataset.py`)
- NetworkConfig dataclass for model hyperparameters (`training/config.py`)
- Training test suite: 32 tests covering encoding, model, training, export, end-to-end pipeline

### Validated
- End-to-end pipeline: generate synthetic data → train → export TorchScript → reload and verify
- GPU training on RTX 3080 (CUDA 12.6)
- Loss decreases over training steps (policy CE + value CE)

---

## [0.2.0] - 2026-04-11

**MCTS Search Engine — Plan 2 complete.** Merged via [PR #2](https://github.com/drewfoos/chess-ai/pull/2).

### Added
- MCTS tree search: `Node` struct with visit count, value, prior, children (`src/mcts/node.h/cpp`)
- MCTS search loop: select (PUCT) → expand → evaluate → backpropagate (`src/mcts/search.h/cpp`)
- `Evaluator` interface for pluggable position evaluation
- `RandomEvaluator`: uniform policy + material-based value (stub for neural network)
- Dirichlet noise at root for exploration diversity
- Temperature-based move selection for self-play
- First Play Urgency (FPU) reduction at root and non-root nodes
- CLI `search` command: `chess_engine search [fen] [iterations]`
- MCTS test suite: 19 tests covering node operations, PUCT, search, and edge cases

### Validated
- Solves back rank mate, knight forks, Scholar's mate, free captures
- Correctly handles checkmate, stalemate, single legal move positions

---

## [0.1.0] - 2026-04-11

**Chess Engine Core — Plan 1 complete.** Merged via [PR #1](https://github.com/drewfoos/chess-ai/pull/1).

### Added
- Core types: `Square`, `Move`, `Bitboard`, `Color`, `PieceType`, `CastlingRight` (`src/core/types.h`)
- Bitboard utilities: `popcount`, `lsb`, `pop_lsb`, directional shifts with wrapping prevention (`src/core/bitboard.h/cpp`)
- Attack tables: precomputed knight/king/pawn, ray-based bishop/rook/queen (`src/core/attacks.h/cpp`)
- Position class: FEN parsing/serialization, make/unmake move, attack/check detection (`src/core/position.h/cpp`)
- Full legal move generation: all piece types, castling, en passant, promotions (`src/core/movegen.h/cpp`)
- Perft validation: 5 standard positions at multiple depths — 77 tests, all passing (`tests/test_perft.cpp`)
- Perft divide CLI for debugging move generation (`src/main.cpp`)
- CMake build system with Google Test via FetchContent (`CMakeLists.txt`)
- System architecture document (`docs/architecture.md`)
- Technical reference: How AlphaZero Engines Work (`docs/how-alphazero-engines-work.md`)

### Performance
- Starting position perft(5) = 4,865,609 nodes in 0.18s (~27M nodes/sec)

### Infrastructure
- CMake 4.3.1, CUDA Toolkit 12.6, PyTorch 2.11.0+cu126 installed
- GitHub repository: drewfoos/chess-ai
