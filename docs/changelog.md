# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

---

## [Unreleased]

### Added — Parallel Self-Play via GameManager
- `play_games_batched()` in `training/selfplay.py` drives `chess_mcts.GameManager` for N concurrent games with cross-game NN batching (single shared model, single forward pass per step across all game trees)
- Dynamic backfill keeps `parallel_games` slots active until `num_games` is reached
- Python-side temperature sampling over `visit_counts` per game (per-game temperature schedule preserved)
- Wired into `generate_games()` C++ path, replacing the prior serial `ThreadPoolExecutor(max_workers=1)` + per-game `CppMCTS` code path that played games strictly back-to-back
- New `--parallel-games` CLI flag (default 16) for `python -m training.selfplay loop` and `train.py` launcher prompt
- Extracted `_finalize_record()` helper so single-game `play_game()` and batched path share tablebase rescoring + WDL labeling + Q-blending logic
- Known trade-off: batched path currently skips per-game playout-cap randomization and KLD-adaptive visits (all moves use full sims)

### Added — C++ MCTS Performance Optimizations
- FP16 inference via `model_.to(torch::kHalf)` for ~2x GPU throughput on RTX 3080 Tensor Cores
- Pinned (page-locked) memory for async CPU→GPU DMA transfers with non-blocking `.to(device_)`
- Pre-allocated encode buffers in GameManager (eliminates 3.6MB/step allocation churn)
- Default batch size increased from 16 to 128 (better GPU kernel amortization)
- Default NN cache size increased from 20K to 200K entries (~40MB RAM, matches Lc0 scale)

### Changed — Node Arena Allocator + Edge/Node Separation
- Replaced `std::vector<std::unique_ptr<Node>> children_` with parallel `Edge[]` + `Node*[]` arrays
- Added `Edge` struct: stores move_bits (uint16_t) and prior_bits (FP16) for 4 bytes per edge
- Added `NodePool` class: arena allocator using `std::deque<Node>` for pointer-stable contiguous allocation
- `create_edges()` bulk-initializes edges from moves/priors arrays (replaces per-child `add_child()`)
- `ensure_child(i, pool)` lazily allocates child Nodes on first visit (deferred allocation)
- `sort_edges_by_prior()` sorts both edges and child_nodes arrays in parallel
- `select_child_advanced()` now works with edge priors directly, avoids allocating unvisited nodes
- Search::run() uses pool-allocated root; pool_.reset() between searches
- GameManager uses shared NodePool for cross-game node allocation
- All Dirichlet noise functions operate on edges instead of child Node priors
- `propagate_terminal()` handles nullptr children (unvisited edges) correctly
- Updated all 137 C++ tests to use new edge/node API

### Added — Play Against Engine
- `/play` route serving interactive play page (`visualization/static/play.html`)
- `/api/play/move` endpoint: sends position to UCI engine, returns bestmove + score + stats
- `/api/play/new` endpoint: resets engine for a new game
- UCI engine subprocess management with threading lock in Flask server
- Drag-and-drop chessboard (chessboard.js) with legal move validation (chess.js)
- Vertical evaluation bar with sigmoid-mapped score display and smooth animation
- SAN move history table with clickable navigation to past positions
- Engine info panel: score (pawns/mate), nodes, NPS
- Thinking indicator with animated dots while engine computes
- Controls: New Game, Flip Board, Undo, think time selector (0.5s--10s), play as White/Black
- Game state detection: check, checkmate, stalemate, draw (threefold, insufficient, 50-move)
- Last-move highlighting on both source and target squares
- Auto-promotion to queen on pawn promotion
- If playing as Black, engine automatically moves first
- Responsive layout: side-by-side on desktop, stacked on mobile
- Dark theme matching existing dashboard (bg: #0f0f1a, accent: #6c63ff)
- "Back to Dashboard" navigation link in header

### Added — UCI Protocol Handler
- `UCIHandler` class (`src/uci/uci.h`, `src/uci/uci.cpp`): full UCI command parser and response handler
- Supports commands: uci, isready, ucinewgame, position (startpos/fen + moves), go (wtime/btime/winc/binc/movestogo/depth/nodes/movetime/infinite), stop, setoption, quit
- Background search thread with atomic stop flag for responsive `stop` handling
- Thread-safe output via mutex-protected `send()` method
- Info callback with rate limiting (500ms / 100 iterations) emitting `info nodes/nps/score/pv`
- PositionHistory captured by value in search thread for safe concurrent access
- NPS estimate updated after each search for adaptive time allocation
- `Iterations` UCI option for configuring base search parameters
- 6 new C++ tests: uci/isready commands, position startpos/fen/moves, go nodes, go infinite+stop

### Added — Multi-Game Parallelism (GameManager)
- `GameManager` C++ class: runs N concurrent MCTS games with cross-game NN batching
- Single GPU forward pass collects leaves from all active games (e.g., 4 per game × 16 games = 64 batch)
- `init_games()` / `init_games_from_fen()` for starting from default or custom positions
- `step()` method: one round of gather/evaluate/scatter across all active games
- Exposed via pybind11 as `chess_mcts.GameManager` with Python-friendly API
- 4 new C++ MCTS integration tests for GameManager

### Added — Adaptive Early-Gen Settings
- `AdaptiveConfig` dataclass: per-generation auto-tuning of sims/max_moves/games
- 3-phase schedule: early (gen≤5, 100 sims), mid (gen 6-15, interpolated), full (gen>15, 400 sims)
- `get_gen_settings()` function returns settings for any generation number
- CLI flags: `--adaptive`/`--no-adaptive`, `--early-sims`, `--early-max-moves`, `--early-games`
- Integrated into training loop: each generation uses phase-appropriate settings
- 6 new Python tests for adaptive config (early, mid interpolation, full, disabled, boundaries)

### Added — C++ MCTS Python Integration
- `CppMCTS` wrapper class in selfplay.py matching Python MCTS API for transparent fallback
- `_CppMCTSConfig` proxy for mutable temperature/num_simulations
- `HAS_CPP_MCTS` flag: auto-detects C++ module, falls back to Python MCTS
- Training loop exports TorchScript model for C++ engine when available
- SWA model export fix: copies SWA parameters into base model for TorchScript export
- 4 new integration tests (C++ search, play game, parallel games, FEN init)

### Added — Python Bindings (pybind11)
- `chess_mcts` Python module exposing C++ MCTS search engine via pybind11
- `SearchEngine` class: loads TorchScript model, runs MCTS search from FEN + move history
- `SearchResult` with best_move, visit_counts dict, policy_target/raw_policy numpy arrays, root_value, raw_value, total_nodes
- UCI move parsing against legal moves for correct MoveFlag resolution
- Configurable SearchParams from Python dict (all fields exposed)
- `BUILD_PYTHON` CMake option (requires `ENABLE_NEURAL=ON` and pybind11)
- `attacks::init()` called automatically on module import

### Added — C++ Core Search Rewrite
- Batched gather/evaluate/scatter MCTS loop in C++ `Search::run()` with virtual loss
- `SearchResult` now includes `policy_target` (1858-dim visit distribution), `raw_policy`, and `raw_value` for training data
- `Search::run(const PositionHistory&)` overload for full history + repetition detection
- MCTS-solver terminal propagation: proven wins/losses/draws propagate up the tree
- Two-fold repetition detection during selection (checks both search path and game history)
- Shaped Dirichlet noise (KataGo-style): concentrates noise on plausible moves in C++
- Variance-scaled cPUCT: dynamic exploration coefficient scales with position complexity in C++
- Sibling blending (Ceres): visited sibling values as FPU for unvisited children in C++
- Uncertainty boosting: exploration bonus proportional to child value variance in C++
- Smart pruning: stops MCTS early when best move has insurmountable visit lead in C++
- NN evaluation cache integration during search (avoids redundant evaluations)
- Contempt support: configurable draw aversion shifts root value away from 0
- `PositionHistory::compute_hash` made public for search-level repetition detection
- 8 new C++ tests for batched search, MCTS-solver, smart pruning, two-fold repetition, shaped Dirichlet, contempt, backward compatibility

### Added — C++ Neural Inference
- Batch neural evaluation: `NeuralEvaluator::evaluate_batch()` runs multiple positions through the network in a single GPU forward pass
- `BatchRequest` / `BatchResult` structs for pre-encoded batch inference with full 1858-dim raw policy output
- 5 new tests for batch evaluation (struct creation, empty batch, single position, multiple positions, consistency with single eval)

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
