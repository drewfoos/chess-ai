# CLAUDE.md

## Project Overview

A chess AI built from scratch in C++17, trained via self-play reinforcement learning using an AlphaZero-style architecture (MCTS + neural network). Training pipeline in Python/PyTorch. Live visualization dashboard for monitoring training progress.

This is a ground-up implementation — not a fork of Leela Chess Zero or any existing engine. Every component (move generation, MCTS, neural network, self-play pipeline) is written from scratch.

**Hardware target:** RTX 3080 (10GB VRAM) + Ryzen 7 5800X (8 cores/16 threads), Windows 11.

## Project Goals

- **Primary: Maximum Elo** through an efficient, well-optimized implementation
- **Secondary: Deep understanding** of every component — RL, MCTS, neural networks, chess engine internals
- Demonstrate technical depth suitable for a PhD application
- Build a complete, self-contained training pipeline that runs on consumer hardware

**Success:** Reach 1800–2000+ Elo with code the author can fully explain.

## Current Milestone

**Phase 6: Visualization Dashboard** (Plan 6 of 6) — **Complete**

Added training monitoring dashboard: MetricsLogger writes per-generation JSON metrics during self-play, Flask server exposes a REST API, and a single-page dashboard displays loss curves (Chart.js), game replay (chessboard.js), and speed statistics. All six plans are now complete.

Plan location: (plan file in Claude Code session plans)

## Core Principles

1. **Correctness first, speed second.** Get move generation right (perft-validated) before optimizing. A fast engine with bugs is worthless.
2. **Prove the loop early.** The MVP is the closed cycle: self-play generates games → Python trains the network → C++ loads new weights → better self-play. Don't build visualization until this works.
3. **Start small, scale up.** Begin with 10-block/128-filter networks and 400 nodes/move. Increase only after validating the pipeline.
4. **No premature abstraction.** Write the straightforward version first. Add magic bitboards, SIMD, and fancy data structures only when profiling shows they're needed.
5. **Measure everything.** Perft counts, nodes/second, games/day, Elo estimates. If you can't measure it, you can't improve it.

## Key Features (Current Scope — Through Phase 6)

- Everything from Phase 1 (bitboard, move generation, perft)
- MCTS tree with Node struct (visit count, value, prior, children)
- PUCT-based child selection with exploration/exploitation balance
- Evaluator interface for neural network integration
- RandomEvaluator: uniform policy + material-based value
- Dirichlet noise at root for exploration diversity
- Temperature-based move selection (τ=1 for exploration, τ→0 for greedy)
- First Play Urgency (FPU) reduction
- CLI `search` command showing best move + visit distribution
- ChessNetwork: SE-ResNet (10 blocks, 128 filters, ~13M params)
- Position encoding: FEN → 112×8×8 tensor (board flipped for black)
- Position history encoding: 8-step history via python-chess Board.move_stack
- Move encoding: 1858-dim policy index (Leela-style 64×73 filtered)
- Training loop: policy CE + value CE (log_softmax) + moves-left Huber loss + AdamW + LR warmup + MultiStepLR
- Value head returns raw logits (softmax applied at export/inference)
- TorchScript export with softmax wrapper for C++ inference (MLH excluded)
- Python MCTS: dynamic PUCT (c=3.0), Dirichlet noise, temperature, FPU reduction (1.2), MCTS-solver, batched inference (virtual loss + gather-batch-scatter), NN evaluation cache, playout cap randomization, KLD-adaptive visit count
- Self-play game generation: temperature schedule, resign logic, WDL labeling, Q-value blending
- Full RL training loop: generate → train → export → repeat with sliding window
- CLI: `python -m training.selfplay generate` and `python -m training.selfplay loop`
- Synthetic data generator for pipeline testing
- Lc0 Tier 1: dynamic c_puct, policy softmax temperature (1.61), gradient clipping, LR warmup (250 steps), Mish activation
- Lc0 Tier 2: moves-left head (Huber loss), Q-value blending, MCTS-solver (certainty propagation), attention policy head
- Lc0 Tier 3: SGD+Nesterov option, tree reuse, diff-focus sampling, SWA, Glorot init
- FPU reduction: 1.2 at root and non-root (updated to Lc0 current defaults)
- Syzygy tablebase rescoring for endgame positions (optional, `--syzygy` flag)
- Playout cap randomization: alternates full/quick search for ~2x data efficiency
- KLD-adaptive visit count: scales simulations based on policy divergence
- Smart pruning: stops MCTS early when best move has insurmountable visit lead
- Opening randomization: 5% of games play random opening moves for diversity
- Auxiliary soft policy target (KataGo-style): forces network to learn non-obvious moves
- Two-fold repetition detection: treats 2-fold repetitions as draws in search tree
- Shaped Dirichlet noise (KataGo): concentrates exploration on plausible moves
- Uncertainty boosting (Ceres): value variance exploration bonus in PUCT
- Variance-scaled cPUCT (KataGo): dynamic exploration scales with position complexity
- WDL rescale/contempt: configurable draw aversion for evaluation play
- Temperature decay: smooth decay with 0.4 floor (replaces hard cutoff at move 30)
- Badgame split: forks games when temperature causes blunders, replays greedy
- Sibling blending (Ceres): visited sibling values as FPU for unvisited children
- Policy edge sorting: children sorted by prior descending for cache locality
- Prior compression: Node priors stored as float16 (halves per-node memory)
- `python -m training` shortcut for launching the training loop
- Mirror data augmentation (horizontal a↔h flip) for 2× training data
- Mixed precision training (FP16 via PyTorch AMP on CUDA)
- 138 Python tests covering encoding, model, training, MCTS, self-play, metrics, server, adaptive config, C++ integration
- C++ position encoder matching Python (112×8×8 tensor, board flipped for Black)
- C++ policy map (1858-dim Leela-style move encoding)
- NeuralEvaluator: TorchScript model loading via LibTorch, single + batch inference
- `search_nn` CLI command for neural network MCTS search
- Optional ENABLE_NEURAL CMake build with LibTorch
- C++ batched MCTS: gather/evaluate/scatter loop with all Python MCTS features ported to C++
- `chess_mcts` pybind11 module: SearchEngine + GameManager exposed to Python
- CppMCTS wrapper: transparent fallback to Python MCTS when C++ module unavailable
- GameManager: N concurrent games with cross-game NN batching (single GPU forward pass)
- Adaptive early-gen settings: 3-phase auto-tune (sims/max_moves/games) for faster early training
- NN evaluation cache in C++: Zobrist-keyed cache avoids redundant inference
- BUILD_PYTHON CMake option for pybind11 module (requires ENABLE_NEURAL=ON)
- MetricsLogger: per-generation JSON metrics with game data (`training/metrics.py`)
- Flask visualization server with REST API endpoints (`visualization/server.py`)
- Single-page dashboard: loss curves (Chart.js), game replay (chessboard.js), speed stats (`visualization/static/index.html`)
- Self-play loop instrumented with metrics logging
- UCI move recording for game replay
- `python -m visualization.server --metrics-dir <path>` to launch dashboard
- Play Against Engine page (`/play`): interactive chessboard with drag-and-drop, eval bar, move history, engine info
- UCI engine subprocess management in Flask server with `/api/play/move` and `/api/play/new` endpoints
- UCI protocol support: `chess_engine` (no args) enters UCI mode, `chess_engine uci <model> <device>` for neural network
- UCIHandler: threaded search with stop flag, info output, time management (wtime/btime/inc/movetime/depth/nodes/infinite)
- Time manager: converts UCI clock parameters to MCTS iteration counts with safety margins
- Compatible with chess GUIs (Arena, CuteChess) and lichess-bot for online play
- Search info callback: emits `info nodes N nps X score cp Y pv MOVE` during search
- 178 C++ tests (including 12 UCI protocol + time manager tests)
- FP16 inference: model runs in half-precision on CUDA for ~2x GPU throughput (RTX 3080 Tensor Cores)
- Pinned memory: page-locked CPU buffers with async non-blocking DMA transfers to GPU
- Edge/Node separation (Lc0-style): unexpanded children stored as 4-byte Edge structs instead of full 56-byte Nodes
- NodePool arena allocator: contiguous `std::deque<Node>` eliminates per-node heap allocation (~2K allocs/step → 0)
- Pre-allocated encode buffers in GameManager (eliminates 3.6MB/step allocation churn)
- Default batch size 128, NN cache 200K entries (up from 16 and 20K)

## Non-Goals (Right Now)

- Magic bitboards or other speed optimizations
- Zobrist hashing / transposition tables
- Opening books or endgame tablebases
- Multi-threaded MCTS (current parallelism is cross-game batching, not intra-game threading)
- WebSocket live streaming (polling is sufficient for current needs)

## Technical Guidelines

### Architecture

```
chess-ai/
├── CMakeLists.txt
├── src/
│   ├── core/           # Board representation, types, move generation
│   │   ├── types.h     # Enums, Move struct, constants
│   │   ├── bitboard.h/cpp
│   │   ├── attacks.h/cpp
│   │   ├── position.h/cpp
│   │   └── movegen.h/cpp
│   ├── mcts/           # (Plan 2) Search
│   ├── neural/         # (Plan 5) C++ inference
│   ├── selfplay/       # (Plan 4) Game generation
│   └── main.cpp
├── training/           # (Plan 3) Python/PyTorch — SE-ResNet, encoder, training, export
├── tests/              # Google Test
└── docs/
```

### Build System

- **C++17** with CMake 3.20+
- **MSVC** (Visual Studio 2022 Community)
- **Google Test** via FetchContent (no system install)
- Generator: `cmake -B build -G "Visual Studio 17 2022"`
- Build: `cmake --build build --config Release`
- Test: `ctest --test-dir build --build-config Release --output-on-failure`

### Code Style

- Prefer `constexpr` and `inline` for small functions in headers
- Use fixed-width integer types (`uint64_t`, `uint16_t`, etc.)
- Bitboard operations should be branchless where possible
- Keep position copying cheap — needed for legal move filtering
- Platform intrinsics guarded by `#ifdef _MSC_VER` (use `__popcnt64`, `_BitScanForward64` on MSVC)

### Performance Targets (Phase 1)

- Starting position perft(5) = 4,865,609 nodes in under 5 seconds
- All 5 perft positions pass at their specified depths

## Repository Etiquette

### Branching
- ALWAYS create a feature branch before major changes
- NEVER commit directly to main
- Naming format: `feature/<description>` or `fix/<description>`

### Commits
- Clear, descriptive commit messages
- Small, focused commits — one logical change per commit
- Commit after each passing test milestone (TDD rhythm)

## Testing Strategy

### Primary: Perft Testing

Perft is the gold standard for chess engine correctness. It counts all leaf nodes at a given search depth and compares against known-correct values. If perft numbers match, move generation is correct.

**Standard test positions:**
1. Starting position (depths 1–5)
2. Kiwipete — stress test for pins, en passant, castling
3. Position 3 — en passant and promotion edge cases
4. Position 4 — mirrored castling
5. Position 5 — complex midgame with promotions

### Debugging Failed Perft

1. Run `divide` at the failing depth (splits count by first move)
2. Compare each move's count against a known-correct engine (e.g., Stockfish `go perft N`)
3. Find the first move with wrong count
4. Recurse: set up that position, run divide at depth-1
5. Repeat until you find the leaf position with the wrong move count
6. Fix the bug

### Unit Tests (Google Test)

- `test_types.cpp` — Move encoding/decoding, square arithmetic
- `test_bitboard.cpp` — Popcount, LSB, shifts, wrapping
- `test_attacks.cpp` — Attack tables for each piece type
- `test_position.cpp` — FEN parsing round-trips, make/unmake, check detection
- `test_movegen.cpp` — Move counts and specific moves for targeted positions
- `test_perft.cpp` — Full perft validation

Run all: `ctest --test-dir build --build-config Release --output-on-failure`

### Python Tests (pytest)

- `training/test_training.py` — 120 tests covering config, move encoding, position encoding, model architecture (attention policy head, SE-ResNet, MLH), dataset (mirror augmentation, surprise weights, playout cap), training step, optimizers (AdamW/SGD), TorchScript export, end-to-end pipeline, GPU training, MCTS (solver, dynamic c_puct, tree reuse, batched inference, virtual loss, NN cache, KLD-adaptive, smart pruning, two-fold repetition, uncertainty boosting, variance-scaled cPUCT, shaped Dirichlet, contempt), self-play (Q-value blending, surprise recording, SWA, playout cap, tablebase rescoring, opening randomization), soft policy target, metrics, server, integration

Run all: `python -m pytest training/test_training.py -v`

### C++ Neural Tests (requires LibTorch)

Build with `-DENABLE_NEURAL=ON` to enable neural evaluator tests (~6 additional tests).

## Iteration Guidelines

- Follow the implementation plan task-by-task (TDD: write failing test → implement → pass → commit)
- Don't skip ahead to later plans — each depends on the previous being solid
- When perft fails, stop and debug immediately. Don't accumulate bugs.
- Profile before optimizing. The engine doesn't need to be fast yet — it needs to be correct.
- Plans 1 and 3 (engine core and NN training pipeline) can be developed in parallel.

## Documentation

- [Architecture](docs/architecture.md) — System design and data flow
- [Changelog](docs/changelog.md) — Version history
- [How AlphaZero Engines Work](docs/how-alphazero-engines-work.md) — Technical reference for the architecture we're implementing
- [Lc0 Optimizations](docs/lc0-optimizations.md) — Research on Lc0 source code optimizations, prioritized for our hardware
- [Brainstorm](brainstorm.md) — Original project brainstorm and feature ideas
- [Plan 1: Chess Engine Core](docs/superpowers/plans/2026-04-11-chess-engine-core.md) — Engine core implementation plan
- [Plan 2: MCTS Search](docs/superpowers/plans/2026-04-11-mcts-search.md) — MCTS implementation plan
- [Plan 3: Neural Network & Training](docs/superpowers/plans/2026-04-11-neural-network-training.md) — Training pipeline implementation plan
- [Plan 4: Self-Play & Data Pipeline](docs/superpowers/plans/2026-04-11-selfplay-pipeline.md) — Self-play pipeline implementation plan
- [Plan 5: C++ Inference](docs/superpowers/plans/2026-04-11-cpp-inference.md) — C++ inference integration plan
- Plan 6: Visualization Dashboard — plan in Claude Code session plans

### Keeping Docs Current

- **Changelog:** Update `docs/changelog.md` with every commit. Add entries under `[Unreleased]` using Added/Changed/Fixed/Removed categories. When a milestone is completed, stamp the version and date.
- **Architecture:** Update `docs/architecture.md` when adding new components, changing data flow, or making structural decisions.
- Update docs in the **same commit** as the code change — not retroactively.

## Plan Roadmap

| Plan | Subsystem | Status |
|------|-----------|--------|
| 1 | Chess Engine Core (types, bitboard, position, movegen, perft) | **Complete** |
| 2 | MCTS Search Engine | **Complete** |
| 3 | Neural Network Architecture & Training (Python/PyTorch) | **Complete** |
| 4 | Self-Play & Data Pipeline (Python) | **Complete** |
| 5 | C++ Neural Net Inference + Integration | **Complete** |
| 6 | Visualization Dashboard | **Complete** |
