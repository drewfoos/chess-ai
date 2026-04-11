# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

---

## [Unreleased]

Nothing yet.

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
