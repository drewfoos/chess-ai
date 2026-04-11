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

**Phase 1: Chess Engine Core** (Plan 1 of 6)

Building the foundation: bitboard board representation, full legal move generation, make/unmake, validated by perft against known results. No search, no neural network, no visualization yet.

**Done when:** All 5 standard perft test positions pass at depth 4–5.

Plan location: `docs/superpowers/plans/2026-04-11-chess-engine-core.md`

## Core Principles

1. **Correctness first, speed second.** Get move generation right (perft-validated) before optimizing. A fast engine with bugs is worthless.
2. **Prove the loop early.** The MVP is the closed cycle: self-play generates games → Python trains the network → C++ loads new weights → better self-play. Don't build visualization until this works.
3. **Start small, scale up.** Begin with 10-block/128-filter networks and 400 nodes/move. Increase only after validating the pipeline.
4. **No premature abstraction.** Write the straightforward version first. Add magic bitboards, SIMD, and fancy data structures only when profiling shows they're needed.
5. **Measure everything.** Perft counts, nodes/second, games/day, Elo estimates. If you can't measure it, you can't improve it.

## Key Features (Current Scope — Phase 1)

- Bitboard-based board representation
- Core types: Square, Move, Bitboard, Color, PieceType, CastlingRight
- Precomputed attack tables (knight, king, pawn)
- Ray-based sliding piece attacks (bishop, rook, queen)
- Position class with FEN parsing and serialization
- Full legal move generation (all piece types, castling, en passant, promotions)
- Make/unmake move with undo stack
- Attack detection and check detection
- Perft validation (5 standard positions, depths 1–5)
- Perft divide CLI for debugging

## Non-Goals (Right Now)

- Neural network architecture or training (Plan 3)
- MCTS search (Plan 2)
- Self-play game generation (Plan 4)
- Visualization dashboard (Plan 6)
- Magic bitboards or other speed optimizations
- Zobrist hashing / transposition tables
- UCI protocol support
- Opening books or endgame tablebases

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
├── training/           # (Plan 3) Python/PyTorch
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

## Iteration Guidelines

- Follow the implementation plan task-by-task (TDD: write failing test → implement → pass → commit)
- Don't skip ahead to later plans — each depends on the previous being solid
- When perft fails, stop and debug immediately. Don't accumulate bugs.
- Profile before optimizing. The engine doesn't need to be fast yet — it needs to be correct.
- Plans 1 and 3 (engine core and NN training pipeline) can be developed in parallel.

## Documentation

- [Brainstorm](brainstorm.md) — Original project brainstorm and feature ideas
- [How AlphaZero Engines Work](docs/how-alphazero-engines-work.md) — Technical reference for the architecture we're implementing
- [Plan 1: Chess Engine Core](docs/superpowers/plans/2026-04-11-chess-engine-core.md) — Current implementation plan

Update docs after completing each plan or making significant architectural decisions.

## Plan Roadmap

| Plan | Subsystem | Status |
|------|-----------|--------|
| 1 | Chess Engine Core (types, bitboard, position, movegen, perft) | **Current** |
| 2 | MCTS Search Engine | Not started |
| 3 | Neural Network Architecture & Training (Python/PyTorch) | Not started |
| 4 | Self-Play & Data Pipeline | Not started |
| 5 | C++ Neural Net Inference + Integration | Not started |
| 6 | Visualization Dashboard | Not started |
