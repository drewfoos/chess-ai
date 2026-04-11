# Chess AI

A chess engine built from scratch in C++17, trained via self-play reinforcement learning using an AlphaZero-style architecture (MCTS + neural network).

## Overview

- **Engine**: Bitboard-based board representation, full legal move generation, MCTS search
- **Training**: Python/PyTorch self-play pipeline with policy + value network
- **Visualization**: Real-time dashboard for monitoring training progress
- **Hardware target**: RTX 3080 (10GB VRAM) + Ryzen 7 5800X

## Building

Requires CMake 3.20+ and Visual Studio 2022 (or any C++17 compiler).

```bash
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

## Project Structure

```
src/core/       # Board representation, move generation, attack tables
src/mcts/       # Monte Carlo Tree Search
src/neural/     # C++ neural network inference
src/selfplay/   # Self-play game generation
training/       # Python/PyTorch training pipeline
tests/          # Google Test suite
docs/           # Technical documentation and plans
```

## Documentation

- [How AlphaZero Engines Work](docs/how-alphazero-engines-work.md) — Technical reference
- [Implementation Plan](docs/superpowers/plans/2026-04-11-chess-engine-core.md) — Phase 1: Engine Core

## Status

Phase 1: Chess Engine Core (in progress)

## License

MIT
