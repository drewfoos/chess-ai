<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.11-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.6-76b900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.6">
  <img src="https://img.shields.io/badge/tests-178%20C%2B%2B%20%7C%20126%20Python-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License">
</p>

# Chess AI

A ground-up AlphaZero-style chess engine — every component written from scratch. No forks, no borrowed code. Bitboard move generation, Monte Carlo Tree Search, SE-ResNet neural network, and a full self-play reinforcement learning pipeline, all built to run on a single consumer GPU.

> **Goal:** Reach 1800-2000+ Elo with code I can fully explain, for a PhD application in machine learning.

---

## Architecture

```
                    Self-Play Loop
            +--------------------------+
            |                          |
            v                          |
  +------------------+    +-------------------+    +------------------+
  |   Neural Network  |    |   MCTS Search     |    |   Game Generator  |
  |   SE-ResNet 10x128|----->  PUCT + Solver   |----->  100 games/gen  |
  |   ~13M params     |    |  Batched Inference |    |  UCI move records |
  +------------------+    +-------------------+    +------------------+
            ^                                              |
            |          +-------------------+               |
            +----------| Training Pipeline |<--------------+
                       |  AdamW + SWA      |
                       |  Mixed Precision  |
                       +-------------------+
```

**Hardware target:** RTX 3080 (10GB VRAM) + Ryzen 7 5800X (8 cores), Windows 11

---

## What's Inside

### Engine Core
- Bitboard board representation with MSVC intrinsics (`__popcnt64`, `_BitScanForward64`)
- Full legal move generation — perft-validated against all standard test positions
- UCI protocol handler with threaded search, time management, and `stop` support
- Compatible with Arena, CuteChess, and any UCI chess GUI

### MCTS Search
- PUCT selection with dynamic c_puct (logarithmic growth, AlphaZero defaults)
- Batched gather/evaluate/scatter loop — single GPU forward pass per iteration
- MCTS-solver: proven wins/losses/draws propagate up the tree
- Smart pruning: stops early when best move has insurmountable visit lead
- KLD-adaptive visit count: scales simulations based on policy divergence
- Shaped Dirichlet noise (KataGo), variance-scaled cPUCT, sibling blending (Ceres)
- Two-fold repetition detection, NN evaluation cache, virtual loss

### Neural Network
- SE-ResNet: 10 residual blocks, 128 filters, squeeze-and-excitation, Mish activation
- Attention policy head: scaled dot-product attention over 64 square tokens (1858-dim output)
- Value head: WDL (win/draw/loss) with softmax
- Moves-left head: auxiliary prediction of remaining plies (Huber loss)
- ~13M parameters, TorchScript export for C++ inference

### Training Pipeline
- Self-play RL loop: generate games -> train network -> export -> repeat
- Policy cross-entropy + value cross-entropy + moves-left Huber loss
- AdamW optimizer with LR warmup (250 steps) + MultiStepLR decay
- Stochastic Weight Averaging for smoother self-play policy
- Mixed precision training (FP16 via PyTorch AMP)
- Mirror data augmentation (horizontal flip) for 2x training data
- Diff-focus sampling: oversamples surprising/informative positions
- Q-value blending, playout cap randomization, opening randomization
- Sliding window over recent generations, configurable batch size
- `--resume-from` any checkpoint to continue training
- Adaptive early-gen settings: auto-tunes sims/moves/games for faster warmup

### Play & Visualization
- **Web play interface** at `/play` — drag-and-drop board, evaluation bar, move history, engine stats
- **Training dashboard** at `/` — loss curves, game replay, speed metrics
- **Lichess bot** integration via lichess-bot for online rated games

---

## Getting Started

### Prerequisites

- CMake 3.20+
- Visual Studio 2022 (or any C++17 compiler)
- Python 3.11+ with PyTorch (CUDA recommended)

### Build

```bash
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

For C++ neural inference (optional, requires [LibTorch](https://pytorch.org/get-started/locally/)):

```bash
cmake -B build -G "Visual Studio 17 2022" -DENABLE_NEURAL=ON -DBUILD_PYTHON=ON -DCMAKE_PREFIX_PATH="path/to/libtorch"
cmake --build build --config Release
```

### Train

```bash
# Start a fresh training run
python -m training loop --games-per-gen 100 --simulations 200 --max-moves 200 --device cuda

# Resume from a checkpoint
python -m training loop --resume-from selfplay_output/checkpoints/model_gen_5.pt --generations 10 --device cuda

# With adaptive early-gen settings (faster warmup)
python -m training loop --adaptive --games-per-gen 100 --simulations 400 --device cuda
```

Checkpoints are saved every generation to `selfplay_output/checkpoints/model_gen_N.pt`.

### Play

**In the browser:**

```bash
python -m visualization.server --metrics-dir selfplay_output/metrics
# Open http://localhost:5050/play
```

**In a chess GUI:**

```bash
# Add as UCI engine in Arena, CuteChess, etc.
chess_engine.exe                                    # Random evaluator
chess_engine.exe uci path/to/model.pt cuda          # Neural network
```

**On Lichess:**

```bash
# 1. Set your bot token in .env (see .env.example)
# 2. Clone lichess-bot: git clone https://github.com/lichess-bot-devs/lichess-bot.git
# 3. pip install -r lichess-bot/requirements.txt
cd lichess-bot && python lichess-bot.py -v
```

---

## Project Structure

```
src/
  core/         Board representation, types, move generation, attack tables
  mcts/         MCTS search, node management, game manager
  neural/       C++ position encoder, policy map, neural evaluator
  uci/          UCI protocol handler, time management
  python/       pybind11 bindings (chess_mcts module)
  main.cpp      CLI entry point (UCI mode, perft, search)

training/
  model.py      SE-ResNet architecture (ChessNetwork)
  encoder.py    Position -> 112x8x8 tensor encoding
  train.py      Training step, optimizer, loss computation
  selfplay.py   Self-play game generation + RL training loop
  mcts.py       Python MCTS (fallback when C++ unavailable)
  dataset.py    ChessDataset with mirror augmentation
  export.py     TorchScript model export
  metrics.py    Per-generation JSON metrics logger

visualization/
  server.py     Flask API + UCI engine subprocess management
  static/
    index.html  Training dashboard (loss curves, game replay)
    play.html   Interactive play page (eval bar, move history)

tests/          178 Google Test cases
docs/           Architecture, changelog, technical references
```

---

## Test Suite

```bash
# C++ tests (178 tests)
ctest --test-dir build --build-config Release --output-on-failure

# Python tests (126 tests)
python -m pytest training/test_training.py -v
```

---

## Implementation Status

| Phase | Description | Status |
|:-----:|-------------|:------:|
| 1 | Engine Core — bitboard, move generation, perft | Done |
| 2 | MCTS Search — PUCT, batching, solver, pruning | Done |
| 3 | Neural Network — SE-ResNet, attention policy, training | Done |
| 4 | Self-Play Pipeline — RL loop, data generation, augmentation | Done |
| 5 | C++ Inference — LibTorch integration, pybind11 bindings | Done |
| 6 | Visualization — training dashboard, play interface, Lichess bot | Done |

---

## Documentation

- **[Architecture](docs/architecture.md)** — System design, data flow, component interactions
- **[Changelog](docs/changelog.md)** — Detailed version history
- **[How AlphaZero Engines Work](docs/how-alphazero-engines-work.md)** — Technical reference for the architecture
- **[Lc0 Optimizations](docs/lc0-optimizations.md)** — Research on optimizations adapted for consumer hardware

---

## License

MIT
