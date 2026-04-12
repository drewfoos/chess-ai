<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.11-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.6-76b900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.6">
  <img src="https://img.shields.io/badge/TensorRT-10.x-76b900?style=flat-square&logo=nvidia&logoColor=white" alt="TensorRT 10">
  <img src="https://img.shields.io/badge/tests-184%20C%2B%2B%20%7C%20138%20Python-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License">
</p>

# Chess AI

A ground-up AlphaZero-style chess engine — every component written from scratch. No forks, no borrowed code. Bitboard move generation, Monte Carlo Tree Search, SE-ResNet neural network, TensorRT FP16 inference, and a full self-play reinforcement learning pipeline, all tuned for a single consumer GPU.

> **Goal:** Reach 1800–2000+ Elo with code I can fully explain, for a PhD application in machine learning.

---

## Architecture

```
  ======================  PYTHON  =========================================

  +----------------------+         +----------------------+
  |  Training Pipeline   |         |   Self-Play driver   |
  |  PyTorch + AMP FP16  |         |  (play_games_        |
  |  AdamW + SWA         |         |   batched, Python)   |
  |  Sliding window +    |         |                      |
  |  mirror augmentation |         +----------+-----------+
  +----------+-----------+                    |
             ^                                |  pybind11
             |                                |  (chess_mcts._core)
  +----------+-----------+                    v
  |  Training data       |         ======================  C++ / CUDA  ===
  |  packed .npz         |
  |  (104 x uint64       |         +---------------------------------+
  |   bitboards)         |         |           GameManager           |
  +----------+-----------+         |  128 parallel games, shared     |
             ^                     |  NodePool arena, cross-game     |
             |                     |  NN batching                    |
             |                     +----------------+----------------+
  +----------+-----------+                          |
  |  PyTorch .pt  (SWA)  |                          v
  |  export.py           |         +----------------------------------+
  +----------+-----------+         |           MCTS Search            |
             |                     |  PUCT + Solver + Multivisit      |
             |  ONNX opset-18      |  KLD-adaptive, smart pruning,    |
             |  (dynamic batch)    |  shaped Dirichlet, variance-cPUCT|
             v                     +---+---------------------------+--+
  +----------------------+             |                           |
  |  build_trt_engine.py |             |  encode + policy_map      |
  |  FP16, batch profile |             v                           v
  |  (min=1, opt=128,    |     +----------------+       +---------------------+
  |   max=256)           |     |  TRTEvaluator  |       |   NeuralEvaluator   |
  +----------+-----------+     |  enqueueV3     |<----->|   LibTorch FP16     |
             |                 |  pinned host + |       |   (fallback)        |
             v                 |  device bufs   |       |                     |
  +----------------------+     +--------+-------+       +----------+----------+
  |  TensorRT engine     |              |                          |
  |  (.plan)             |--------------+   RawBatchEvaluator      |
  +----------------------+                  (shared interface) <---+
                                                    |
                                                    v
                                   +-----------------------------------+
                                   |  Self-play games → training data  |
                                   |  fed back into the sliding window |
                                   +-----------------------------------+
```

Python owns training, ONNX + TRT engine export, and orchestration.
C++ owns hot-path inference (TRT by default, LibTorch fallback), MCTS search,
and the parallel self-play loop. The two sides talk over pybind11 via the
`chess_mcts._core` module.

**Hardware target:** RTX 3080 (10 GB VRAM) + Ryzen 7 5800X (8 cores / 16 threads), Windows 11.
Smaller / larger GPUs work — just reduce/raise `--parallel-games` and the network tier.

---

## What's Inside

### Engine Core
- Bitboard board representation with MSVC intrinsics (`__popcnt64`, `_BitScanForward64`)
- Full legal move generation — perft-validated against all standard test positions
- UCI protocol handler with threaded search, time management, `stop` support
- Compatible with Arena, CuteChess, lichess-bot, and any UCI GUI

### MCTS Search
- PUCT with dynamic c_puct (AlphaZero log growth), FPU reduction 1.2
- Cross-game batching (GameManager): one GPU forward pass covers 128 parallel games
- Multivisit collapse (up to 8 same-child descents per batch entry)
- MCTS-solver (proven W/L/D propagate up), smart pruning, KLD-adaptive visits
- Shaped Dirichlet noise (KataGo), variance-scaled cPUCT, sibling blending (Ceres)
- Two-fold repetition detection, Zobrist-keyed NN eval cache (200K entries)
- Tree reuse, virtual loss, playout cap randomization

### Neural Network & Inference
- SE-ResNet: 10 residual blocks, 128 filters, squeeze-and-excitation, Mish activation
- Attention policy head: scaled dot-product attention over 64 squares → 1858 moves
- Value head (WDL), moves-left head (Huber auxiliary)
- **Two inference backends**, selected at runtime:
  - **TensorRT FP16** (default on RTX class GPUs) — ONNX opset 18 → TRT engine (.plan) built per generation with batch profile `(1, 128, 256)`; `enqueueV3` with pinned buffers. 2–3× faster than LibTorch FP16 in our benchmarks.
  - **LibTorch FP16** — TorchScript fallback. Used when TensorRT isn't installed.
- Shared `RawBatchEvaluator` interface lets GameManager swap backends transparently.
- Pinned-host + async non-blocking DMA on both backends.

### Training Pipeline
- Self-play RL loop: generate → train → export (TorchScript + ONNX → TRT) → repeat
- Policy cross-entropy + value cross-entropy + moves-left Huber
- AdamW with 250-step LR warmup + MultiStepLR decay, grad clipping
- Stochastic Weight Averaging (SWA) for smoother self-play weights
- Mixed-precision training (FP16 via PyTorch AMP)
- Mirror data augmentation (horizontal flip) → 2× training data
- Diff-focus sampling, Q-value blending, opening randomization, Syzygy rescoring (optional)
- Sliding window over recent generations, auxiliary soft-policy target (KataGo)
- **Tiered networks:** start at 6b64f for fast early generations, scale up to 10b128f once data accumulates. Opt-in via `network_schedule=[(0, 6, 64), (20, 10, 128)]`.
- `--resume-from` any checkpoint; architecture is read from the checkpoint itself (no silent mismatches).

### Play & Visualization
- **Web play interface** at `/play` — drag-and-drop board, eval bar, move history
- **Training dashboard** at `/` — loss curves (Chart.js), game replay (chessboard.js), per-gen network size ("10b128f")
- **Lichess bot** integration via lichess-bot for online rated games

---

## Getting Started

### 1. Prerequisites

- **Windows 11** (instructions below assume PowerShell; WSL/Linux works but isn't documented here)
- **Visual Studio 2022** with the *Desktop development with C++* workload
- **CMake 3.20+** — [cmake.org](https://cmake.org/download/), ensure it's on `PATH`
- **Python 3.11** — recommend [python.org](https://www.python.org/downloads/) installer
- **NVIDIA GPU** with CUDA 12.x drivers (RTX 2000+ for Tensor Cores)
- **~15 GB free disk** (LibTorch ~4 GB, TensorRT ~3 GB, build artifacts ~1 GB)

### 2. Clone & Python deps

```powershell
git clone <this repo> E:\dev\chess-ai
cd E:\dev\chess-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Other Python deps (`numpy`, `python-chess`, `flask`, `pybind11`) are installed
automatically when you run `pip install -e .` in step 5 below — they're
declared in `pyproject.toml` as project dependencies.

### 3. Install LibTorch (required for C++ neural inference)

Download the **LibTorch C++ Windows** zip (Release, CUDA 12.6) from
<https://pytorch.org/get-started/locally/> and extract so that `libtorch/` sits at the repo root:

```
E:\dev\chess-ai\libtorch\
    bin\  include\  lib\  share\  ...
```

(Alternatively, set `$env:LIBTORCH_DIR` or pass `-LibTorchDir` to `build.ps1`.)

### 4. Install TensorRT (optional but recommended on RTX)

1. Download **TensorRT 10.x ZIP for Windows + CUDA 12.x** from
   <https://developer.nvidia.com/tensorrt-download> (free NVIDIA dev account required).
2. Extract to e.g. `C:\TensorRT-10.16.1.11`.
3. Install the Python wheel that matches your Python version from the SDK:
   ```powershell
   pip install C:\TensorRT-10.16.1.11\python\tensorrt-10.16.1.11-cp311-none-win_amd64.whl
   ```
4. Set the env var so both the build script and runtime find the DLLs:
   ```powershell
   [Environment]::SetEnvironmentVariable('TENSORRT_PATH','C:\TensorRT-10.16.1.11','User')
   # restart the shell afterwards, or set it for this session:
   $env:TENSORRT_PATH = 'C:\TensorRT-10.16.1.11'
   ```

`build.ps1` auto-enables TensorRT whenever `TENSORRT_PATH` is set; pass `-NoTensorRT` to skip.

### 5. Build

**Python workflow (recommended — installs `chess_mcts` properly so
`import chess_mcts` works from any shell / CWD):**

```powershell
# One-time editable install — TensorRT on by default when TENSORRT_PATH is set.
pip install -e .

# Without TensorRT (LibTorch-only):
pip install -e . --config-settings=cmake.define.ENABLE_TENSORRT=OFF
```

scikit-build-core handles CMake invocation, bakes LibTorch + TensorRT DLL
paths into the installed package, and auto-rebuilds on C++ source change
when you re-import.

**C++-only workflow (`chess_engine.exe` + `ctest`), via `build.ps1`:**

```powershell
# Full build: engine + LibTorch + pybind11 + TensorRT (if TENSORRT_PATH is set)
.\build.ps1

# Clean reconfigure
.\build.ps1 -Clean

# Engine only (skip neural backends)
.\build.ps1 -NoNeural

# Explicitly disable TensorRT even if SDK is installed
.\build.ps1 -NoTensorRT

# Explicit paths if env vars aren't set
.\build.ps1 -LibTorchDir C:\libs\libtorch -TensorRTDir C:\TensorRT-10.16.1.11
```

`build.ps1` wraps:
```
cmake -B build -G "Visual Studio 17 2022" -A x64 \
      -DENABLE_NEURAL=ON -DBUILD_PYTHON=ON -DENABLE_TENSORRT=ON \
      -DCMAKE_PREFIX_PATH=<libtorch> -DTENSORRT_ROOT=<trt>
cmake --build build --config Release --parallel
```

### 6. Test

```powershell
# C++ tests (184 total, includes TRT tests when built with TensorRT)
ctest --test-dir build --build-config Release --output-on-failure

# Python tests (138 total)
python -m pytest training/test_training.py -v
```

---

## Train

Defaults below are tuned for an RTX 3080 with TensorRT installed.

```powershell
# Interactive launcher (prompts with smart defaults)
python train.py

# Non-interactive CLI
python -m training loop `
  --generations 50 `
  --games-per-gen 400 `
  --simulations 400 `
  --parallel-games 128 `
  --batch-size 2048 `
  --device cuda
  # TensorRT is on by default; pass --no-use-trt to use LibTorch FP16 instead.

# Resume
python -m training loop --resume-from selfplay_output/checkpoints/model_gen_5.pt
```

Throughput reference (RTX 3080, 10b128f net, TRT FP16, 400 sims, 128 parallel games):
roughly 2–3× the LibTorch-FP16 baseline. Checkpoints land in
`selfplay_output/checkpoints/model_gen_N.pt`; ONNX and TRT artifacts live alongside
`cpp_model.pt` and are rebuilt automatically each generation.

### Tiered networks (recommended for cold starts)

Small net early → large net once the replay buffer has material. Via `train.py`:

```
  Enable tiered-net schedule? [y/N]: y
    Initial blocks:   6
    Initial filters:  64
    Scale-up at generation: 20
    Final blocks:     10
    Final filters:    128
```

Or programmatically: `training_loop(network_schedule=[(0, 6, 64), (20, 10, 128)], ...)`.

---

## Play

**In the browser:**

```powershell
python -m visualization.server --metrics-dir selfplay_output/metrics
# http://localhost:5050/        dashboard
# http://localhost:5050/play    play against the engine
```

**In a chess GUI (Arena, CuteChess, BanksiaGUI):**

```powershell
# Add one of these as a UCI engine. No args = plain random-eval UCI.
build\Release\chess_engine.exe
build\Release\chess_engine.exe uci     <model.pt>  cuda    # LibTorch backend
build\Release\chess_engine.exe uci_trt <engine.trt>        # TensorRT backend
```

**On Lichess:** clone [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot.git),
set your token in `.env`, point its config at the `chess_engine.exe` above, and run
`python lichess-bot.py -v`.

---

## Project Structure

```
src/
  core/          Board representation, types, move generation, attack tables
  mcts/          MCTS search, node arena, game manager (cross-game batching)
  neural/        Position encoder, policy map, NeuralEvaluator, TRTEvaluator
  uci/           UCI protocol handler, time management
  python/
    bindings.cpp   pybind11 entry -> chess_mcts._core
    chess_mcts/    Python package (DLL registration + re-exports)
  main.cpp       CLI entry point (UCI modes, perft, search, search_nn[_trt])

pyproject.toml  scikit-build-core backend + project metadata (pip install -e .)

training/
  model.py           SE-ResNet (ChessNetwork)
  encoder.py         Position -> 112x8x8 tensor
  train.py           Training step, optimizer, loss
  selfplay.py        Self-play game generation + RL training loop (TRT pipeline)
  mcts.py            Python MCTS (fallback when C++ unavailable)
  dataset.py         Mirror augmentation, diff-focus sampling
  export.py          TorchScript + ONNX export
  build_trt_engine.py ONNX -> TensorRT FP16 engine builder
  metrics.py         Per-generation JSON metrics logger

visualization/
  server.py          Flask API + UCI subprocess management
  static/
    index.html       Training dashboard
    play.html        Interactive play page

tests/               184 Google Test cases (neural + TRT tests opt-in at build)
docs/                Architecture, changelog, technical references
build.ps1            One-shot Windows build script
train.py             Interactive training launcher
```

---

## Implementation Status

| Phase | Description | Status |
|:-----:|-------------|:------:|
| 1 | Engine Core — bitboard, move generation, perft | Done |
| 2 | MCTS Search — PUCT, batching, solver, pruning | Done |
| 3 | Neural Network — SE-ResNet, attention policy, training | Done |
| 4 | Self-Play Pipeline — RL loop, data generation, augmentation | Done |
| 5 | C++ Inference — LibTorch + TensorRT, pybind11 bindings | Done |
| 6 | Visualization — training dashboard, play interface, Lichess bot | Done |

---

## Documentation

- **[Architecture](docs/architecture.md)** — System design, data flow, component interactions
- **[Changelog](docs/changelog.md)** — Detailed version history
- **[How AlphaZero Engines Work](docs/how-alphazero-engines-work.md)** — Technical reference
- **[Lc0 Optimizations](docs/lc0-optimizations.md)** — Research on optimizations adapted for consumer hardware

---

## License

MIT
