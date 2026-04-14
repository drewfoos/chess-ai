<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.11-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.6-76b900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.6">
  <img src="https://img.shields.io/badge/TensorRT-10.x-76b900?style=flat-square&logo=nvidia&logoColor=white" alt="TensorRT 10">
  <img src="https://img.shields.io/badge/tests-199%20C%2B%2B%20%7C%20146%20Python-brightgreen?style=flat-square" alt="Tests">
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
- SE-ResNet: configurable depth/width (10b×128f baseline, 20b×256f for serious runs), squeeze-and-excitation, Mish activation
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
- AdamW with LR warmup + MultiStepLR decay, grad clipping
- Stochastic Weight Averaging (SWA) for smoother self-play weights
- Mixed-precision training (FP16 via PyTorch AMP)
- Mirror data augmentation (horizontal flip) → 2× training data
- Diff-focus sampling, Q-value blending, opening randomization, Syzygy rescoring (optional)
- Sliding window over recent generations, auxiliary soft-policy target (KataGo)
- **Tiered networks:** optional `network_schedule=[(0, 6, 64), (20, 10, 128)]` style schedule rebuilds the model + optimizer + SWA at tier boundaries; saved architecture is honored on resume.
- **Supervised pretraining (optional cold-start):** Lichess PGN → one-hot shards (phase A) and Stockfish multi-PV distillation → soft-policy shards (phase B). Both feed the same streaming trainer with `--resume-from` so RL picks up from the supervised checkpoint.
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
- **Python 3.11 or 3.12** — recommend [python.org](https://www.python.org/downloads/) installer
- **NVIDIA GPU** with CUDA 12.x drivers (RTX 2000+ for Tensor Cores)
- **~15 GB free disk** (LibTorch ~4 GB, TensorRT ~3 GB, build artifacts ~1 GB)

> ⚠️ **Run all build steps from "x64 Native Tools Command Prompt for VS 2022"**
> (Start menu → Visual Studio 2022 folder). A plain PowerShell, Git Bash, or the
> default *Developer PowerShell* (which targets x86) will fail during CMake's CUDA
> compiler-ID test with `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` or a
> generator-platform mismatch. The x64 Native Tools prompt sets up `vcvars64`
> automatically.

### 2. Clone & Python venv

```bat
git clone <this repo> E:\dev\chess-ai
cd E:\dev\chess-ai
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

:: Torch with matching CUDA build (3 GB download — the default pypi wheel is CPU-only).
:: This MUST come before `pip install -e .` so scikit-build-core sees GPU torch.
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Other Python deps (`numpy`, `python-chess`, `flask`, `onnx`, `pybind11`) install
automatically when you run `pip install -e .` in step 5 — they're declared in
`pyproject.toml`.

### 3. LibTorch (only for the C++-only `build.ps1` workflow)

If you're using the **Python workflow** (`pip install -e .`, step 5 below),
**skip this step** — the PyTorch pip wheel already bundles LibTorch, and
scikit-build-core discovers it automatically from your venv's site-packages.

If you're building via `build.ps1` and don't want the full Python stack,
download the **LibTorch C++ Windows** zip (Release, CUDA 12.6) from
<https://pytorch.org/get-started/locally/> and extract so that `libtorch/`
sits at the repo root:

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

```bat
:: C++ tests (199 total, includes TRT + Syzygy tests when built with those)
:: For the pip/scikit-build path, the build dir is wheel-tagged:
ctest --test-dir build\cp311-cp311-win_amd64 --build-config Release --output-on-failure
:: For the build.ps1 path:
ctest --test-dir build --build-config Release --output-on-failure

:: Python tests (146 total)
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

### Supervised pretraining from Lichess games (optional cold-start boost)

Self-play from random weights is expensive. If you'd rather warm-start the network
from human games first, the pretraining stack does that end-to-end:

- `training/pretrain_dataset.py` — Phase A shard builder (one-hot policy from PGN)
- `training/stockfish_label.py` — Phase B labeler (Stockfish multi-PV soft policy)
- `scripts/phase_b_parallel.py` — orchestrator that fans out N parallel SF workers
- `training/pretrain.py` — streaming supervised trainer (shared by both phases)

**Phase A — one-hot policy from a Lichess PGN dump:**

1. Download a monthly database from <https://database.lichess.org/> (e.g.
   `lichess_db_standard_rated_2026-03.pgn.zst`) and `zstd -d` it. Both
   `.pgn` and `.pgn.zst` are supported as input.
2. Build shards (filters by Elo and time control, encodes 112×8×8 tensors with
   real 8-step history):
   ```bat
   python -m training.pretrain_dataset ^
     --pgn lichess_db_standard_rated_2026-03.pgn ^
     --out-dir pretrain_data\phase_a ^
     --min-elo 2000 --min-base-s 480 ^
     --shard-size 100000 --max-positions 20000000
   ```
3. Train on the shards (streams shard groups to keep peak memory bounded;
   `StreamingShardDataset` dodges the Windows kernel SHM leak that would
   otherwise fire after ~50 iterations with `num_workers>0`):
   ```bat
   python -m training.pretrain ^
     --shard-dir pretrain_data\phase_a ^
     --out checkpoints\phase_a.pt ^
     --blocks 20 --filters 256 --epochs 1 ^
     --batch-size 1024 --lr 1e-3 --warmup-steps 500
   ```
4. Launch self-play RL from the pretrained weights:
   ```bat
   python -m training loop --resume-from checkpoints\phase_a.pt
   ```

**Phase B — Stockfish multi-PV distillation (soft policy targets):**

Labels positions with Stockfish at fixed depth + `multipv=N`, converting cp
scores to a softmax policy and a Lc0-style WDL value. Shards drop into the
same `training.pretrain` entry point via `--soft-policy-weight 0.2` and
`--resume-from <phase_a.pt>`.

On a Ryzen 5800X, 16 SF workers × 1 thread each outperforms 2 threads × 8
workers for `multipv=10` searches (lazy-SMP overhead). The orchestrator
pre-partitions the PGN into disjoint game ranges and handles crash resume by
counting existing shards:

```bat
python scripts/phase_b_parallel.py ^
  --pgn lichess_db_standard_rated_2026-03.pgn ^
  --out-dir pretrain_data\phase_b ^
  --stockfish engines\stockfish\stockfish-windows-x86-64-avx2.exe ^
  --workers 16 --threads 1 --depth 13 --multipv 10 ^
  --positions-per-worker 125000 --games-per-worker 200000

python -m training.pretrain ^
  --shard-dir pretrain_data\phase_b ^
  --out checkpoints\phase_b.pt ^
  --resume-from checkpoints\phase_a.pt ^
  --soft-policy-weight 0.2 ^
  --epochs 1 --batch-size 1024 --lr 5e-4 --warmup-steps 500
```

The Lichess `.pgn` / `.pgn.zst` files and `pretrain_data/` are gitignored.

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

**On Lichess:** clone [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot.git)
into `./lichess-bot/`, put `LICHESS_BOT_TOKEN=...` in `.env`, and launch via
the bundled wrapper, which loads `.env`, prepends `$TENSORRT_PATH\bin` to
`PATH`, and points lichess-bot at a `chess_engine_trt.cmd` shim that invokes
`chess_engine.exe uci_trt <engine.trt>` (lichess-bot's `engine_options` only
accepts `--key=value` flags, so positional args go through the cmd wrapper):

```powershell
.\run_bot.ps1
```

---

## Project Structure

```
src/
  core/          Board representation, types, move generation, attack tables
  mcts/          MCTS search, node arena, game manager (cross-game batching)
  neural/        Position encoder, policy map, NeuralEvaluator, TRTEvaluator
  syzygy/        Fathom WDL wrapper (in-search endgame tablebase probing)
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
  pretrain_dataset.py Lichess PGN -> format-v2 .npz shards (Phase A warm start)
  pretrain.py        Supervised pretraining loop over shard groups
  stockfish_label.py Phase B: Stockfish multi-PV distillation for soft targets

visualization/
  server.py          Flask API + UCI subprocess management
  static/
    index.html       Training dashboard
    play.html        Interactive play page

external/
  fathom/            Vendored Syzygy tablebase probing library (C11)

tests/               199 Google Test cases (neural + TRT + Syzygy tests opt-in at build)
docs/                Architecture, changelog, technical references
build.ps1            One-shot Windows build script
train.py             Interactive training launcher
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` during CUDA compiler-ID test | Shell doesn't have MSVC on `PATH` | Use **x64 Native Tools Command Prompt for VS 2022** (not PowerShell or Developer PowerShell) |
| `generator platform: Win32 Does not match the platform used previously: x64` | Mixing x86 and x64 shells between runs | `rmdir /s /q build` and restart from the x64 Native Tools prompt |
| `#error: "C atomics require C11 or later"` or `"C atomic support is not enabled"` when compiling Fathom | MSVC gates C11 atomics behind `/experimental:c11atomics` | Already handled in `CMakeLists.txt`; ensure you're on MSVC 19.43+ (VS 17.13+) |
| `torch.onnx.OnnxExporterError: Module onnx is not installed` during training tests | `onnx` missing from the venv | `pip install onnx` (now listed in `pyproject.toml`, so fresh installs pick it up) |
| `library kineto not found` CMake warning | Harmless — optional perf profiler, not wired into the build | Ignore |

---

## Documentation

- **[Architecture](docs/architecture.md)** — System design, data flow, component interactions
- **[Changelog](docs/changelog.md)** — Detailed version history
- **[How AlphaZero Engines Work](docs/how-alphazero-engines-work.md)** — Technical reference
- **[Lc0 Optimizations](docs/lc0-optimizations.md)** — Research on optimizations adapted for consumer hardware

---

## License

MIT
