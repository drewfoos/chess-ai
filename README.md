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

An AlphaZero-style chess engine built from scratch on a single consumer GPU — C++ MCTS, PyTorch training, TensorRT inference, and a full self-play reinforcement learning loop. No forks of Leela or Stockfish; every component (bitboard move generation, MCTS, SE-ResNet, self-play pipeline) is original.

---

## 1. What It Is

A complete, self-contained chess system:

- **Engine** (C++17): bitboard move generation, MCTS with PUCT, UCI protocol
- **Network** (PyTorch): SE-ResNet with attention policy + WDL value + moves-left heads
- **Inference** (C++): TensorRT FP16 (default) or LibTorch FP16 fallback, swappable behind one interface
- **Training** (Python): supervised pretraining (Lichess PGN + Stockfish distillation) → self-play RL
- **Play**: web UI, any UCI GUI (Arena/CuteChess), or Lichess bot

Everything targets a single RTX 3080 + Ryzen 7 5800X. Smaller or larger GPUs work by adjusting `--parallel-games` and the network tier.

---

## 2. Why It Matters

The project exists to demonstrate technical depth across three fields — reinforcement learning, neural networks, and chess engine internals — in code the author can fully explain, for a PhD application.

The challenge isn't "build a strong engine" — Stockfish and Leela are open source. The challenge is **reaching a strong Elo on one consumer GPU** instead of a distributed cluster. Pure-Zero from random weights would take months of wall-clock on a 3080; Leela only got there with a volunteer GPU farm. The hybrid supervised-pretrain → self-play arc used here turns that into a weeks-scale project while keeping the AlphaZero learning dynamic intact for the final phase.

Goal: **1800–2000+ Elo** in a system the author can defend line-by-line.

---

## 3. Core Architecture

```
  ======================  PYTHON  =========================================

  +----------------------+         +----------------------+
  |  Training            |         |  Self-Play driver    |
  |  PyTorch + AMP FP16  |         |  (play_games_        |
  |  AdamW + SWA         |         |   batched)           |
  |  Mirror augmentation |         +----------+-----------+
  +----------+-----------+                    |
             ^                                |  pybind11
             |                                |  (chess_mcts._core)
  +----------+-----------+                    v
  |  Training data       |         ======================  C++ / CUDA  ===
  |  (packed .npz:       |
  |   104 x uint64       |         +---------------------------------+
  |   bitboards)         |         |           GameManager           |
  +----------+-----------+         |  N parallel games, shared       |
             ^                     |  NodePool arena, cross-game     |
             |                     |  NN batching                    |
             |                     +----------------+----------------+
  +----------+-----------+                          |
  |  PyTorch .pt  (SWA)  |                          v
  +----------+-----------+         +----------------------------------+
             |                     |           MCTS Search            |
             | ONNX opset-18       |  PUCT + Solver + Multivisit      |
             | (dynamic batch)     |  KLD-adaptive, shaped Dirichlet, |
             v                     |  variance-cPUCT, smart pruning   |
  +----------------------+         +---+---------------------------+--+
  |  build_trt_engine.py |             |                           |
  |  FP16, batch profile |             v                           v
  |  (1, 128, 256)       |     +----------------+       +---------------------+
  +----------+-----------+     |  TRTEvaluator  |<----->|   NeuralEvaluator   |
             |                 |  enqueueV3     |       |   LibTorch FP16     |
             v                 |  pinned bufs   |       |   (fallback)        |
  +----------------------+     +--------+-------+       +----------+----------+
  |  TensorRT engine     |              |                          |
  |  (.plan)             |--------------+   RawBatchEvaluator <----+
  +----------------------+                  (shared interface)
```

**Engine core** — bitboard representation with MSVC intrinsics, perft-validated legal move generation, UCI protocol handler with threaded search and `stop` support.

**MCTS** — PUCT selection with dynamic `c_puct`, MCTS-solver (proven W/L/D propagates), virtual loss, KataGo-style shaped Dirichlet noise, variance-scaled cPUCT, KLD-adaptive visit counts, smart pruning, tree reuse, Zobrist-keyed NN cache. A `GameManager` drives N concurrent games and batches their leaves into one GPU forward pass per step; a **multivisit** optimization collapses up to 8 same-child descents into a single evaluation when PUCT stays stable.

**Network** — configurable SE-ResNet (10b×128f baseline, 20b×256f for serious runs), squeeze-and-excitation blocks, Mish activation, attention-based policy head over 64 square tokens producing 1858 move logits, WDL value head, auxiliary moves-left Huber head. Tiered-network schedule can start at 6b×64f and scale up at a configured generation.

**Inference** — `RawBatchEvaluator` is an abstract interface with two implementations: `TRTEvaluator` (default) using `enqueueV3` against a TRT engine built with dynamic batch profile `(1, 128, 256)`, and `NeuralEvaluator` using LibTorch FP16 as a fallback. Both use pinned host memory and async non-blocking DMA. The engine is rebuilt (or weight-refit) each generation; `BuilderFlag.REFIT` lets weight swaps finish in 5–10s vs a 30–90s full rebuild.

**Tablebase** — optional in-search Syzygy probing via vendored Fathom; MCTS leaves with `piece_count ≤ TB_LARGEST` resolve exactly from WDL and skip NN evaluation entirely.

---

## 4. Training Pipeline

```
  Phase A (hours)          Phase B (~6h)              Phase C (days)
  +------------------+     +-------------------+      +------------------+
  | Lichess PGN      |     | Phase A checkpoint|      | Phase B checkpt  |
  | ~20M positions   |     | + 2M SF-labeled   |      | + self-play RL   |
  | one-hot policy   | --> | soft-policy       | -->  | (MCTS generates  |
  | WDL from result  |     | WDL from cp score |      |  own targets)    |
  +------------------+     +-------------------+      +------------------+
  ~1600 Elo baseline       sharper policy signal      AlphaZero discovery
```

**Phase A — human-game warm start.** `training/pretrain_dataset.py` streams a monthly Lichess PGN dump, filters by Elo + time control + game result, and writes format-v2 `.npz` shards (112×8×8 planes packed as 104×uint64 bitboards). `training/pretrain.py` trains over them with a one-hot policy target. A `StreamingShardDataset` (IterableDataset) feeds one persistent DataLoader for the whole run, which sidesteps a Windows kernel shared-memory leak that otherwise fires after ~50 shard-group iterations.

**Phase B — Stockfish distillation.** `training/stockfish_label.py` drives a Stockfish UCI subprocess at fixed depth with `multipv=N` per sampled position; `softmax(cp / T)` over the top-N PV moves becomes a soft policy target, and the best-move cp maps to a Lc0-style WDL value. `scripts/phase_b_parallel.py` fans out 16 SF workers over disjoint PGN game ranges (1 thread/worker outperforms 2/worker on `multipv=10` because SF lazy-SMP overhead dominates at that breadth). Resume works by counting shard files on disk.

**Phase C — self-play RL.** The generate → train → export → repeat loop, resumed from Phase B weights. Each generation: `GameManager` plays ~400 games with ~400 MCTS sims each, PyTorch trains on a sliding window of the last N generations, and the new model is re-exported to TorchScript + ONNX + TRT before the next round. AdamW with LR warmup + MultiStepLR decay, mixed-precision FP16 (AMP), SWA for smoother weights, mirror augmentation (2× effective data), diff-focus sampling (oversamples surprising positions), Q-value blending, optional Syzygy rescoring for endgames. KataGo-style auxiliary soft-policy target keeps the net learning non-obvious moves.

`--resume-from` works across all three phases and reads architecture from the checkpoint itself — no silent `--blocks`/`--filters` mismatches.

---

## 5. Results & Benchmarks

Ongoing training run (20 blocks × 256 filters, ~25M parameters):

| Checkpoint | Source | Approx. Elo | Notes |
|---|---|---|---|
| Phase A | SL on 20M Lichess positions (min-Elo 2000) | ~1600–1700 | Stockfish gauntlet: 9-0-1 vs SF lvl 1, 6-2-2 vs SF lvl 2, 2-2-6 vs SF lvl 3 |
| Phase B | Phase A + 2M SF-distilled positions (depth 13, multipv 10) | in progress | Expected uplift from sharper policy signal |
| Phase C | Phase B + self-play RL | pending | — |

**Infrastructure benchmarks (RTX 3080 + Ryzen 7 5800X):**

- Perft from starting position: 4,865,609 nodes at depth 5 in **0.18s** (~27M nodes/sec)
- Phase A pretraining: 20M positions × 1 epoch at batch 1024 completed in ~3h
- Phase B labeling: 2M positions with SF depth 13 / multipv 10 in **~6.4h** (16 workers × 1 thread)
- Self-play throughput with TRT FP16 at 10b×128f / 400 sims / 128 parallel games: **~2–3×** the LibTorch-FP16 baseline
- TRT weight refit: ~5–10s vs 30–90s full rebuild per generation

**Test suite:** 199 Google Test cases (C++), 146 pytest cases (Python), all passing. Perft-validated against all 5 standard positions.

---

## 6. How to Run

### Prerequisites

Windows 11, Visual Studio 2022 with the *Desktop development with C++* workload, CMake 3.20+, Python 3.11 or 3.12, NVIDIA GPU with CUDA 12.x drivers, ~15 GB free disk.

> Run all build steps from **x64 Native Tools Command Prompt for VS 2022**. Plain PowerShell or the default Developer PowerShell (which targets x86) will fail during CMake's CUDA compiler-ID test.

### Install

```powershell
git clone <this repo> E:\dev\chess-ai
cd E:\dev\chess-ai
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip

# Torch with matching CUDA build MUST come before `pip install -e .`
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Optional: TensorRT (recommended on RTX). Download from
# https://developer.nvidia.com/tensorrt-download, extract, and:
$env:TENSORRT_PATH = 'C:\TensorRT-10.16.1.11'
pip install $env:TENSORRT_PATH\python\tensorrt-10.16.1.11-cp311-none-win_amd64.whl

# Editable install — scikit-build-core invokes CMake, bakes LibTorch + TRT
# DLL paths into the installed package, and auto-rebuilds on C++ change.
pip install -e .

# Without TensorRT:
# pip install -e . --config-settings=cmake.define.ENABLE_TENSORRT=OFF
```

### Test

```powershell
ctest --test-dir build\cp311-cp311-win_amd64 --build-config Release --output-on-failure
python -m pytest training/test_training.py -v
```

### Train

```powershell
# Phase A: Lichess PGN → one-hot shards → supervised pretraining
python -m training.pretrain_dataset `
  --pgn lichess_db_standard_rated_2026-03.pgn `
  --out-dir pretrain_data\phase_a `
  --min-elo 2000 --min-base-s 480 `
  --shard-size 100000 --max-positions 20000000

python -m training.pretrain `
  --shard-dir pretrain_data\phase_a `
  --out checkpoints\phase_a.pt `
  --blocks 20 --filters 256 --epochs 1 `
  --batch-size 1024 --lr 1e-3 --warmup-steps 500

# Phase B: Stockfish multipv distillation (16 parallel workers)
python scripts\phase_b_parallel.py `
  --pgn lichess_db_standard_rated_2026-03.pgn `
  --out-dir pretrain_data\phase_b `
  --stockfish engines\stockfish\stockfish-windows-x86-64-avx2.exe `
  --workers 16 --threads 1 --depth 13 --multipv 10 `
  --positions-per-worker 125000 --games-per-worker 200000

python -m training.pretrain `
  --shard-dir pretrain_data\phase_b `
  --out checkpoints\phase_b.pt `
  --resume-from checkpoints\phase_a.pt `
  --soft-policy-weight 0.2 --epochs 1 --batch-size 1024 --lr 5e-4

# Phase C: self-play RL (interactive launcher with smart defaults)
python train.py
# or non-interactively:
python -m training loop `
  --resume-from checkpoints\phase_b.pt `
  --generations 100 --games-per-gen 400 --simulations 400 `
  --parallel-games 128 --batch-size 2048
```

### Play

```powershell
# Web dashboard + play page
python -m visualization.server --metrics-dir selfplay_output\metrics
# http://localhost:5050/        training dashboard
# http://localhost:5050/play    play against the engine

# In a UCI GUI (Arena, CuteChess, BanksiaGUI)
build\cp311-cp311-win_amd64\Release\chess_engine.exe uci_trt <engine.trt>

# On Lichess (requires lichess-bot/ cloned next to the repo and
# LICHESS_BOT_TOKEN set in .env):
.\run_bot.ps1
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` | Shell has no MSVC on `PATH` | Use **x64 Native Tools Command Prompt for VS 2022** |
| `generator platform: Win32 Does not match the platform used previously: x64` | Mixing x86/x64 shells across runs | `rmdir /s /q build` and restart from the x64 Native Tools prompt |
| `#error: "C atomics require C11 or later"` when compiling Fathom | MSVC gates C11 atomics behind `/experimental:c11atomics` | Handled in `CMakeLists.txt`; ensure MSVC 19.43+ (VS 17.13+) |
| `OSError: [WinError 1450] Insufficient system resources` during pretrain | Windows kernel SHM leak from repeated DataLoader spawns | Already fixed via `StreamingShardDataset` — make sure you're running current `training/pretrain.py` |
| `ImportError: DLL load failed` when `import chess_mcts` | `pip install -e .` not run, or `TENSORRT_PATH` unset at install time | Re-run `pip install -e .` with `TENSORRT_PATH` set |

---

## Documentation

- **[Architecture](docs/architecture.md)** — component-level design and data flow
- **[Changelog](docs/changelog.md)** — detailed version history
- **[How AlphaZero Engines Work](docs/how-alphazero-engines-work.md)** — technical primer
- **[Lc0 Optimizations](docs/lc0-optimizations.md)** — research on optimizations adapted for consumer hardware

---

## License

MIT
