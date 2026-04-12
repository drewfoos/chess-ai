# C++ MCTS Performance Optimizations

Research document analyzing Lc0, Ceres, and general GPU inference best practices to identify concrete optimization opportunities in our C++ MCTS implementation. Focused on GPU utilization, memory layout, threading, batching efficiency, and cache performance.

**Hardware target:** RTX 3080 (10GB VRAM, 8704 CUDA cores, Tensor Cores) + Ryzen 7 5800X (8C/16T)

**Companion document:** [lc0-optimizations.md](lc0-optimizations.md) covers algorithm-level optimizations (policy temperature, MCTS-solver, attention head, etc.). This document covers systems-level C++ performance.

---

## Table of Contents

1. [Current Architecture Summary](#1-current-architecture-summary)
2. [Node Memory Layout](#2-node-memory-layout)
3. [GPU Inference Pipeline](#3-gpu-inference-pipeline)
4. [Position Replay Overhead](#4-position-replay-overhead)
5. [Threading and Parallelism](#5-threading-and-parallelism)
6. [NN Evaluation Cache](#6-nn-evaluation-cache)
7. [Batching Strategy](#7-batching-strategy)
8. [Allocation and Object Lifetime](#8-allocation-and-object-lifetime)
9. [Encoding Overhead](#9-encoding-overhead)
10. [Prioritized Recommendations](#10-prioritized-recommendations)

---

## 1. Current Architecture Summary

### What we have

Our C++ MCTS follows a gather-evaluate-scatter loop:

1. **Gather:** For each leaf slot in the batch, traverse the tree from root to a leaf node using PUCT selection. Apply virtual loss. Build the leaf position by replaying moves from root (`replay_moves`). Check the NN cache. If miss, queue the leaf for evaluation.
2. **Evaluate:** Encode all queued leaf positions into a contiguous float buffer. Call `evaluate_batch_raw()` which creates a `torch::Tensor` via `from_blob()`, transfers to GPU with `.to(device_)`, runs `model_.forward()`, and transfers results back to CPU.
3. **Scatter:** For each evaluated leaf, expand the node (generate legal moves, create children), cache the result, revert virtual loss, and backpropagate.

### Key files

| File | Role | Size |
|------|------|------|
| `src/mcts/search.cpp` | Single-game batched MCTS | ~720 lines |
| `src/mcts/game_manager.cpp` | Multi-game cross-game batching | ~740 lines |
| `src/mcts/node.h` / `node.cpp` | Node struct, children as `vector<unique_ptr<Node>>` | ~115 / ~60 lines |
| `src/neural/neural_evaluator.cpp` | LibTorch TorchScript inference | ~250 lines |
| `src/mcts/nn_cache.h` | Zobrist-keyed `unordered_map<uint64_t, CacheEntry>` | ~50 lines |
| `src/neural/encoder.cpp` | Position -> 112x8x8 float tensor | ~115 lines |

---

## 2. Node Memory Layout

### What Lc0 does

Lc0 underwent a major "node diet" ([issue #13](https://github.com/LeelaChessZero/lc0/issues/13)) that reduced per-node memory from **112 bytes to ~24 bytes** for internal nodes and **~6 bytes for leaf edges**. Key techniques:

- **Edge/Node separation ([PR #145](https://github.com/LeelaChessZero/lc0/pull/145)):** Unexpanded children are stored as lightweight "Edge" structs (just a move + prior probability, ~6 bytes). Only when a child is visited does it get promoted to a full Node. Since typical MCTS trees have ~35x more unexpanded leaves than visited nodes, this saves enormous memory.
- **No board state in nodes:** Nodes do not store a `Position` or `ChessBoard`. The position is reconstructed by replaying moves from root (same as us), or by maintaining a position stack during selection.
- **Children stored as a flat array:** Rather than `vector<unique_ptr<Node>>`, children are stored in a contiguous allocation. The original linked-list design (child + sibling pointers) was replaced with array-based storage, saving 7 bytes per node and improving cache locality.
- **Compact field types:** `uint16_t` for move encoding (2 bytes instead of 4), `uint16_t` for in-flight count, half-precision (FP16) for prior probability.
- **Q stored directly:** Instead of storing `total_value` and computing `Q = total_value / N`, Lc0 stores Q directly and updates it incrementally, saving 4 bytes per node.

**Ceres** (another MCTS chess engine by David Elliott) goes further:
- **Fixed-size, cache-aligned nodes** pre-reserved in a large contiguous array
- **32-bit node indices** instead of 64-bit pointers (halves pointer storage)
- **SIMD-accelerated PUCT selection** using AVX intrinsics across the children array
- **Virtual subtrees** to avoid materializing transposition nodes

### What we do

Our `Node` class (`src/mcts/node.h`) stores:

```cpp
Move move_;                              // 4 bytes (uint16_t internally, but Move is likely 4)
uint16_t prior_bits_;                    // 2 bytes (FP16 prior -- good!)
int visit_count_;                        // 4 bytes
float total_value_;                      // 4 bytes
float sum_sq_value_;                     // 4 bytes
int8_t terminal_status_;                 // 1 byte
int16_t pending_evals_;                  // 2 bytes
Node* parent_;                           // 8 bytes
vector<unique_ptr<Node>> children_;      // 24 bytes (vector overhead on MSVC x64)
// Padding to alignment boundary
// Estimated total: ~56-64 bytes per node
```

**Problems:**

1. **`vector<unique_ptr<Node>>` for children:** Each child is a separate heap allocation. With ~35 legal moves per position, expanding a node triggers ~35 individual `new Node()` calls scattered across the heap. This destroys cache locality during PUCT selection (which iterates all children) and creates allocator pressure.

2. **Every node is a full Node:** Unexpanded leaf edges carry the same 56-64 byte overhead as heavily-visited internal nodes, even though they only need a move + prior (~6 bytes).

3. **`sum_sq_value_` on every node:** Only used for uncertainty estimation on nodes with 2+ visits. Could be stored separately or lazily.

### Recommendations

#### 2a. Edge/Node separation (HIGH IMPACT)

Introduce an `Edge` struct:

```cpp
struct Edge {
    uint16_t move_bits;    // Encoded move (2 bytes)
    uint16_t prior_bits;   // FP16 prior (2 bytes)
    // Total: 4 bytes per edge
};
```

When a node is expanded, allocate a flat `Edge[]` array for all children. Only create a full `Node` when a child is actually visited. This reduces memory for unexpanded children from ~56 bytes to 4 bytes (~14x savings on the majority of the tree).

**Files to change:** `src/mcts/node.h`, `src/mcts/node.cpp`, all files that call `node->child(i)`.

#### 2b. Flat child array with arena allocation (HIGH IMPACT)

Replace `vector<unique_ptr<Node>>` with a contiguous block of child `Edge` structs, plus a parallel array of `Node*` (or indices into an arena) for visited children.

```cpp
class Node {
    Edge* edges_;           // Flat array, allocated once at expansion
    Node** child_nodes_;    // Parallel array, nullptr for unvisited (or use arena index)
    uint16_t num_edges_;
    // ... other fields
};
```

Alternatively, use a node arena (pool allocator) so all Nodes live in a contiguous memory region:

```cpp
class NodeArena {
    std::vector<Node> pool_;  // Contiguous storage
    int next_ = 0;
public:
    Node* allocate() { return &pool_[next_++]; }
};
```

This gives cache-friendly iteration during PUCT selection and eliminates per-node `new`/`delete` overhead.

**Estimated impact:** 2-5x reduction in tree memory, measurable improvement in selection NPS due to cache locality.

#### 2c. Store Q directly instead of total_value (LOW IMPACT)

Replace `total_value_` + division in `mean_value()` with direct Q storage:

```cpp
void update(float value) {
    visit_count_++;
    q_ += (value - q_) / visit_count_;  // Running mean
}
```

Saves 4 bytes if `total_value_` is removed. Minor compute savings (eliminates a division in the hot `select_child` path). However, the running mean is slightly less numerically stable.

---

## 3. GPU Inference Pipeline

### What Lc0 does

Lc0's CUDA backend is heavily optimized:

- **Custom CUDA kernels:** The `cudnn` and `cuda-fp16` backends use hand-written CUDA kernels for residual blocks, SE layers, and attention heads. They do NOT use TorchScript or ONNX -- the entire forward pass is custom CUDA code.
- **FP16 inference with Tensor Cores:** The `cuda-fp16` backend runs the entire network in half precision, exploiting the RTX 3080's Tensor Cores for ~2x throughput over FP32. Benchmarks show ~32,000 NPS with FP16 on similar hardware.
- **Persistent GPU memory:** Weight tensors and intermediate buffers are allocated once at startup and reused across all forward passes. No per-batch allocation.
- **Large batch sizes:** Default `minibatch-size=256` (our default is 16 in SearchParams, 64 at the Python level). Larger batches amortize GPU kernel launch overhead and improve Tensor Core utilization.
- **Async exploration of CUDA streams:** While ultimately not adopted due to diminishing returns, Lc0 explored ([issue #456](https://github.com/LeelaChessZero/lc0/issues/456)) using multiple CUDA streams with event-based completion to overlap compute and data transfer. Benchmarks on RTX 2080 Ti showed 1.02M convolutions/sec at batch=256 with 8 streams.

### What we do

Our `NeuralEvaluator::evaluate_batch_raw()` (`src/neural/neural_evaluator.cpp`):

```cpp
// 1. Encode positions into batch_buffer_ (CPU, contiguous)
for (int b = 0; b < batch_size; b++) {
    std::memcpy(batch_buffer_.data() + b * TENSOR_SIZE, ...);
}

// 2. Create tensor from buffer and transfer to GPU
auto input = torch::from_blob(batch_buffer_.data(), {B, 112, 8, 8}, kFloat32).to(device_);

// 3. Forward pass
auto output = model_.forward({input}).toTuple();

// 4. Transfer results back to CPU
auto policy_logits = output->elements()[0].toTensor().to(torch::kCPU);
auto wdl_probs = output->elements()[1].toTensor().to(torch::kCPU);
```

**Problems:**

1. **Synchronous `.to(device_)` transfer:** `torch::from_blob(...).to(device_)` is a synchronous host-to-device copy that blocks the CPU until the transfer completes. This creates a pipeline bubble where neither CPU nor GPU is working.

2. **FP32 inference:** We run the entire network in FP32. The RTX 3080 has 272 Tensor Cores designed for FP16/TF32 mixed precision. Running FP32 leaves half the GPU's compute throughput on the table.

3. **Per-batch tensor creation:** `torch::from_blob()` creates a new tensor wrapper each call. While cheap, the subsequent `.to(device_)` allocates GPU memory each time (or relies on PyTorch's caching allocator).

4. **Synchronous result transfer:** `.to(torch::kCPU)` blocks until the GPU computation finishes and the data is transferred back.

5. **No CUDA stream overlap:** Encoding the next batch could overlap with the current GPU inference, but everything runs synchronously on the default stream.

### Recommendations

#### 3a. Use pinned (page-locked) memory for host buffers (HIGH IMPACT)

Pinned memory enables asynchronous DMA transfers between CPU and GPU. LibTorch supports this:

```cpp
// At initialization:
auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
input_pinned_ = torch::empty({max_batch, 112, 8, 8}, options);

// At inference time:
// Copy encoded positions directly into pinned tensor
std::memcpy(input_pinned_.data_ptr<float>(), batch_buffer_.data(), ...);

// Non-blocking transfer to GPU
auto input_gpu = input_pinned_.to(device_, /*non_blocking=*/true);
```

Non-blocking transfers allow the CPU to continue with other work (encoding the next batch, running PUCT selection) while the DMA engine handles the transfer.

**Files to change:** `src/neural/neural_evaluator.h` (add pinned tensor member), `src/neural/neural_evaluator.cpp` (use non-blocking transfers).

**Estimated impact:** 10-30% reduction in per-batch latency depending on batch size, by overlapping transfer with computation.

#### 3b. FP16 inference via TorchScript (HIGH IMPACT)

Export the model in FP16 or use `model_.to(torch::kHalf)` at load time:

```cpp
// At initialization:
model_.to(torch::kHalf);
model_.to(device_);

// At inference time:
auto input_fp16 = input_pinned_.to(device_, torch::kHalf, /*non_blocking=*/true);
```

Alternatively, use `torch.cuda.amp.autocast` in the TorchScript export, or convert the model to TensorRT (which Lc0 effectively does with its custom CUDA backend).

**RTX 3080 FP16 throughput:** 29.77 TFLOPS (vs 14.96 TFLOPS FP32). This is a **2x theoretical speedup** on the forward pass.

**Caveat:** Verify that FP16 inference does not degrade policy/value quality. Test by comparing evaluations on a set of benchmark positions. Small SE-ResNets with Mish activation are generally FP16-safe.

**Files to change:** `src/neural/neural_evaluator.cpp` (model loading), `training/export.py` (export in FP16).

**Estimated impact:** Up to 2x inference throughput.

#### 3c. Pre-allocate GPU tensors and reuse buffers (MEDIUM IMPACT)

Instead of creating new tensors each forward pass, pre-allocate input/output GPU tensors at initialization:

```cpp
// At initialization (once):
input_gpu_ = torch::empty({max_batch, 112, 8, 8}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));

// At inference time:
// Copy from pinned memory into pre-allocated GPU tensor
input_gpu_.narrow(0, 0, batch_size).copy_(input_pinned_.narrow(0, 0, batch_size), /*non_blocking=*/true);
```

This avoids repeated GPU memory allocation (even with PyTorch's caching allocator, there's overhead in the allocation path).

**Files to change:** `src/neural/neural_evaluator.h`, `src/neural/neural_evaluator.cpp`.

#### 3d. CUDA stream pipelining (MEDIUM IMPACT, HIGH COMPLEXITY)

Use separate CUDA streams for data transfer and computation:

```cpp
// Stream 1: transfer batch N+1 to GPU
// Stream 2: compute forward pass on batch N
// Stream 1: transfer results of batch N-1 back to CPU
```

This requires managing multiple in-flight batches, which significantly complicates the search loop. Lc0 explored this ([issue #456](https://github.com/LeelaChessZero/lc0/issues/456)) but found diminishing returns because the forward pass dominates latency at large batch sizes. Worth considering only after the simpler optimizations are in place.

**Files to change:** `src/neural/neural_evaluator.cpp` (major refactor).

#### 3e. Consider TensorRT conversion (HIGH IMPACT, HIGH EFFORT)

NVIDIA TensorRT can optimize a TorchScript model by:
- Fusing layers (conv + BN + activation → single kernel)
- Auto-tuning kernel selection for the specific GPU
- Leveraging Tensor Cores with mixed precision

TensorRT typically achieves 2-4x speedup over vanilla TorchScript inference. The `torch2trt` library or `torch.compile` with the TensorRT backend can automate this.

**Estimated impact:** 2-4x inference throughput beyond FP16 alone.

---

## 4. Position Replay Overhead

### What Lc0 does

Lc0 also replays moves from root to reconstruct positions at leaf nodes. However, their Node/Edge separation means they typically don't need to reconstruct positions during PUCT selection -- only at leaf expansion. Their two-fold repetition check uses hash history maintained during selection, not full position reconstruction.

### What we do

In `gather_leaf()` / `gather_leaf_from_game()`, we call `replay_moves()` at multiple points:

1. **Inside the selection loop for two-fold repetition detection:** Every time we descend one level, we call `replay_moves(root_pos, path_moves, child_pos)` to get the position hash. This replays ALL moves from root to the current depth, making this O(d^2) in total for a selection path of depth d.

2. **At the leaf:** Another `replay_moves()` call to get the final leaf position.

3. **Position copies:** `PendingEval` / `PendingLeaf` stores `Position position;` which copies the full Position struct (~320 bytes with bitboards, piece arrays, etc.).

### Recommendations

#### 4a. Incremental position tracking during selection (HIGH IMPACT)

Instead of replaying all moves from root at each level, maintain a running Position during the tree descent:

```cpp
void gather_leaf(Node* root, const PositionHistory& history, vector<PendingEval>& batch) {
    Position pos = history.current();  // Copy once
    UndoInfo undo;
    // ...
    while (!node->is_leaf()) {
        Node* child = select_child_advanced(node, is_root);
        pos.make_move(child->move(), undo);  // O(1) incremental update
        uint64_t hash = compute_hash(pos);    // For two-fold check
        // ...
        node = child;
    }
    // pos is now the leaf position -- no replay needed
}
```

This changes the selection phase from O(d^2) to O(d). For typical MCTS trees with d=20-40, this is a **10-20x speedup** in position reconstruction during selection.

**Files to change:** `src/mcts/search.cpp` (`gather_leaf`), `src/mcts/game_manager.cpp` (`gather_leaf_from_game`).

**Estimated impact:** Significant CPU-side speedup, especially in deep trees. This is likely the single biggest CPU optimization available.

#### 4b. Move leaf position by reference instead of copying (LOW IMPACT)

Currently `PendingEval` and `PendingLeaf` store `Position position;` by value. Since the position is only needed for encoding and expansion (which happens before the next gather pass), we could avoid the copy by using the incrementally-tracked position directly.

However, with incremental tracking (4a), the position is built on the stack and needs to survive until the scatter phase. Storing it in the PendingEval struct by value is actually reasonable. The bigger win is eliminating the O(d^2) replay.

---

## 5. Threading and Parallelism

### What Lc0 does

- **Multiple search worker threads** (default: 2) run `ExecuteOneIteration()` concurrently on a shared tree
- **Mutex-based synchronization** on tree modifications (Lc0 found that >3 threads hits severe mutex contention)
- **Virtual loss** enables concurrent tree traversal -- threads descend different paths
- **Collision detection:** When multiple threads select the same leaf, Lc0 detects this via `is_collision` flags and `max-collision-events` (default: 32) / `max-collision-visits` (default: 9999) parameters. Collisions waste time, so the search cancels early if too many occur.
- **Prefetch mechanism:** When the batch isn't full, Lc0 "prefetches" up to 32 likely-to-be-needed positions into the NN cache speculatively.

### What we do

- **Single-threaded MCTS within each game.** The `Search::run()` and `GameManager::step()` loops are entirely single-threaded.
- **Cross-game parallelism only:** Python `ThreadPoolExecutor` runs 4 games, each with its own `NeuralEvaluator` (NOT thread-safe). The `GameManager` class does cross-game batching but runs on a single thread.
- **No collision detection:** Since we're single-threaded per search, all virtual losses from the same batch can collide (multiple paths converge to the same node), but we don't detect or handle this.

### Recommendations

#### 5a. Multi-threaded gather phase (MEDIUM IMPACT, HIGH COMPLEXITY)

The gather phase (tree traversal + position tracking) is CPU-bound and could run on multiple threads while the GPU processes the previous batch:

```
Thread 1-N: Gather leaves from tree (with virtual loss)
Main thread: Submit previous batch to GPU, receive results, scatter
```

This requires:
- Atomic operations on `visit_count_` and `pending_evals_` (already `int`, easily made `std::atomic<int>`)
- Mutex on node expansion (to prevent two threads from expanding the same node)
- The `NeuralEvaluator` remains single-threaded (only one GPU forward pass at a time)

**Estimated complexity:** High. Lc0's experience suggests diminishing returns beyond 2-3 threads due to lock contention.

#### 5b. Collision detection and handling (MEDIUM IMPACT)

Even with single-threaded gather, when `batch_size=64` and we gather 64 leaves, many may converge due to virtual loss interactions. Detect collisions (multiple leaves selecting the same node) and either:
- Re-select from a different path
- Record the collision and adjust the effective batch size

Lc0's `max-collision-events=32` and `max-collision-visits=9999` parameters control this.

**Files to change:** `src/mcts/search.cpp` (`gather_leaf`), `src/mcts/game_manager.cpp` (`gather_leaf_from_game`).

#### 5c. Overlap gather and GPU inference (HIGH IMPACT, MEDIUM COMPLEXITY)

The most impactful threading optimization is to overlap CPU work with GPU work:

```
Step 1: Gather batch N
Step 2: Submit batch N to GPU (non-blocking) | Start gathering batch N+1
Step 3: Wait for batch N results, scatter   | Continue gathering batch N+1
Step 4: Submit batch N+1 to GPU             | Scatter batch N results
```

This requires:
- One thread for GPU submission/wait (or just using async CUDA operations)
- The main thread continues gathering while GPU computes
- Pinned memory (3a) is a prerequisite for non-blocking transfers

This is the "double buffering" pattern. It can nearly double throughput when CPU gather time ~ GPU inference time.

**Files to change:** `src/neural/neural_evaluator.cpp`, `src/mcts/search.cpp`, `src/mcts/game_manager.cpp`.

---

## 6. NN Evaluation Cache

### What Lc0 does

- **NNCache** with default size of **2,000,000 entries** (vs our 20,000)
- **Cache history length = 7:** The cache key includes the last 7 positions (not just the current one), preventing false hits from transpositions with different history.
- Cache is checked before queuing a position for NN evaluation

### What we do

`NNCache` (`src/mcts/nn_cache.h`):
- `std::unordered_map<uint64_t, CacheEntry>` with max 20,000 entries
- Eviction: remove oldest 25% when full (but `unordered_map` iteration order is arbitrary, not FIFO -- so "oldest 25%" is really "arbitrary 25%")
- `CacheEntry` stores `vector<float> policy` (heap allocation per entry) + `float value` + `int num_moves`

**Problems:**

1. **Tiny cache:** 20,000 entries vs Lc0's 2,000,000. With 800 simulations per move and ~35 legal moves per position, the cache fills fast and useful entries are evicted.

2. **Arbitrary eviction:** `unordered_map::begin()` doesn't give LRU or FIFO order. Evicting from the front of the hash table removes essentially random entries, not the least useful ones.

3. **Heap allocation per entry:** Each `CacheEntry` has a `vector<float> policy` which is a separate heap allocation. With 20,000 entries, that's 20,000 small allocations for policy vectors.

4. **No history in cache key:** We hash only the current position. Two positions reached via different move orders (with different repetition counts or 50-move clock values) could get the same hash, leading to incorrect cached evaluations.

### Recommendations

#### 6a. Increase cache size to 200,000-500,000 (HIGH IMPACT, TRIVIAL)

Simply increase `nn_cache_size` default from 20,000 to at least 200,000. Memory cost: ~200K * (35 floats * 4 bytes + overhead) = ~30-50 MB. Trivial on a system with 16+ GB RAM.

**Files to change:** `src/mcts/search.h` (`SearchParams::nn_cache_size`).

#### 6b. Replace unordered_map with LRU cache (MEDIUM IMPACT)

Implement a proper LRU cache using a hash map + doubly-linked list, or use a simpler bounded hash table with random replacement (like Stockfish's transposition table).

A fixed-size array with open addressing and linear probing would eliminate all per-entry heap allocation:

```cpp
struct CacheSlot {
    uint64_t hash;
    float value;
    float policy[MAX_LEGAL_MOVES]; // Fixed-size, or use a separate policy pool
    uint8_t num_moves;
    bool occupied;
};

class NNCache {
    std::vector<CacheSlot> table_;  // Power-of-2 size, contiguous memory
    // Lookup: table_[hash & (size-1)]
    // Collision: replace (always-replace or depth-preferred)
};
```

**Estimated impact:** Eliminates ~20,000 small heap allocations, improves cache locality for lookups.

#### 6c. Include history hash in cache key (LOW IMPACT, CORRECTNESS)

Hash the last N positions (not just current) into the cache key to avoid false transposition hits. This matches Lc0's `cache-history-length=7`.

---

## 7. Batching Strategy

### What Lc0 does

- **Default minibatch-size = 256** (our SearchParams default is 16)
- **Max-prefetch = 32:** When the batch isn't full, Lc0 speculatively evaluates positions that are likely to be needed next search iteration
- **Multivisit optimization:** A single PUCT descent can claim multiple visits at a leaf (instead of one visit per descent), reducing tree traversal overhead
- **Cross-game batching** for self-play (similar to our GameManager)

### What we do

- `SearchParams::batch_size = 16` (but Python caller may override to 64)
- No prefetch mechanism
- No multivisit -- each `gather_leaf` call claims exactly one visit
- `GameManager::step()` gathers `per_game = batch_size / num_active_games` leaves per game

### Recommendations

#### 7a. Increase default batch_size to 128-256 (HIGH IMPACT, TRIVIAL)

GPU kernel launch overhead is amortized over the batch. The RTX 3080 can handle batch=256 easily within 10GB VRAM for our 10-block/128-filter network.

Benchmark at batch sizes 64, 128, 256, and 512 to find the sweet spot. Lc0's default of 256 is a good starting point for our network size.

**Files to change:** `src/mcts/search.h` (`SearchParams::batch_size`).

#### 7b. Multivisit optimization (MEDIUM IMPACT)

When PUCT selects a leaf, instead of claiming 1 visit, calculate how many visits the leaf would receive before PUCT switches to a different child. Claim all those visits at once:

```cpp
// After selecting best child:
int max_visits = compute_max_visits_before_switch(node, best_child, c_puct);
best_child->apply_virtual_loss(max_visits);  // Apply N virtual losses at once
```

This reduces the number of tree traversals needed to fill a batch. Lc0 implements this as the `maxvisit` field in `NodeToProcess`.

**Estimated impact:** 20-50% reduction in tree traversal overhead for large batch sizes.

#### 7c. Prefetch for cache warming (LOW IMPACT)

When the batch isn't full (e.g., many cache hits or terminals), speculatively evaluate likely-to-be-needed positions. This trades GPU idle time for potential future cache hits.

---

## 8. Allocation and Object Lifetime

### Current allocation hotspots

1. **`Node::add_child()` calls `make_unique<Node>()`:** ~35 heap allocations per node expansion. With batch_size=64, that's up to 64 * 35 = 2,240 allocations per MCTS step.

2. **`vector<float>` in EvalResult and CacheEntry:** Each evaluation creates a `vector<float>` for the policy. Each cache insert copies this vector (another allocation).

3. **`vector<Move>`, `vector<uint64_t>`, `vector<Node*>` in PendingEval:** Selection paths allocate vectors that are used briefly and discarded.

4. **`vector<vector<float>> encode_buffers` in GameManager::step():** Line 606 allocates `batch_size` vectors of 7168 floats each. That's up to 64 * 7168 * 4 = 1.8 MB of heap allocation per step, immediately freed.

### Recommendations

#### 8a. Node arena allocator (HIGH IMPACT)

Pre-allocate a pool of Nodes in contiguous memory:

```cpp
class NodePool {
    std::vector<Node> nodes_;
    size_t next_free_ = 0;
public:
    NodePool(size_t capacity) : nodes_(capacity) {}
    Node* allocate() { return &nodes_[next_free_++]; }
    void reset() { next_free_ = 0; }  // Reuse for next search
};
```

Benefits:
- Eliminates thousands of `new`/`delete` calls per search step
- Contiguous memory improves CPU cache utilization during PUCT selection
- Trivial "deallocation" by resetting the pool between searches

**Files to change:** `src/mcts/node.h`, `src/mcts/node.cpp`, `src/mcts/search.cpp`, `src/mcts/game_manager.cpp`.

#### 8b. Pre-allocate encode buffers (MEDIUM IMPACT)

Replace the per-step allocation in `GameManager::step()`:

```cpp
// Current (line 606):
std::vector<std::vector<float>> encode_buffers(batch_size, std::vector<float>(TENSOR_SIZE));

// Better: class member, allocated once
std::vector<float> encode_pool_;  // batch_size * TENSOR_SIZE contiguous floats

// In constructor:
encode_pool_.resize(max_batch_size * TENSOR_SIZE);

// In step():
float* buf = encode_pool_.data() + b * TENSOR_SIZE;
```

**Files to change:** `src/mcts/game_manager.h`, `src/mcts/game_manager.cpp`.

#### 8c. Use fixed-size arrays for selection path (LOW IMPACT)

Replace `vector<Move>`, `vector<uint64_t>`, `vector<Node*>` in PendingEval with fixed-size arrays (max tree depth is bounded, typically <100):

```cpp
struct PendingEval {
    Node* leaf;
    Position position;
    Move path_moves[MAX_DEPTH];
    uint64_t path_hashes[MAX_DEPTH];
    Node* path_nodes[MAX_DEPTH];
    int path_length;
};
```

Eliminates 3 heap allocations per leaf gathered.

---

## 9. Encoding Overhead

### What we do

`encode_position()` (`src/neural/encoder.cpp`) zeroes 7168 floats, then iterates over all 64 squares for 8 time steps to fill piece planes. For a batch of 64 positions, this encodes 64 * 8 = 512 positions.

The `encode_pieces()` inner loop checks every square:
```cpp
for (int sq = 0; sq < 64; sq++) {
    PieceType pt = pos.piece_on(Square(sq));
    if (pt == NO_PIECE_TYPE) continue;
    // ...
}
```

### Recommendations

#### 9a. Bitboard-based encoding (MEDIUM IMPACT)

Instead of checking all 64 squares, iterate over set bits in piece bitboards:

```cpp
for (int pt = PAWN; pt <= KING; pt++) {
    Bitboard bb = pos.pieces(us, PieceType(pt));
    while (bb) {
        int sq = pop_lsb(bb);
        // Set output[plane * 64 + sq] = 1.0f
    }
}
```

With ~16 pieces per side, this touches ~32 squares per position instead of 64, and avoids the `piece_on()` / `color_on()` lookups.

#### 9b. Write directly into pinned memory (LOW IMPACT)

If we adopt pinned memory (3a), encode positions directly into the pinned tensor buffer instead of going through an intermediate `batch_buffer_` + `memcpy`:

```cpp
float* dest = input_pinned_.data_ptr<float>() + b * TENSOR_SIZE;
encode_position(history, dest);
```

This eliminates one `memcpy` of 7168 * 4 = 28 KB per position.

---

## 10. Prioritized Recommendations

Ranked by expected impact-to-effort ratio for our specific setup (RTX 3080 + Ryzen 7 5800X, batch_size=64, 10-block/128-filter network):

### Tier 1: Do First (Highest ROI)

| # | Optimization | Expected Speedup | Effort | Section |
|---|---|---|---|---|
| 1 | **Incremental position tracking** (eliminate O(d^2) replay) | 10-20x CPU selection speedup | Low | 4a |
| 2 | **FP16 inference** | ~2x GPU throughput | Low-Medium | 3b |
| 3 | **Increase batch_size to 256** | 30-50% GPU utilization improvement | Trivial | 7a |
| 4 | **Increase NN cache to 200K+** | Fewer redundant evaluations | Trivial | 6a |
| 5 | **Pinned memory + non-blocking transfers** | 10-30% latency reduction | Low | 3a |

### Tier 2: Do Next (Strong Impact)

| # | Optimization | Expected Speedup | Effort | Section |
|---|---|---|---|---|
| 6 | **Pre-allocate encode buffers** | Eliminate 1.8 MB/step alloc | Low | 8b |
| 7 | **Edge/Node separation** | ~14x tree memory savings | Medium | 2a |
| 8 | **Node arena allocator** | Eliminate thousands of new/delete | Medium | 8a |
| 9 | **Overlap gather and GPU inference** | Up to 2x throughput | Medium-High | 5c |
| 10 | **Pre-allocate GPU tensors** | Avoid per-batch GPU alloc | Low | 3c |

### Tier 3: Polish (Diminishing Returns)

| # | Optimization | Expected Speedup | Effort | Section |
|---|---|---|---|---|
| 11 | **LRU cache replacement** | Better cache hit rate | Medium | 6b |
| 12 | **Multivisit optimization** | 20-50% less tree traversal | Medium | 7b |
| 13 | **Bitboard-based encoding** | ~2x encoding speed | Low | 9a |
| 14 | **Collision detection** | Avoid wasted batch slots | Medium | 5b |
| 15 | **Fixed-size path arrays** | Eliminate 3 allocs/leaf | Low | 8c |
| 16 | **TensorRT conversion** | 2-4x beyond FP16 | High | 3e |

### Implementation Order

For maximum impact with minimum risk, implement in this order:

1. **4a** (incremental position tracking) -- pure CPU win, no API changes
2. **7a + 6a** (increase batch_size and cache size) -- config changes only
3. **3a + 3b** (pinned memory + FP16) -- GPU pipeline improvements
4. **8b** (pre-allocate encode buffers) -- simple refactor
5. **2a + 8a** (Edge/Node separation + arena) -- major refactor, do together
6. **5c** (double-buffered gather/evaluate) -- requires 3a as prerequisite

---

## References

- [Lc0 GitHub repository](https://github.com/LeelaChessZero/lc0)
- [Technical Explanation of Leela Chess Zero](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/)
- [Lc0 Node memory reduction (issue #13)](https://github.com/LeelaChessZero/lc0/issues/13)
- [Lc0 Edge/Node separation (PR #145)](https://github.com/LeelaChessZero/lc0/pull/145)
- [Lc0 async GPU processing discussion (issue #456)](https://github.com/LeelaChessZero/lc0/issues/456)
- [Lc0 backend configuration](https://lczero.org/blog/2019/04/backend-configuration/)
- [Lc0 engine parameters / FLAGS.md](https://github.com/LeelaChessZero/lc0/blob/master/FLAGS.md)
- [Lc0 options wiki](https://lczero.org/dev/wiki/lc0-options/)
- [Ceres MCTS chess engine](https://github.com/dje-dev/Ceres) -- fixed-size cache-aligned nodes, 32-bit indices, SIMD PUCT selection
- [Ceres on Chessprogramming wiki](https://www.chessprogramming.org/Ceres)
- [Leela Chess Zero on Chessprogramming wiki](https://www.chessprogramming.org/Leela_Chess_Zero)
- [PyTorch CUDA semantics (pinned memory, streams)](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [LibTorch advanced usage (pinned memory, CUDA streams)](https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-3-advanced-libtorch/)
- [LCZero OpenCL vs CUDA vs FP16 benchmarks (Phoronix)](https://www.phoronix.com/news/LCZero-NVIDIA-Benchmarks)
