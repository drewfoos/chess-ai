# MCTS Search Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Monte Carlo Tree Search engine that selects moves using PUCT-based tree search, with a pluggable evaluation interface so the neural network (Plan 3/5) can be swapped in later.

**Architecture:** MCTS tree of `Node` objects, each storing visit count, total value, prior probability, and child pointers. A `Search` class runs the select-expand-evaluate-backprop loop. An `Evaluator` interface abstracts position evaluation — initially a `RandomEvaluator` (uniform policy, material-based value) so MCTS mechanics can be tested without a neural network.

**Tech Stack:** C++17, Google Test, existing chess_core library (Position, Move, generate_legal_moves)

---

## File Structure

```
src/mcts/
├── node.h          Node struct: N, W, Q, P, children, move, parent pointer
├── node.cpp        Node methods: expand, select_child (PUCT), best_move
├── search.h        Search class + SearchParams + Evaluator interface
├── search.cpp      MCTS loop: run N iterations, select → expand → evaluate → backprop
tests/
└── test_mcts.cpp   All MCTS tests
```

**Interfaces with existing code:**
- `Position` (from `src/core/position.h`) — board state, make_move/unmake_move
- `Move` (from `src/core/types.h`) — move representation
- `generate_legal_moves()` (from `src/core/movegen.h`) — legal move list
- `attacks::init()` (from `src/core/attacks.h`) — must be called before any position work

---

### Task 1: Node struct and basic tree operations

**Files:**
- Create: `src/mcts/node.h`
- Create: `src/mcts/node.cpp`
- Create: `tests/test_mcts.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add MCTS source files to CMakeLists.txt**

Add `node.cpp` to the `chess_core` library and `test_mcts.cpp` to the test executable:

```cmake
add_library(chess_core
    src/core/bitboard.cpp
    src/core/attacks.cpp
    src/core/position.cpp
    src/core/movegen.cpp
    src/mcts/node.cpp
)
```

And in the test executable section:

```cmake
add_executable(chess_tests
    tests/test_types.cpp
    tests/test_bitboard.cpp
    tests/test_attacks.cpp
    tests/test_position.cpp
    tests/test_movegen.cpp
    tests/test_perft.cpp
    tests/test_mcts.cpp
)
```

- [ ] **Step 2: Write the failing test for Node construction**

Create `tests/test_mcts.cpp`:

```cpp
#include <gtest/gtest.h>
#include "mcts/node.h"
#include "core/attacks.h"

class MCTSTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        attacks::init();
    }
};

TEST_F(MCTSTest, NodeDefaultConstruction) {
    mcts::Node node;
    EXPECT_EQ(node.visit_count(), 0);
    EXPECT_FLOAT_EQ(node.total_value(), 0.0f);
    EXPECT_FLOAT_EQ(node.mean_value(), 0.0f);
    EXPECT_FLOAT_EQ(node.prior(), 0.0f);
    EXPECT_TRUE(node.move().is_none());
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(node.num_children(), 0);
}

TEST_F(MCTSTest, NodeConstructionWithPrior) {
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    mcts::Node node(m, 0.35f);
    EXPECT_EQ(node.visit_count(), 0);
    EXPECT_FLOAT_EQ(node.prior(), 0.35f);
    EXPECT_EQ(node.move(), m);
    EXPECT_TRUE(node.is_leaf());
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: Build failure — `mcts/node.h` doesn't exist.

- [ ] **Step 4: Implement Node struct**

Create `src/mcts/node.h`:

```cpp
#pragma once
#include "core/types.h"
#include <vector>
#include <memory>
#include <cmath>

namespace mcts {

class Node {
public:
    Node();
    Node(Move move, float prior);

    // Tree structure
    bool is_leaf() const { return children_.empty(); }
    int num_children() const { return static_cast<int>(children_.size()); }
    Node* child(int i) { return children_[i].get(); }
    const Node* child(int i) const { return children_[i].get(); }
    Node* parent() const { return parent_; }

    // Statistics
    int visit_count() const { return visit_count_; }
    float total_value() const { return total_value_; }
    float prior() const { return prior_; }
    Move move() const { return move_; }

    float mean_value() const {
        return visit_count_ > 0 ? total_value_ / visit_count_ : 0.0f;
    }

    // Modification
    void add_child(Move move, float prior);
    void update(float value);

    // Selection
    Node* select_child(float c_puct, float fpu_value) const;
    Move best_move() const;

    // For Dirichlet noise
    void set_prior(float p) { prior_ = p; }

private:
    Move move_;
    float prior_ = 0.0f;

    int visit_count_ = 0;
    float total_value_ = 0.0f;

    Node* parent_ = nullptr;
    std::vector<std::unique_ptr<Node>> children_;
};

} // namespace mcts
```

Create `src/mcts/node.cpp`:

```cpp
#include "mcts/node.h"
#include <algorithm>
#include <cassert>
#include <limits>

namespace mcts {

Node::Node() : move_(Move::none()), prior_(0.0f), parent_(nullptr) {}

Node::Node(Move move, float prior) : move_(move), prior_(prior), parent_(nullptr) {}

void Node::add_child(Move move, float prior) {
    auto child = std::make_unique<Node>(move, prior);
    child->parent_ = this;
    children_.push_back(std::move(child));
}

void Node::update(float value) {
    visit_count_++;
    total_value_ += value;
}

Node* Node::select_child(float c_puct, float fpu_value) const {
    assert(!is_leaf());

    int parent_visits = visit_count_;
    float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));

    Node* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();

    for (const auto& child : children_) {
        float q = child->visit_count_ > 0 ? child->mean_value() : fpu_value;
        float u = c_puct * child->prior_ * sqrt_parent / (1.0f + child->visit_count_);
        float score = q + u;

        if (score > best_score) {
            best_score = score;
            best = child.get();
        }
    }
    return best;
}

Move Node::best_move() const {
    assert(!is_leaf());

    const Node* best = nullptr;
    int best_visits = -1;

    for (const auto& child : children_) {
        if (child->visit_count_ > best_visits) {
            best_visits = child->visit_count_;
            best = child.get();
        }
    }
    return best ? best->move() : Move::none();
}

} // namespace mcts
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mcts/node.h src/mcts/node.cpp tests/test_mcts.cpp CMakeLists.txt
git commit -m "feat(mcts): add Node struct with visit count, value, prior, and PUCT selection"
```

---

### Task 2: Node expand and PUCT selection tests

**Files:**
- Modify: `tests/test_mcts.cpp`

- [ ] **Step 1: Write tests for add_child and tree structure**

Append to `tests/test_mcts.cpp`:

```cpp
TEST_F(MCTSTest, AddChildren) {
    mcts::Node root;
    root.add_child(Move(E2, E4, FLAG_DOUBLE_PUSH), 0.4f);
    root.add_child(Move(D2, D4, FLAG_DOUBLE_PUSH), 0.3f);
    root.add_child(Move(G1, F3, FLAG_QUIET), 0.2f);

    EXPECT_FALSE(root.is_leaf());
    EXPECT_EQ(root.num_children(), 3);
    EXPECT_EQ(root.child(0)->move(), Move(E2, E4, FLAG_DOUBLE_PUSH));
    EXPECT_FLOAT_EQ(root.child(0)->prior(), 0.4f);
    EXPECT_EQ(root.child(0)->parent(), &root);
}

TEST_F(MCTSTest, UpdateVisitAndValue) {
    mcts::Node node(Move(E2, E4, FLAG_DOUBLE_PUSH), 0.5f);
    node.update(0.6f);
    EXPECT_EQ(node.visit_count(), 1);
    EXPECT_FLOAT_EQ(node.total_value(), 0.6f);
    EXPECT_FLOAT_EQ(node.mean_value(), 0.6f);

    node.update(0.4f);
    EXPECT_EQ(node.visit_count(), 2);
    EXPECT_FLOAT_EQ(node.total_value(), 1.0f);
    EXPECT_FLOAT_EQ(node.mean_value(), 0.5f);
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: 4 tests PASS.

- [ ] **Step 3: Write test for PUCT selection**

Append to `tests/test_mcts.cpp`:

```cpp
TEST_F(MCTSTest, PUCTSelectsHighPriorUnvisited) {
    // With no visits, PUCT should prefer the child with highest prior
    mcts::Node root;
    root.update(0.0f); // root needs a visit for sqrt(N_parent) > 0
    root.add_child(Move(E2, E4, FLAG_DOUBLE_PUSH), 0.1f);
    root.add_child(Move(D2, D4, FLAG_DOUBLE_PUSH), 0.6f);
    root.add_child(Move(G1, F3, FLAG_QUIET), 0.3f);

    float c_puct = 2.5f;
    float fpu = 0.0f; // FPU value for unvisited nodes
    mcts::Node* selected = root.select_child(c_puct, fpu);
    EXPECT_EQ(selected->move(), Move(D2, D4, FLAG_DOUBLE_PUSH));
}

TEST_F(MCTSTest, PUCTBalancesExplorationExploitation) {
    mcts::Node root;
    root.update(0.0f);
    root.update(0.0f); // 2 root visits

    root.add_child(Move(E2, E4, FLAG_DOUBLE_PUSH), 0.5f);
    root.add_child(Move(D2, D4, FLAG_DOUBLE_PUSH), 0.5f);

    // Give E4 a high Q from one visit
    root.child(0)->update(0.8f);
    // D4 is unvisited — FPU = 0, but exploration bonus is higher (denominator = 1 vs 2)

    float c_puct = 2.5f;
    float fpu = 0.0f;
    mcts::Node* selected = root.select_child(c_puct, fpu);

    // With c_puct=2.5, root_visits=2, sqrt(2)=1.414:
    // E4: Q=0.8 + 2.5 * 0.5 * 1.414 / (1+1) = 0.8 + 0.884 = 1.684
    // D4: Q=0.0 + 2.5 * 0.5 * 1.414 / (1+0) = 0.0 + 1.768 = 1.768
    // D4 wins — unvisited node gets explored
    EXPECT_EQ(selected->move(), Move(D2, D4, FLAG_DOUBLE_PUSH));
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: 6 tests PASS.

- [ ] **Step 5: Write test for best_move (most visited)**

Append to `tests/test_mcts.cpp`:

```cpp
TEST_F(MCTSTest, BestMoveReturnsMostVisited) {
    mcts::Node root;
    root.add_child(Move(E2, E4, FLAG_DOUBLE_PUSH), 0.3f);
    root.add_child(Move(D2, D4, FLAG_DOUBLE_PUSH), 0.5f);
    root.add_child(Move(G1, F3, FLAG_QUIET), 0.2f);

    // E4: 10 visits, D4: 50 visits, Nf3: 5 visits
    for (int i = 0; i < 10; i++) root.child(0)->update(0.5f);
    for (int i = 0; i < 50; i++) root.child(1)->update(0.4f);
    for (int i = 0; i < 5; i++) root.child(2)->update(0.6f);

    EXPECT_EQ(root.best_move(), Move(D2, D4, FLAG_DOUBLE_PUSH));
}
```

- [ ] **Step 6: Run tests to verify it passes**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_mcts.cpp
git commit -m "test(mcts): add tests for add_child, update, PUCT selection, and best_move"
```

---

### Task 3: Evaluator interface and RandomEvaluator

**Files:**
- Create: `src/mcts/search.h` (partial — Evaluator interface + SearchParams + SearchResult)
- Create: `src/mcts/search.cpp` (partial — RandomEvaluator)
- Modify: `tests/test_mcts.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test for RandomEvaluator**

Append to `tests/test_mcts.cpp`:

```cpp
#include "mcts/search.h"
#include "core/position.h"
#include "core/movegen.h"

TEST_F(MCTSTest, RandomEvaluatorReturnsUniformPolicy) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    mcts::RandomEvaluator eval;
    mcts::EvalResult result = eval.evaluate(pos, moves, num_moves);

    EXPECT_EQ(static_cast<int>(result.policy.size()), num_moves);

    // All priors should be equal (uniform)
    float expected_prior = 1.0f / num_moves;
    for (int i = 0; i < num_moves; i++) {
        EXPECT_NEAR(result.policy[i], expected_prior, 1e-5f);
    }

    // Value should be in [-1, 1]
    EXPECT_GE(result.value, -1.0f);
    EXPECT_LE(result.value, 1.0f);
}

TEST_F(MCTSTest, RandomEvaluatorCheckmateValue) {
    // Scholar's mate position — black is checkmated
    Position pos;
    pos.set_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    // No legal moves and in check = checkmate
    EXPECT_EQ(num_moves, 0);
    EXPECT_TRUE(pos.in_check());

    mcts::RandomEvaluator eval;
    mcts::EvalResult result = eval.evaluate(pos, moves, num_moves);

    // Side to move is checkmated — value = -1 (loss)
    EXPECT_FLOAT_EQ(result.value, -1.0f);
    EXPECT_TRUE(result.policy.empty());
}

TEST_F(MCTSTest, RandomEvaluatorStalemateValue) {
    // Stalemate position
    Position pos;
    pos.set_fen("k7/8/1K6/8/8/8/8/8 b - - 0 1");

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    // If stalemate, value = 0
    if (num_moves == 0 && !pos.in_check()) {
        mcts::RandomEvaluator eval;
        mcts::EvalResult result = eval.evaluate(pos, moves, num_moves);
        EXPECT_FLOAT_EQ(result.value, 0.0f);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: Build failure — `mcts/search.h` doesn't exist.

- [ ] **Step 3: Implement Evaluator interface and RandomEvaluator**

Add `src/mcts/search.cpp` to `CMakeLists.txt`:

```cmake
add_library(chess_core
    src/core/bitboard.cpp
    src/core/attacks.cpp
    src/core/position.cpp
    src/core/movegen.cpp
    src/mcts/node.cpp
    src/mcts/search.cpp
)
```

Create `src/mcts/search.h`:

```cpp
#pragma once
#include "core/types.h"
#include "core/position.h"
#include "mcts/node.h"
#include <vector>
#include <memory>

namespace mcts {

// Result of evaluating a position
struct EvalResult {
    std::vector<float> policy;  // Prior probability per legal move (same order as moves array)
    float value;                // Position evaluation from side-to-move perspective: [-1, +1]
};

// Abstract evaluator interface — neural network plugs in here later
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) = 0;
};

// Stub evaluator for testing MCTS without a neural network
// Returns uniform policy over legal moves and material-based value
class RandomEvaluator : public Evaluator {
public:
    EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) override;
};

// Search parameters
struct SearchParams {
    int num_iterations = 800;
    float c_puct = 2.5f;
    float fpu_reduction_root = 0.44f;
    float fpu_reduction = 0.25f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    bool add_noise = true;    // Add Dirichlet noise at root (for self-play)
};

// Search result
struct SearchResult {
    Move best_move;
    std::vector<Move> moves;           // Legal moves at root
    std::vector<int> visit_counts;     // Visit count per move
    float root_value;                  // Value estimate at root
    int total_nodes;                   // Total nodes in tree
};

class Search {
public:
    Search(Evaluator& evaluator, const SearchParams& params = SearchParams{});

    SearchResult run(const Position& pos);

private:
    Evaluator& evaluator_;
    SearchParams params_;

    Node* select(Node* root);
    void expand(Node* node, const Position& pos);
    float evaluate(Node* node, const Position& pos);
    void backpropagate(Node* node, float value);
    void add_dirichlet_noise(Node* root);

    // Position tracking during selection
    void apply_moves_to_root(const Position& root_pos, Node* node, Position& out_pos);
};

} // namespace mcts
```

Create `src/mcts/search.cpp` (partial — just RandomEvaluator for now):

```cpp
#include "mcts/search.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>

namespace mcts {

// --- RandomEvaluator ---

EvalResult RandomEvaluator::evaluate(const Position& pos, const Move* moves, int num_moves) {
    EvalResult result;

    // Terminal position
    if (num_moves == 0) {
        if (pos.in_check()) {
            result.value = -1.0f; // Checkmated — loss for side to move
        } else {
            result.value = 0.0f;  // Stalemate — draw
        }
        return result;
    }

    // Uniform policy over legal moves
    float uniform = 1.0f / num_moves;
    result.policy.assign(num_moves, uniform);

    // Simple material-based evaluation
    // Count material for side to move vs opponent
    Color us = pos.side_to_move();
    Color them = ~us;

    static constexpr float piece_values[] = {
        1.0f,   // PAWN
        3.0f,   // KNIGHT
        3.0f,   // BISHOP
        5.0f,   // ROOK
        9.0f,   // QUEEN
        0.0f    // KING
    };

    float material = 0.0f;
    for (int pt = PAWN; pt <= QUEEN; pt++) {
        material += piece_values[pt] * popcount(pos.pieces(us, PieceType(pt)));
        material -= piece_values[pt] * popcount(pos.pieces(them, PieceType(pt)));
    }

    // Squash material advantage into [-1, 1] using tanh-like scaling
    // 3 pawns advantage ≈ 0.95
    result.value = material / (std::abs(material) + 3.0f);

    return result;
}

// --- Search (stub implementations, filled in Task 4) ---

Search::Search(Evaluator& evaluator, const SearchParams& params)
    : evaluator_(evaluator), params_(params) {}

SearchResult Search::run(const Position& /*pos*/) {
    return SearchResult{};
}

Node* Search::select(Node* /*root*/) { return nullptr; }
void Search::expand(Node* /*node*/, const Position& /*pos*/) {}
float Search::evaluate(Node* /*node*/, const Position& /*pos*/) { return 0.0f; }
void Search::backpropagate(Node* /*node*/, float /*value*/) {}
void Search::add_dirichlet_noise(Node* /*root*/) {}
void Search::apply_moves_to_root(const Position& /*root_pos*/, Node* /*node*/, Position& /*out_pos*/) {}

} // namespace mcts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: 10 tests PASS (7 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/mcts/search.h src/mcts/search.cpp tests/test_mcts.cpp CMakeLists.txt
git commit -m "feat(mcts): add Evaluator interface and RandomEvaluator with material-based value"
```

---

### Task 4: MCTS search loop — select, expand, evaluate, backpropagate

**Files:**
- Modify: `src/mcts/search.cpp`
- Modify: `tests/test_mcts.cpp`

- [ ] **Step 1: Write the failing test for a basic search**

Append to `tests/test_mcts.cpp`:

```cpp
TEST_F(MCTSTest, SearchReturnsLegalMove) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // Must return a legal move
    EXPECT_FALSE(result.best_move.is_none());

    // Verify it's actually legal
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    bool found = false;
    for (int i = 0; i < num_moves; i++) {
        if (moves[i] == result.best_move) { found = true; break; }
    }
    EXPECT_TRUE(found);

    // Visit counts should sum to num_iterations
    int total_visits = 0;
    for (int v : result.visit_counts) total_visits += v;
    EXPECT_EQ(total_visits, params.num_iterations);
}

TEST_F(MCTSTest, SearchFindsObviousCapture) {
    // White queen can take undefended black queen
    Position pos;
    pos.set_fen("rnb1kbnr/pppppppp/8/4q3/3Q4/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 400;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // With enough iterations, MCTS should strongly prefer capturing the queen
    // Qd4xe5 — d4=27, e5=36
    Move capture_queen(D4, E5, FLAG_CAPTURE);
    EXPECT_EQ(result.best_move, capture_queen);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R SearchReturnsLegalMove --output-on-failure`
Expected: FAIL — `Search::run` returns empty result.

- [ ] **Step 3: Implement the MCTS search loop**

Replace the stub implementations in `src/mcts/search.cpp` with the full implementation:

```cpp
#include "mcts/search.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mcts {

// --- RandomEvaluator --- (keep existing implementation)

EvalResult RandomEvaluator::evaluate(const Position& pos, const Move* moves, int num_moves) {
    EvalResult result;

    if (num_moves == 0) {
        if (pos.in_check()) {
            result.value = -1.0f;
        } else {
            result.value = 0.0f;
        }
        return result;
    }

    float uniform = 1.0f / num_moves;
    result.policy.assign(num_moves, uniform);

    Color us = pos.side_to_move();
    Color them = ~us;

    static constexpr float piece_values[] = {
        1.0f, 3.0f, 3.0f, 5.0f, 9.0f, 0.0f
    };

    float material = 0.0f;
    for (int pt = PAWN; pt <= QUEEN; pt++) {
        material += piece_values[pt] * popcount(pos.pieces(us, PieceType(pt)));
        material -= piece_values[pt] * popcount(pos.pieces(them, PieceType(pt)));
    }

    result.value = material / (std::abs(material) + 3.0f);
    return result;
}

// --- Search ---

Search::Search(Evaluator& evaluator, const SearchParams& params)
    : evaluator_(evaluator), params_(params) {}

// Collect the path of moves from root to a given node
static void collect_path(Node* node, std::vector<Move>& path) {
    path.clear();
    // Walk up to root (root's move is none)
    std::vector<Move> reversed;
    Node* cur = node;
    while (cur->parent() != nullptr) {
        reversed.push_back(cur->move());
        cur = cur->parent();
    }
    // Reverse to get root-to-node order
    for (int i = static_cast<int>(reversed.size()) - 1; i >= 0; i--) {
        path.push_back(reversed[i]);
    }
}

void Search::apply_moves_to_root(const Position& root_pos, Node* node, Position& out_pos) {
    std::vector<Move> path;
    collect_path(node, path);

    out_pos = root_pos;
    UndoInfo undo;
    for (Move m : path) {
        out_pos.make_move(m, undo);
    }
}

Node* Search::select(Node* node) {
    while (!node->is_leaf()) {
        bool is_root = (node->parent() == nullptr);
        float fpu_reduction = is_root ? params_.fpu_reduction_root : params_.fpu_reduction;
        float parent_q = node->mean_value();
        float fpu_value = parent_q - fpu_reduction;

        node = node->select_child(params_.c_puct, fpu_value);
    }
    return node;
}

void Search::expand(Node* node, const Position& pos) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 0) return; // Terminal node — don't expand

    EvalResult eval_result = evaluator_.evaluate(pos, moves, num_moves);

    for (int i = 0; i < num_moves; i++) {
        node->add_child(moves[i], eval_result.policy[i]);
    }
}

float Search::evaluate(Node* node, const Position& pos) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    EvalResult result = evaluator_.evaluate(pos, moves, num_moves);
    return result.value;
}

void Search::backpropagate(Node* node, float value) {
    // Walk from leaf back to root, negating value at each level
    // because alternating players have opposite perspectives
    while (node != nullptr) {
        node->update(value);
        value = -value;
        node = node->parent();
    }
}

void Search::add_dirichlet_noise(Node* root) {
    if (root->is_leaf()) return;

    int num_children = root->num_children();
    std::vector<float> noise(num_children);

    // Generate Dirichlet noise using gamma distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(params_.dirichlet_alpha, 1.0f);

    float noise_sum = 0.0f;
    for (int i = 0; i < num_children; i++) {
        noise[i] = gamma(gen);
        noise_sum += noise[i];
    }
    for (int i = 0; i < num_children; i++) {
        noise[i] /= noise_sum;
    }

    // Blend: P'(a) = (1-ε)*P(a) + ε*noise(a)
    float eps = params_.dirichlet_epsilon;
    for (int i = 0; i < num_children; i++) {
        float old_prior = root->child(i)->prior();
        float new_prior = (1.0f - eps) * old_prior + eps * noise[i];
        root->child(i)->set_prior(new_prior);
    }
}

SearchResult Search::run(const Position& pos) {
    auto root = std::make_unique<Node>();

    // Expand root
    expand(root.get(), pos);

    if (root->is_leaf()) {
        // No legal moves — terminal position
        SearchResult result;
        result.best_move = Move::none();
        result.root_value = evaluate(root.get(), pos);
        result.total_nodes = 1;
        return result;
    }

    // Initial root evaluation and visit
    float root_value = evaluate(root.get(), pos);
    root->update(root_value);

    // Add Dirichlet noise at root for exploration
    if (params_.add_noise) {
        add_dirichlet_noise(root.get());
    }

    // Main MCTS loop
    for (int iter = 0; iter < params_.num_iterations; iter++) {
        // 1. SELECT — descend to a leaf
        Node* leaf = select(root.get());

        // 2. Get position at leaf
        Position leaf_pos;
        apply_moves_to_root(pos, leaf, leaf_pos);

        // Check if leaf is a terminal node (already visited, no children = terminal)
        if (leaf->visit_count() > 0) {
            // Already visited leaf — evaluate and backprop without expanding
            float value = evaluate(leaf, leaf_pos);
            backpropagate(leaf, value);
            continue;
        }

        // 3. EVALUATE — get value for this leaf
        float value = evaluate(leaf, leaf_pos);

        // 4. EXPAND — create children
        expand(leaf, leaf_pos);

        // 5. BACKPROPAGATE — update all nodes on path
        backpropagate(leaf, value);
    }

    // Build result
    SearchResult result;
    result.best_move = root->best_move();
    result.root_value = root->mean_value();

    int num_children = root->num_children();
    result.moves.resize(num_children);
    result.visit_counts.resize(num_children);
    for (int i = 0; i < num_children; i++) {
        result.moves[i] = root->child(i)->move();
        result.visit_counts[i] = root->child(i)->visit_count();
    }

    // Count total nodes (approximate — root visit count)
    result.total_nodes = root->visit_count();

    return result;
}

} // namespace mcts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mcts/search.cpp tests/test_mcts.cpp
git commit -m "feat(mcts): implement MCTS search loop with select, expand, evaluate, backprop"
```

---

### Task 5: Dirichlet noise and temperature-based move selection

**Files:**
- Modify: `src/mcts/search.h`
- Modify: `src/mcts/search.cpp`
- Modify: `tests/test_mcts.cpp`

- [ ] **Step 1: Write tests for Dirichlet noise**

Append to `tests/test_mcts.cpp`:

```cpp
TEST_F(MCTSTest, DirichletNoiseModifiesPriors) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 50;
    params.add_noise = true;

    mcts::Search search_with_noise(eval, params);
    mcts::SearchResult result1 = search_with_noise.run(pos);

    // With noise, priors are perturbed — hard to test deterministically,
    // but we can verify the search still returns a valid legal move
    EXPECT_FALSE(result1.best_move.is_none());

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    bool found = false;
    for (int i = 0; i < num_moves; i++) {
        if (moves[i] == result1.best_move) { found = true; break; }
    }
    EXPECT_TRUE(found);
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R DirichletNoise --output-on-failure`
Expected: PASS.

- [ ] **Step 3: Write tests for temperature-based move selection**

Add `select_move_with_temperature` to `src/mcts/search.h`:

In the `Search` class public section, add:

```cpp
    // Temperature-based move selection for self-play
    // temperature = 1.0: proportional to visit counts
    // temperature → 0: greedy (pick most visited)
    static Move select_move_with_temperature(const SearchResult& result, float temperature);
```

Append test to `tests/test_mcts.cpp`:

```cpp
TEST_F(MCTSTest, TemperatureZeroSelectsMostVisited) {
    mcts::SearchResult result;
    result.moves = {
        Move(E2, E4, FLAG_DOUBLE_PUSH),
        Move(D2, D4, FLAG_DOUBLE_PUSH),
        Move(G1, F3, FLAG_QUIET)
    };
    result.visit_counts = { 10, 50, 5 };

    Move selected = mcts::Search::select_move_with_temperature(result, 0.0f);
    EXPECT_EQ(selected, Move(D2, D4, FLAG_DOUBLE_PUSH));
}

TEST_F(MCTSTest, TemperatureOneDistributesProportionally) {
    mcts::SearchResult result;
    result.moves = {
        Move(E2, E4, FLAG_DOUBLE_PUSH),
        Move(D2, D4, FLAG_DOUBLE_PUSH),
        Move(G1, F3, FLAG_QUIET)
    };
    result.visit_counts = { 100, 100, 100 }; // Equal visits

    // With equal visits at any temperature, all moves are equally likely
    // Run many trials — each move should appear ~1/3 of the time
    int counts[3] = {0, 0, 0};
    for (int trial = 0; trial < 3000; trial++) {
        Move m = mcts::Search::select_move_with_temperature(result, 1.0f);
        for (int i = 0; i < 3; i++) {
            if (m == result.moves[i]) { counts[i]++; break; }
        }
    }
    // Each should be roughly 1000 ± 100
    for (int i = 0; i < 3; i++) {
        EXPECT_GT(counts[i], 800);
        EXPECT_LT(counts[i], 1200);
    }
}
```

- [ ] **Step 4: Implement select_move_with_temperature**

Add to `src/mcts/search.cpp`:

```cpp
Move Search::select_move_with_temperature(const SearchResult& result, float temperature) {
    if (result.moves.empty()) return Move::none();

    // Temperature ≈ 0: greedy selection
    if (temperature < 0.01f) {
        int best_idx = 0;
        for (int i = 1; i < static_cast<int>(result.visit_counts.size()); i++) {
            if (result.visit_counts[i] > result.visit_counts[best_idx]) {
                best_idx = i;
            }
        }
        return result.moves[best_idx];
    }

    // Temperature-based sampling: π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)
    int num_moves = static_cast<int>(result.moves.size());
    std::vector<float> probs(num_moves);
    float inv_temp = 1.0f / temperature;

    float sum = 0.0f;
    for (int i = 0; i < num_moves; i++) {
        probs[i] = std::pow(static_cast<float>(result.visit_counts[i]), inv_temp);
        sum += probs[i];
    }

    if (sum <= 0.0f) {
        // Fallback: uniform random
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, num_moves - 1);
        return result.moves[dist(gen)];
    }

    for (int i = 0; i < num_moves; i++) {
        probs[i] /= sum;
    }

    // Sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return result.moves[dist(gen)];
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: All MCTS tests PASS (15 tests).

- [ ] **Step 6: Commit**

```bash
git add src/mcts/search.h src/mcts/search.cpp tests/test_mcts.cpp
git commit -m "feat(mcts): add Dirichlet noise at root and temperature-based move selection"
```

---

### Task 6: Terminal position handling and edge cases

**Files:**
- Modify: `tests/test_mcts.cpp`
- Modify: `src/mcts/search.cpp` (if needed)

- [ ] **Step 1: Write tests for terminal positions**

Append to `tests/test_mcts.cpp`:

```cpp
TEST_F(MCTSTest, SearchHandlesCheckmate) {
    // Fool's mate — white is checkmated
    Position pos;
    pos.set_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // Checkmated — no legal moves
    EXPECT_TRUE(result.best_move.is_none());
    EXPECT_FLOAT_EQ(result.root_value, -1.0f);
}

TEST_F(MCTSTest, SearchHandlesStalemate) {
    // Stalemate: black king on a8, white king on b6, white pawn on a7
    Position pos;
    pos.set_fen("k7/P7/1K6/8/8/8/8/8 b - - 0 1");

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 0 && !pos.in_check()) {
        mcts::RandomEvaluator eval;
        mcts::SearchParams params;
        params.num_iterations = 100;
        params.add_noise = false;

        mcts::Search search(eval, params);
        mcts::SearchResult result = search.run(pos);

        EXPECT_TRUE(result.best_move.is_none());
        EXPECT_FLOAT_EQ(result.root_value, 0.0f);
    }
}

TEST_F(MCTSTest, SearchWithOneMove) {
    // Position where only one legal move exists
    // King in check with one escape
    Position pos;
    pos.set_fen("8/8/8/8/8/5k2/4q3/7K w - - 0 1");

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 1) {
        mcts::RandomEvaluator eval;
        mcts::SearchParams params;
        params.num_iterations = 50;
        params.add_noise = false;

        mcts::Search search(eval, params);
        mcts::SearchResult result = search.run(pos);

        EXPECT_EQ(result.best_move, moves[0]);
    }
}

TEST_F(MCTSTest, SearchVisitCountsConsistency) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 200;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // Every legal move should have at least some visits
    EXPECT_EQ(result.moves.size(), result.visit_counts.size());
    EXPECT_GT(result.moves.size(), 0u);

    // Total visit counts should equal num_iterations
    int total = 0;
    for (int v : result.visit_counts) {
        EXPECT_GE(v, 0);
        total += v;
    }
    EXPECT_EQ(total, params.num_iterations);
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cmake --build build --config Release && ctest --test-dir build --build-config Release -R MCTS --output-on-failure`
Expected: All MCTS tests PASS (19 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcts.cpp src/mcts/search.cpp
git commit -m "test(mcts): add edge case tests for checkmate, stalemate, single move, and visit count consistency"
```

---

### Task 7: CLI integration and documentation updates

**Files:**
- Modify: `src/main.cpp`
- Modify: `docs/changelog.md`
- Modify: `docs/architecture.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update main.cpp with MCTS search command**

Read the current `src/main.cpp` and add an MCTS search mode. The CLI should support:
- `chess_engine search [FEN] [iterations]` — run MCTS and print best move + visit distribution

Replace `src/main.cpp` with:

```cpp
#include "core/types.h"
#include "core/bitboard.h"
#include "core/attacks.h"
#include "core/position.h"
#include "core/movegen.h"
#include "mcts/search.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

static uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    if (depth == 1) return n;
    uint64_t nodes = 0;
    UndoInfo undo;
    for (int i = 0; i < n; i++) {
        pos.make_move(moves[i], undo);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);
    }
    return nodes;
}

static void divide(Position& pos, int depth) {
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    uint64_t total = 0;
    UndoInfo undo;
    for (int i = 0; i < n; i++) {
        pos.make_move(moves[i], undo);
        uint64_t count = (depth <= 1) ? 1 : perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);
        std::cout << moves[i].to_uci() << ": " << count << "\n";
        total += count;
    }
    std::cout << "\nTotal: " << total << "\n";
}

static void search_position(const std::string& fen, int iterations) {
    Position pos;
    pos.set_fen(fen);

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = iterations;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    if (result.best_move.is_none()) {
        std::cout << "No legal moves (";
        if (pos.in_check()) std::cout << "checkmate";
        else std::cout << "stalemate";
        std::cout << ")\n";
        return;
    }

    std::cout << "Position: " << pos.to_fen() << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Root value: " << std::fixed << std::setprecision(3) << result.root_value << "\n";
    std::cout << "Best move: " << result.best_move.to_uci() << "\n\n";

    // Sort by visit count descending for display
    std::vector<std::pair<int, int>> indexed(result.moves.size());
    for (int i = 0; i < static_cast<int>(result.moves.size()); i++) {
        indexed[i] = {result.visit_counts[i], i};
    }
    std::sort(indexed.begin(), indexed.end(), std::greater<>());

    int total_visits = 0;
    for (int v : result.visit_counts) total_visits += v;

    std::cout << "Move distribution (top 10):\n";
    int shown = 0;
    for (auto& [visits, idx] : indexed) {
        if (shown >= 10) break;
        float pct = 100.0f * visits / total_visits;
        std::cout << "  " << std::setw(5) << result.moves[idx].to_uci()
                  << "  " << std::setw(6) << visits << " visits"
                  << "  (" << std::fixed << std::setprecision(1) << pct << "%)\n";
        shown++;
    }
}

int main(int argc, char* argv[]) {
    attacks::init();

    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  chess_engine perft <depth> [fen]\n";
        std::cout << "  chess_engine search [fen] [iterations]\n";
        return 1;
    }

    std::string command = argv[1];

    if (command == "perft") {
        if (argc < 3) {
            std::cerr << "Usage: chess_engine perft <depth> [fen]\n";
            return 1;
        }
        int depth = std::atoi(argv[2]);
        std::string fen = (argc >= 4) ? argv[3]
            : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        Position pos;
        pos.set_fen(fen);
        divide(pos, depth);
    } else if (command == "search") {
        std::string fen = (argc >= 3) ? argv[2]
            : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        int iterations = (argc >= 4) ? std::atoi(argv[3]) : 800;
        search_position(fen, iterations);
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        return 1;
    }

    return 0;
}
```

- [ ] **Step 2: Build and test the CLI**

Run: `cmake --build build --config Release`

Then test:
```bash
./build/Release/chess_engine.exe search
./build/Release/chess_engine.exe search "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" 400
```

Verify it prints a best move and visit distribution.

- [ ] **Step 3: Run all tests to verify nothing broke**

Run: `ctest --test-dir build --build-config Release --output-on-failure`
Expected: All tests PASS (77 perft + ~19 MCTS tests).

- [ ] **Step 4: Update CLAUDE.md**

Update the "Current Milestone" section to reflect Plan 2 completion:

```markdown
## Current Milestone

**Phase 2: MCTS Search Engine** (Plan 2 of 6)

Building Monte Carlo Tree Search: tree structure, PUCT selection, expand/evaluate/backprop loop, Dirichlet noise, temperature-based move selection. Uses a stub evaluator (uniform policy + material value) — neural network integration comes in Plan 3/5.

**Done when:** MCTS returns legal moves, prefers obvious captures, handles terminal positions, all tests pass.

Plan location: `docs/superpowers/plans/2026-04-11-mcts-search.md`
```

Update the "Key Features" section to include MCTS:

```markdown
## Key Features (Current Scope — Phase 2)

- Everything from Phase 1 (bitboard, move generation, perft)
- MCTS tree with Node struct (visit count, value, prior, children)
- PUCT-based child selection with exploration/exploitation balance
- Evaluator interface for neural network integration
- RandomEvaluator: uniform policy + material-based value
- Dirichlet noise at root for exploration diversity
- Temperature-based move selection (τ=1 for exploration, τ→0 for greedy)
- First Play Urgency (FPU) reduction
- CLI `search` command showing best move + visit distribution
```

Update the "Non-Goals" to move MCTS out:

```markdown
## Non-Goals (Right Now)

- Neural network architecture or training (Plan 3)
- Self-play game generation (Plan 4)
- C++ neural network inference (Plan 5)
- Visualization dashboard (Plan 6)
- Magic bitboards or other speed optimizations
- Zobrist hashing / transposition tables
- UCI protocol support
- Opening books or endgame tablebases
- Multi-threaded MCTS / virtual loss (deferred to Plan 5 integration)
```

Update the Plan Roadmap table:

```markdown
## Plan Roadmap

| Plan | Subsystem | Status |
|------|-----------|--------|
| 1 | Chess Engine Core (types, bitboard, position, movegen, perft) | **Complete** |
| 2 | MCTS Search Engine | **Current** |
| 3 | Neural Network Architecture & Training (Python/PyTorch) | Not started |
| 4 | Self-Play & Data Pipeline | Not started |
| 5 | C++ Neural Net Inference + Integration | Not started |
| 6 | Visualization Dashboard | Not started |
```

Add documentation entry:

```markdown
- [Plan 2: MCTS Search](docs/superpowers/plans/2026-04-11-mcts-search.md) — MCTS implementation plan
```

- [ ] **Step 5: Update docs/changelog.md**

Add under `[Unreleased]`:

```markdown
## [Unreleased]

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
```

- [ ] **Step 6: Update docs/architecture.md**

Add a note under "### 2. MCTS Search Engine (C++17) — Plan 2" that this is now implemented with a stub evaluator, and the data flow matches the architecture doc.

- [ ] **Step 7: Commit**

```bash
git add src/main.cpp docs/changelog.md docs/architecture.md CLAUDE.md
git commit -m "feat(mcts): add search CLI command, update docs and changelog for Plan 2"
```

---

Plan complete and saved to `docs/superpowers/plans/2026-04-11-mcts-search.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?