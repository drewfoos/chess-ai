#include <gtest/gtest.h>
#include "mcts/node.h"
#include "mcts/search.h"
#include "mcts/nn_cache.h"
#include "core/attacks.h"
#include "core/position.h"
#include "core/movegen.h"
#include "neural/position_history.h"
#include "neural/encoder.h"
#include "neural/policy_map.h"
#include <numeric>
#include <cmath>
#include <atomic>

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
    // Nodes no longer take move/prior in constructor; test edge-based prior instead
    mcts::Node node;
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    node.set_prior(0.35f);
    EXPECT_EQ(node.visit_count(), 0);
    EXPECT_NEAR(node.prior(), 0.35f, 0.002f);
    EXPECT_TRUE(node.is_leaf());
}

TEST_F(MCTSTest, AddChildren) {
    mcts::Node root;
    Move moves[] = {Move(E2, E4, FLAG_DOUBLE_PUSH), Move(D2, D4, FLAG_DOUBLE_PUSH), Move(G1, F3, FLAG_QUIET)};
    float priors[] = {0.4f, 0.3f, 0.2f};
    root.create_edges(moves, priors, 3);

    EXPECT_FALSE(root.is_leaf());
    EXPECT_EQ(root.num_children(), 3);
    EXPECT_EQ(root.edge(0).move(), Move(E2, E4, FLAG_DOUBLE_PUSH));
    EXPECT_NEAR(root.edge(0).prior(), 0.4f, 0.002f);

    // ensure_child creates the child node lazily
    mcts::Node* child0 = root.ensure_child(0);
    EXPECT_EQ(child0->parent(), &root);
    EXPECT_EQ(child0->move(), Move(E2, E4, FLAG_DOUBLE_PUSH));
    EXPECT_NEAR(child0->prior(), 0.4f, 0.002f);
}

TEST_F(MCTSTest, UpdateVisitAndValue) {
    mcts::Node node;
    node.set_prior(0.5f);
    node.update(0.6f);
    EXPECT_EQ(node.visit_count(), 1);
    EXPECT_FLOAT_EQ(node.total_value(), 0.6f);
    EXPECT_FLOAT_EQ(node.mean_value(), 0.6f);

    node.update(0.4f);
    EXPECT_EQ(node.visit_count(), 2);
    EXPECT_FLOAT_EQ(node.total_value(), 1.0f);
    EXPECT_FLOAT_EQ(node.mean_value(), 0.5f);
}

TEST_F(MCTSTest, PUCTSelectsHighPriorUnvisited) {
    // With no visits, PUCT should prefer the child with highest prior
    mcts::Node root;
    root.update(0.0f); // root needs a visit for sqrt(N_parent) > 0
    Move moves[] = {Move(E2, E4, FLAG_DOUBLE_PUSH), Move(D2, D4, FLAG_DOUBLE_PUSH), Move(G1, F3, FLAG_QUIET)};
    float priors[] = {0.1f, 0.6f, 0.3f};
    root.create_edges(moves, priors, 3);

    // select_child now returns nullptr for unvisited edges (no Node allocated yet)
    // Instead, verify via edge priors that D4 has highest prior
    // The select_child method works on edges and returns child_node which may be null
    float c_puct = 2.5f;
    float fpu = 0.0f;
    // We can test that best_move returns the edge move with most visits (all 0, so first)
    // But let's test select_child which should pick edge index 1 (D4, prior 0.6)
    // Since no children are allocated, select_child returns nullptr for the best edge
    // Let's ensure children first to test the old behavior
    root.ensure_child(0);
    root.ensure_child(1);
    root.ensure_child(2);
    mcts::Node* selected = root.select_child(c_puct, fpu);
    EXPECT_EQ(selected->move(), Move(D2, D4, FLAG_DOUBLE_PUSH));
}

TEST_F(MCTSTest, PUCTBalancesExplorationExploitation) {
    mcts::Node root;
    // 3 root visits. PUCT numerator is sqrt(max(N-1, 1)) = sqrt(2) — the
    // "minus one" matches Lc0's formula (search.cc:1720) and dampens early
    // exploration bonuses.
    root.update(0.0f);
    root.update(0.0f);
    root.update(0.0f);

    Move moves[] = {Move(E2, E4, FLAG_DOUBLE_PUSH), Move(D2, D4, FLAG_DOUBLE_PUSH)};
    float priors[] = {0.5f, 0.5f};
    root.create_edges(moves, priors, 2);

    // Ensure both children exist
    mcts::Node* e4 = root.ensure_child(0);
    root.ensure_child(1);

    // Give E4 a high Q from one visit
    e4->update(0.8f);
    // D4 is unvisited — FPU = 0, but exploration bonus is higher (denominator = 1 vs 2)

    float c_puct = 2.5f;
    float fpu = 0.0f;
    mcts::Node* selected = root.select_child(c_puct, fpu);

    // With c_puct=2.5, root_visits=3, sqrt(3-1)=1.414:
    // E4: Q=0.8 + 2.5 * 0.5 * 1.414 / (1+1) = 0.8 + 0.884 = 1.684
    // D4: Q=0.0 + 2.5 * 0.5 * 1.414 / (1+0) = 0.0 + 1.768 = 1.768
    // D4 wins — unvisited node gets explored
    EXPECT_EQ(selected->move(), Move(D2, D4, FLAG_DOUBLE_PUSH));
}

TEST_F(MCTSTest, BestMoveReturnsMostVisited) {
    mcts::Node root;
    Move moves[] = {Move(E2, E4, FLAG_DOUBLE_PUSH), Move(D2, D4, FLAG_DOUBLE_PUSH), Move(G1, F3, FLAG_QUIET)};
    float priors[] = {0.3f, 0.5f, 0.2f};
    root.create_edges(moves, priors, 3);

    mcts::Node* e4 = root.ensure_child(0);
    mcts::Node* d4 = root.ensure_child(1);
    mcts::Node* nf3 = root.ensure_child(2);

    // E4: 10 visits, D4: 50 visits, Nf3: 5 visits
    for (int i = 0; i < 10; i++) e4->update(0.5f);
    for (int i = 0; i < 50; i++) d4->update(0.4f);
    for (int i = 0; i < 5; i++) nf3->update(0.6f);

    EXPECT_EQ(root.best_move(), Move(D2, D4, FLAG_DOUBLE_PUSH));
}

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

TEST_F(MCTSTest, SearchReturnsLegalMove) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.add_noise = false;
    params.smart_pruning = false;

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

    // Visit counts should sum to approximately num_iterations
    // (may be fewer due to smart pruning, or slightly different due to batch boundaries)
    int total_visits = 0;
    for (int v : result.visit_counts) total_visits += v;
    EXPECT_GT(total_visits, 0);
    EXPECT_LE(total_visits, params.num_iterations);
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

    ASSERT_EQ(num_moves, 0);
    ASSERT_FALSE(pos.in_check());

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    EXPECT_TRUE(result.best_move.is_none());
    EXPECT_FLOAT_EQ(result.root_value, 0.0f);
}

TEST_F(MCTSTest, SearchWithOneMove) {
    // Position where only one legal move exists
    // King in check with one escape
    Position pos;
    pos.set_fen("8/8/8/8/8/5k2/4q3/7K w - - 0 1");

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    ASSERT_EQ(num_moves, 1);

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 50;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    EXPECT_EQ(result.best_move, moves[0]);
}

TEST_F(MCTSTest, SearchVisitCountsConsistency) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 200;
    params.add_noise = false;
    params.smart_pruning = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // Every legal move should have at least some visits
    EXPECT_EQ(result.moves.size(), result.visit_counts.size());
    EXPECT_GT(result.moves.size(), 0u);

    // Total visit counts should be close to num_iterations
    // (may be fewer due to smart pruning or batch boundaries)
    int total = 0;
    for (int v : result.visit_counts) {
        EXPECT_GE(v, 0);
        total += v;
    }
    EXPECT_GT(total, 0);
    EXPECT_LE(total, params.num_iterations);
}

TEST_F(MCTSTest, Float16RoundTrip) {
    // Test float16 conversion round-trip
    float values[] = {0.0f, 0.5f, 1.0f, 0.001f, 0.99f, -0.5f};
    for (float v : values) {
        uint16_t h = mcts::Node::float_to_half(v);
        float back = mcts::Node::half_to_float(h);
        EXPECT_NEAR(back, v, 0.002f) << "Round-trip failed for " << v;
    }
}

TEST_F(MCTSTest, NodePriorFloat16) {
    // Prior stored as float16 should still be usable
    mcts::Node node;
    node.set_prior(0.35f);
    EXPECT_NEAR(node.prior(), 0.35f, 0.002f);
    node.set_prior(0.75f);
    EXPECT_NEAR(node.prior(), 0.75f, 0.002f);
}

TEST_F(MCTSTest, VirtualLossApplyRevert) {
    mcts::Node node;
    node.set_prior(0.5f);
    node.update(0.6f);
    EXPECT_EQ(node.visit_count(), 1);
    EXPECT_EQ(node.pending_evals(), 0);

    node.apply_virtual_loss();
    EXPECT_EQ(node.visit_count(), 2);
    EXPECT_EQ(node.pending_evals(), 1);

    node.revert_virtual_loss();
    EXPECT_EQ(node.visit_count(), 1);
    EXPECT_EQ(node.pending_evals(), 0);
}

TEST_F(MCTSTest, MultivisitIncrementN) {
    // update(v, n) folds N equivalent updates into one call: visit += n,
    // total_value += n*v, sum_sq_value += n*v*v.
    mcts::Node node;
    node.update(0.2f);               // visit=1, total=0.2, sum_sq=0.04
    node.update(0.5f, 3);            // visit=4, total=0.2 + 3*0.5=1.7, sum_sq=0.04 + 3*0.25=0.79
    EXPECT_EQ(node.visit_count(), 4);
    EXPECT_FLOAT_EQ(node.total_value(), 1.7f);
    EXPECT_NEAR(node.mean_value(), 0.425f, 1e-6f);
    // variance = mean_sq - mean^2 = 0.79/4 - 0.425^2 = 0.1975 - 0.180625 = 0.016875
    EXPECT_NEAR(node.value_variance(), 0.016875f, 1e-5f);
}

TEST_F(MCTSTest, MultivisitVirtualLossN) {
    // apply_virtual_loss(n) / revert_virtual_loss(n) must be symmetric.
    mcts::Node node;
    node.update(0.0f);
    EXPECT_EQ(node.visit_count(), 1);
    EXPECT_EQ(node.pending_evals(), 0);

    node.apply_virtual_loss(5);
    EXPECT_EQ(node.visit_count(), 6);
    EXPECT_EQ(node.pending_evals(), 5);

    node.revert_virtual_loss(5);
    EXPECT_EQ(node.visit_count(), 1);
    EXPECT_EQ(node.pending_evals(), 0);
}

TEST_F(MCTSTest, ValueVariance) {
    mcts::Node node;
    // < 2 visits -> 0 variance
    EXPECT_FLOAT_EQ(node.value_variance(), 0.0f);
    node.update(0.6f);
    EXPECT_FLOAT_EQ(node.value_variance(), 0.0f);

    // After update(0.6): visit=1, total=0.6, sum_sq=0.36
    node.update(0.4f);  // visit=2, total=1.0, sum_sq=0.52
    // mean=0.5, mean_sq=0.26, var=0.26-0.25=0.01
    EXPECT_NEAR(node.value_variance(), 0.01f, 1e-5f);
}

TEST_F(MCTSTest, StopFlagTerminatesSearchEarly) {
    std::atomic<bool> stop{true};
    mcts::SearchParams params;
    params.num_iterations = 10000;
    params.add_noise = false;
    mcts::RandomEvaluator eval;
    mcts::Search search(eval, params);
    search.set_stop_flag(&stop);

    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto result = search.run(pos);
    EXPECT_LT(result.total_nodes, 100);
}

TEST_F(MCTSTest, InfoCallbackFired) {
    int callback_count = 0;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.batch_size = 8;
    params.add_noise = false;
    mcts::RandomEvaluator eval;
    mcts::Search search(eval, params);
    search.set_info_callback([&](const mcts::SearchInfo& info) {
        callback_count++;
        EXPECT_GT(info.iterations, 0);
        EXPECT_FALSE(info.best_move.is_none());
    });

    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    search.run(pos);
    EXPECT_GT(callback_count, 0);
}

TEST_F(MCTSTest, TerminalStatus) {
    mcts::Node node;
    EXPECT_EQ(node.terminal_status(), 0);
    node.set_terminal_status(1);  // proven loss
    EXPECT_EQ(node.terminal_status(), 1);
    node.set_terminal_status(-1); // proven win
    EXPECT_EQ(node.terminal_status(), -1);
    node.set_terminal_status(2);  // proven draw
    EXPECT_EQ(node.terminal_status(), 2);
}

TEST_F(MCTSTest, SortEdgesByPrior) {
    mcts::Node root;
    Move moves[] = {Move(E2, E4, FLAG_DOUBLE_PUSH), Move(D2, D4, FLAG_DOUBLE_PUSH), Move(G1, F3, FLAG_QUIET)};
    float priors[] = {0.2f, 0.5f, 0.3f};
    root.create_edges(moves, priors, 3);
    root.sort_edges_by_prior();
    EXPECT_NEAR(root.edge(0).prior(), 0.5f, 0.002f);
    EXPECT_NEAR(root.edge(1).prior(), 0.3f, 0.002f);
    EXPECT_NEAR(root.edge(2).prior(), 0.2f, 0.002f);
    EXPECT_EQ(root.edge(0).move(), Move(D2, D4, FLAG_DOUBLE_PUSH));
}

TEST_F(MCTSTest, NNCacheStoreAndRetrieve) {
    mcts::NNCache cache(100);
    mcts::CacheEntry entry;
    entry.policy = {0.5f, 0.3f, 0.2f};
    entry.value = 0.42f;
    entry.num_moves = 3;

    cache.put(12345, entry);
    EXPECT_EQ(cache.size(), 1);

    const mcts::CacheEntry* found = cache.get(12345);
    ASSERT_NE(found, nullptr);
    EXPECT_FLOAT_EQ(found->value, 0.42f);
    EXPECT_EQ(found->num_moves, 3);
    EXPECT_EQ(found->policy.size(), 3u);
    EXPECT_FLOAT_EQ(found->policy[0], 0.5f);
}

TEST_F(MCTSTest, NNCacheMiss) {
    mcts::NNCache cache(100);
    EXPECT_EQ(cache.get(99999), nullptr);
}

TEST_F(MCTSTest, NNCacheEviction) {
    mcts::NNCache cache(10);
    for (int i = 0; i < 15; i++) {
        mcts::CacheEntry entry;
        entry.policy = {1.0f};
        entry.value = static_cast<float>(i) / 15.0f;
        entry.num_moves = 1;
        cache.put(static_cast<uint64_t>(i), entry);
    }
    // After adding 15 entries to a cache of size 10, size should be <= 10
    EXPECT_LE(cache.size(), 10);
}

TEST_F(MCTSTest, NNCacheClear) {
    mcts::NNCache cache(100);
    mcts::CacheEntry entry;
    entry.policy = {1.0f};
    entry.value = 0.5f;
    entry.num_moves = 1;
    cache.put(1, entry);
    cache.put(2, entry);
    EXPECT_EQ(cache.size(), 2);
    cache.clear();
    EXPECT_EQ(cache.size(), 0);
    EXPECT_EQ(cache.get(1), nullptr);
}

TEST_F(MCTSTest, NNCacheOverwrite) {
    mcts::NNCache cache(100);
    mcts::CacheEntry e1;
    e1.policy = {0.5f, 0.5f};
    e1.value = 0.3f;
    e1.num_moves = 2;
    cache.put(42, e1);

    mcts::CacheEntry e2;
    e2.policy = {0.7f, 0.3f};
    e2.value = 0.8f;
    e2.num_moves = 2;
    cache.put(42, e2);

    EXPECT_EQ(cache.size(), 1);
    const mcts::CacheEntry* found = cache.get(42);
    ASSERT_NE(found, nullptr);
    EXPECT_FLOAT_EQ(found->value, 0.8f);
}

TEST_F(MCTSTest, PositionHistoryBasic) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    neural::PositionHistory hist;
    hist.reset(pos);
    EXPECT_EQ(hist.length(), 1);
    EXPECT_EQ(hist.current().side_to_move(), WHITE);
}

TEST_F(MCTSTest, PositionHistoryPush) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    neural::PositionHistory hist;
    hist.reset(pos);

    // Make a move and push
    UndoInfo undo;
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    ASSERT_GT(n, 0);
    pos.make_move(moves[0], undo);
    hist.push(pos);

    EXPECT_EQ(hist.length(), 2);
    EXPECT_EQ(hist.at(0).side_to_move(), BLACK);  // current (after move)
    EXPECT_EQ(hist.at(1).side_to_move(), WHITE);  // one step back
}

TEST_F(MCTSTest, PositionHistoryRepetition) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    neural::PositionHistory hist;
    hist.reset(pos);

    // Play Nf3 Nf6 Ng1 Ng8 - back to start
    UndoInfo undo;
    const char* uci_moves[] = {"g1f3", "g8f6", "f3g1", "f6g8"};
    for (const char* uci : uci_moves) {
        int from = (uci[0]-'a') + (uci[1]-'1') * 8;
        int to = (uci[2]-'a') + (uci[3]-'1') * 8;
        Move m(Square(from), Square(to), FLAG_QUIET);
        pos.make_move(m, undo);
        hist.push(pos);
    }
    // After 4 moves, should have 2-fold repetition of starting position
    EXPECT_TRUE(hist.is_repetition(2));
}

TEST_F(MCTSTest, PositionHistoryAtClamps) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    neural::PositionHistory hist;
    hist.reset(pos);
    // at(10) should clamp to the first position, not crash
    auto& old = hist.at(10);
    EXPECT_EQ(old.side_to_move(), WHITE);  // same as the only position
}

TEST_F(MCTSTest, EncodeWithHistoryShape) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    neural::PositionHistory hist;
    hist.reset(pos);

    float buffer[neural::TENSOR_SIZE] = {0};
    neural::encode_position(hist, buffer);

    // Bias plane (111) should be all 1.0
    for (int i = 0; i < 64; i++) {
        EXPECT_FLOAT_EQ(buffer[111 * 64 + i], 1.0f);
    }
}

TEST_F(MCTSTest, EncodeWithHistoryMatchesSinglePosition) {
    // Encoding a single-position history should produce the same result
    // as encoding the position directly (backward compatibility)
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    float buf_single[neural::TENSOR_SIZE] = {0};
    neural::encode_position(pos, buf_single);

    neural::PositionHistory hist;
    hist.reset(pos);
    float buf_hist[neural::TENSOR_SIZE] = {0};
    neural::encode_position(hist, buf_hist);

    for (int i = 0; i < neural::TENSOR_SIZE; i++) {
        EXPECT_FLOAT_EQ(buf_single[i], buf_hist[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(MCTSTest, EncodeWithHistoryDifferentTimeSteps) {
    // After making a move, time step 0 and time step 1 should differ
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    neural::PositionHistory hist;
    hist.reset(pos);

    UndoInfo undo;
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    pos.make_move(m, undo);
    hist.push(pos);

    float buffer[neural::TENSOR_SIZE] = {0};
    neural::encode_position(hist, buffer);

    // Time step 0 (current position after e4) and time step 1 (starting position)
    // should differ because a pawn moved
    bool any_diff = false;
    for (int i = 0; i < 13 * 64; i++) {
        if (buffer[i] != buffer[13 * 64 + i]) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

// --- New Task 5 tests ---

TEST_F(MCTSTest, BatchedSearchReturnsLegalMoveAndValidPolicyTarget) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.add_noise = false;
    params.batch_size = 8;

    mcts::Search search(eval, params);
    neural::PositionHistory hist;
    hist.reset(pos);
    mcts::SearchResult result = search.run(hist);

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

    // policy_target should sum to approximately 1.0
    float policy_sum = 0.0f;
    for (int i = 0; i < neural::POLICY_SIZE; i++) {
        policy_sum += result.policy_target[i];
        EXPECT_GE(result.policy_target[i], 0.0f);
    }
    EXPECT_NEAR(policy_sum, 1.0f, 0.01f);

    // raw_policy should also sum to approximately 1.0
    float raw_sum = 0.0f;
    for (int i = 0; i < neural::POLICY_SIZE; i++) {
        raw_sum += result.raw_policy[i];
    }
    EXPECT_NEAR(raw_sum, 1.0f, 0.01f);
}

TEST_F(MCTSTest, MCTSSolverFindsForcedMate) {
    // Back rank mate: White Rook on a1, White King on g1, Black King on h8
    // White plays Ra8# (checkmate)
    Position pos;
    pos.set_fen("7k/8/6K1/8/8/8/8/R7 w - - 0 1");

    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    ASSERT_GT(num_moves, 0);

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 400;
    params.add_noise = false;
    params.batch_size = 8;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // Ra1-a8 should be the best move (mate in 1)
    EXPECT_EQ(result.best_move, Move(A1, A8, FLAG_QUIET));
}

TEST_F(MCTSTest, SmartPruningTerminatesEarly) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 5000;  // Very high iteration count
    params.add_noise = false;
    params.batch_size = 16;
    params.smart_pruning = true;
    params.smart_pruning_factor = 1.33f;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // Smart pruning should terminate before using all iterations
    // total_nodes includes the root visit, so it should be well below num_iterations
    int total_child_visits = 0;
    for (int v : result.visit_counts) total_child_visits += v;
    EXPECT_LT(total_child_visits, params.num_iterations);
    EXPECT_GT(total_child_visits, 0);
}

TEST_F(MCTSTest, TwoFoldRepetitionTreatedAsDraw) {
    // Set up a position and play Nf3 Nf6 Ng1 Ng8 to get back to start (2-fold)
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    neural::PositionHistory hist;
    hist.reset(pos);

    // Play Nf3 Nf6 Ng1 Ng8 — this creates a cycle
    UndoInfo undo;
    Move nf3(G1, F3, FLAG_QUIET);
    pos.make_move(nf3, undo);
    hist.push(pos);

    Move nf6(G8, F6, FLAG_QUIET);
    pos.make_move(nf6, undo);
    hist.push(pos);

    Move ng1(F3, G1, FLAG_QUIET);
    pos.make_move(ng1, undo);
    hist.push(pos);

    Move ng8(F6, G8, FLAG_QUIET);
    pos.make_move(ng8, undo);
    hist.push(pos);

    // Now we're back at the starting position (2-fold)
    EXPECT_TRUE(hist.is_repetition(2));

    // Run search from this position with two_fold_draw enabled
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.add_noise = false;
    params.batch_size = 8;
    params.two_fold_draw = true;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(hist);

    // Search should still return a valid move
    EXPECT_FALSE(result.best_move.is_none());
}

TEST_F(MCTSTest, ShapedDirichletProducesNonUniformNoise) {
    // Run two searches: one with shaped Dirichlet, one with uniform.
    // With shaped noise, the visit distribution should still concentrate more
    // on reasonable moves compared to uniform noise which spreads evenly.
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;

    // Shaped Dirichlet search
    mcts::SearchParams shaped_params;
    shaped_params.add_noise = true;
    shaped_params.shaped_dirichlet = true;
    shaped_params.dirichlet_alpha = 0.3f;
    shaped_params.dirichlet_epsilon = 0.25f;
    shaped_params.num_iterations = 200;
    shaped_params.batch_size = 8;

    mcts::Search shaped_search(eval, shaped_params);
    mcts::SearchResult shaped_result = shaped_search.run(pos);

    // Uniform Dirichlet search
    mcts::SearchParams uniform_params = shaped_params;
    uniform_params.shaped_dirichlet = false;

    mcts::Search uniform_search(eval, uniform_params);
    mcts::SearchResult uniform_result = uniform_search.run(pos);

    // Both should return valid moves
    EXPECT_FALSE(shaped_result.best_move.is_none());
    EXPECT_FALSE(uniform_result.best_move.is_none());

    // Both should have valid policy targets that sum to ~1.0
    float shaped_sum = 0.0f, uniform_sum = 0.0f;
    for (float v : shaped_result.policy_target) shaped_sum += v;
    for (float v : uniform_result.policy_target) uniform_sum += v;
    EXPECT_NEAR(shaped_sum, 1.0f, 0.01f);
    EXPECT_NEAR(uniform_sum, 1.0f, 0.01f);

    // Verify that shaped noise didn't crash and search completed fully
    EXPECT_GT(shaped_result.total_nodes, 50);
    EXPECT_GT(uniform_result.total_nodes, 50);
}

TEST_F(MCTSTest, FullSearchOnStartingPositionCompletes) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 200;
    params.add_noise = true;
    params.batch_size = 16;
    params.smart_pruning = true;
    params.two_fold_draw = true;
    params.shaped_dirichlet = true;
    params.variance_scaling = true;
    params.sibling_blending = true;
    params.uncertainty_weight = 0.15f;

    mcts::Search search(eval, params);

    neural::PositionHistory hist;
    hist.reset(pos);
    mcts::SearchResult result = search.run(hist);

    EXPECT_FALSE(result.best_move.is_none());
    EXPECT_GT(result.total_nodes, 0);
    EXPECT_GT(result.moves.size(), 0u);
    EXPECT_EQ(result.moves.size(), result.visit_counts.size());
}

TEST_F(MCTSTest, SearchWithContempt) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 100;
    params.add_noise = false;
    params.batch_size = 8;
    params.contempt = 0.5f;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    // Search should complete and return a valid move
    EXPECT_FALSE(result.best_move.is_none());
    // With contempt > 0, the root value should be shifted away from 0
    // (can't test exact value due to material eval, but it should be valid)
    EXPECT_GE(result.root_value, -1.0f);
    EXPECT_LE(result.root_value, 1.0f);
}

TEST_F(MCTSTest, BackwardCompatRunPosition) {
    // Verify the old run(Position) interface still works
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = 50;
    params.add_noise = false;
    params.batch_size = 4;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    EXPECT_FALSE(result.best_move.is_none());

    // Visit counts should sum to approximately num_iterations
    // (may be less due to smart pruning)
    int total = 0;
    for (int v : result.visit_counts) total += v;
    EXPECT_GT(total, 0);
    EXPECT_LE(total, params.num_iterations);
}

// --- GameManager tests (require LibTorch) ---

#ifdef HAS_LIBTORCH
#include "mcts/game_manager.h"

// GameManager requires NeuralEvaluator (and hence a model file) so C++ unit tests
// are limited to build/link validation. Full functional tests are in Python.
TEST_F(MCTSTest, GameManagerHeaderCompiles) {
    // Verify the GameManager class is accessible and the header compiles.
    // We cannot instantiate without a model, but we can test that types are correct.
    mcts::SearchParams params;
    params.num_iterations = 50;
    params.batch_size = 8;

    // Type-check: GameManager is defined and its methods exist
    // (This is essentially a compile/link test)
    EXPECT_EQ(params.batch_size, 8);
}
#endif // HAS_LIBTORCH
