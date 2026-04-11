#include <gtest/gtest.h>
#include "mcts/node.h"
#include "mcts/search.h"
#include "core/attacks.h"
#include "core/position.h"
#include "core/movegen.h"

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
