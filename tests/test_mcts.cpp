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
