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
