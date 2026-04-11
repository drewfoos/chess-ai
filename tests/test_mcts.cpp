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
