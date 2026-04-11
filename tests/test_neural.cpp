// tests/test_neural.cpp
#include <gtest/gtest.h>
#include "neural/policy_map.h"
#include "core/types.h"
#include "core/attacks.h"

class NeuralTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { attacks::init(); }
};

// Reference values from Python: training/encoder.py move_to_index()
TEST_F(NeuralTest, PolicyMap_E2E4) {
    // E2=12, E4=28, queen-move encoding
    EXPECT_EQ(neural::move_to_policy_index(E2, E4, NO_PIECE_TYPE), 304);
}

TEST_F(NeuralTest, PolicyMap_KnightG1F3) {
    // G1=6, F3=21
    EXPECT_EQ(neural::move_to_policy_index(G1, F3, NO_PIECE_TYPE), 170);
}

TEST_F(NeuralTest, PolicyMap_QueenPromoE7E8) {
    // Queen promo uses normal queen-move encoding (promo=NO_PIECE_TYPE)
    EXPECT_EQ(neural::move_to_policy_index(E7, E8, NO_PIECE_TYPE), 1522);
}

TEST_F(NeuralTest, PolicyMap_UnderpromoKnightE7E8) {
    EXPECT_EQ(neural::move_to_policy_index(E7, E8, KNIGHT), 1554);
}

TEST_F(NeuralTest, PolicyMap_UnderpromoRookE7D8) {
    EXPECT_EQ(neural::move_to_policy_index(E7, D8, ROOK), 1553);
}

TEST_F(NeuralTest, PolicyMap_A2A3) {
    EXPECT_EQ(neural::move_to_policy_index(A2, A3, NO_PIECE_TYPE), 194);
}

TEST_F(NeuralTest, PolicyMap_B1C3) {
    EXPECT_EQ(neural::move_to_policy_index(B1, C3, NO_PIECE_TYPE), 44);
}

TEST_F(NeuralTest, PolicyMap_D1H5) {
    EXPECT_EQ(neural::move_to_policy_index(D1, H5, NO_PIECE_TYPE), 82);
}

TEST_F(NeuralTest, PolicyMap_TotalSize) {
    EXPECT_EQ(neural::POLICY_SIZE, 1858);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_White) {
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    EXPECT_EQ(neural::move_to_policy_index(m, WHITE), 304);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_BlackMirrors) {
    // Black E7->E5: mirror to E2->E4
    Move m(E7, E5, FLAG_DOUBLE_PUSH);
    EXPECT_EQ(neural::move_to_policy_index(m, BLACK), 304);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_QueenPromo) {
    // Queen promo flag -> NO_PIECE_TYPE (uses normal encoding)
    Move m(E7, E8, FLAG_PROMO_QUEEN);
    EXPECT_EQ(neural::move_to_policy_index(m, WHITE), 1522);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_KnightPromo) {
    Move m(E7, E8, FLAG_PROMO_KNIGHT);
    EXPECT_EQ(neural::move_to_policy_index(m, WHITE), 1554);
}

TEST_F(NeuralTest, PolicyMap_InvalidReturnsNeg1) {
    // Same square -> invalid
    EXPECT_EQ(neural::move_to_policy_index(A1, A1, NO_PIECE_TYPE), -1);
}
