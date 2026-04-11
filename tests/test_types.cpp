#include <gtest/gtest.h>
#include "core/types.h"

TEST(Types, SquareCoordinates) {
    EXPECT_EQ(rank_of(A1), 0);
    EXPECT_EQ(file_of(A1), 0);
    EXPECT_EQ(rank_of(H8), 7);
    EXPECT_EQ(file_of(H8), 7);
    EXPECT_EQ(make_square(4, 3), E4);
    EXPECT_EQ(make_square(0, 7), A8);
}

TEST(Types, ColorFlip) {
    EXPECT_EQ(~WHITE, BLACK);
    EXPECT_EQ(~BLACK, WHITE);
}

TEST(Types, MoveEncodeDecode) {
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    EXPECT_EQ(m.from(), E2);
    EXPECT_EQ(m.to(), E4);
    EXPECT_EQ(m.flag(), FLAG_DOUBLE_PUSH);
    EXPECT_FALSE(m.is_capture());
    EXPECT_FALSE(m.is_promotion());
}

TEST(Types, MoveCapture) {
    Move m(D4, E5, FLAG_CAPTURE);
    EXPECT_TRUE(m.is_capture());
    EXPECT_FALSE(m.is_promotion());
}

TEST(Types, MovePromotion) {
    Move m(A7, A8, FLAG_PROMO_QUEEN);
    EXPECT_TRUE(m.is_promotion());
    EXPECT_FALSE(m.is_capture());
    EXPECT_EQ(m.promo_piece(), QUEEN);
}

TEST(Types, MovePromotionCapture) {
    Move m(B7, A8, FLAG_PROMO_CAP_N);
    EXPECT_TRUE(m.is_promotion());
    EXPECT_TRUE(m.is_capture());
    EXPECT_EQ(m.promo_piece(), KNIGHT);
}

TEST(Types, MoveCastle) {
    Move m(E1, G1, FLAG_KING_CASTLE);
    EXPECT_TRUE(m.is_castle());
    EXPECT_FALSE(m.is_capture());
}

TEST(Types, MoveNone) {
    EXPECT_TRUE(Move::none().is_none());
    Move m(A2, A3);
    EXPECT_FALSE(m.is_none());
}

TEST(Types, CastlingBitOps) {
    CastlingRight cr = WHITE_OO | BLACK_OOO;
    EXPECT_EQ(cr & WHITE_OO, WHITE_OO);
    EXPECT_EQ(cr & WHITE_OOO, NO_CASTLING);
    EXPECT_EQ(cr & BLACK_OOO, BLACK_OOO);
}

TEST(Types, MoveToUci) {
    EXPECT_EQ(Move(E2, E4, FLAG_DOUBLE_PUSH).to_uci(), "e2e4");
    EXPECT_EQ(Move(A7, A8, FLAG_PROMO_QUEEN).to_uci(), "a7a8q");
    EXPECT_EQ(Move(E1, G1, FLAG_KING_CASTLE).to_uci(), "e1g1");
}
