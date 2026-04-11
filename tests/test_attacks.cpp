#include <gtest/gtest.h>
#include "core/attacks.h"

class AttacksTest : public ::testing::Test {
protected:
    void SetUp() override { attacks::init(); }
};

TEST_F(AttacksTest, KnightCenter) {
    Bitboard atk = attacks::knight(E4);
    EXPECT_EQ(popcount(atk), 8);
    EXPECT_TRUE(atk & square_bb(D6));
    EXPECT_TRUE(atk & square_bb(F6));
    EXPECT_TRUE(atk & square_bb(G5));
    EXPECT_TRUE(atk & square_bb(G3));
    EXPECT_TRUE(atk & square_bb(F2));
    EXPECT_TRUE(atk & square_bb(D2));
    EXPECT_TRUE(atk & square_bb(C3));
    EXPECT_TRUE(atk & square_bb(C5));
}

TEST_F(AttacksTest, KnightCorner) {
    Bitboard atk = attacks::knight(A1);
    EXPECT_EQ(popcount(atk), 2);
    EXPECT_TRUE(atk & square_bb(B3));
    EXPECT_TRUE(atk & square_bb(C2));
}

TEST_F(AttacksTest, KingCenter) {
    Bitboard atk = attacks::king(E4);
    EXPECT_EQ(popcount(atk), 8);
    EXPECT_TRUE(atk & square_bb(D5));
    EXPECT_TRUE(atk & square_bb(E5));
    EXPECT_TRUE(atk & square_bb(F5));
    EXPECT_TRUE(atk & square_bb(D4));
    EXPECT_TRUE(atk & square_bb(F4));
    EXPECT_TRUE(atk & square_bb(D3));
    EXPECT_TRUE(atk & square_bb(E3));
    EXPECT_TRUE(atk & square_bb(F3));
}

TEST_F(AttacksTest, KingCorner) {
    Bitboard atk = attacks::king(A1);
    EXPECT_EQ(popcount(atk), 3);
}

TEST_F(AttacksTest, WhitePawnCenter) {
    Bitboard atk = attacks::pawn(WHITE, E4);
    EXPECT_EQ(popcount(atk), 2);
    EXPECT_TRUE(atk & square_bb(D5));
    EXPECT_TRUE(atk & square_bb(F5));
}

TEST_F(AttacksTest, WhitePawnAFile) {
    Bitboard atk = attacks::pawn(WHITE, A4);
    EXPECT_EQ(popcount(atk), 1);
    EXPECT_TRUE(atk & square_bb(B5));
}

TEST_F(AttacksTest, BlackPawnCenter) {
    Bitboard atk = attacks::pawn(BLACK, E5);
    EXPECT_EQ(popcount(atk), 2);
    EXPECT_TRUE(atk & square_bb(D4));
    EXPECT_TRUE(atk & square_bb(F4));
}

TEST_F(AttacksTest, BishopOpenBoard) {
    Bitboard atk = attacks::bishop(E4, 0ULL);
    EXPECT_TRUE(atk & square_bb(D5));
    EXPECT_TRUE(atk & square_bb(A8));
    EXPECT_TRUE(atk & square_bb(H7));
    EXPECT_TRUE(atk & square_bb(H1));
    EXPECT_TRUE(atk & square_bb(B1));
    EXPECT_FALSE(atk & square_bb(E5));
}

TEST_F(AttacksTest, BishopBlocker) {
    Bitboard occ = square_bb(F5);
    Bitboard atk = attacks::bishop(E4, occ);
    EXPECT_TRUE(atk & square_bb(F5));
    EXPECT_FALSE(atk & square_bb(G6));
}

TEST_F(AttacksTest, RookOpenBoard) {
    Bitboard atk = attacks::rook(E4, 0ULL);
    EXPECT_TRUE(atk & square_bb(E1));
    EXPECT_TRUE(atk & square_bb(E8));
    EXPECT_TRUE(atk & square_bb(A4));
    EXPECT_TRUE(atk & square_bb(H4));
    EXPECT_FALSE(atk & square_bb(D5));
}

TEST_F(AttacksTest, RookBlocker) {
    Bitboard occ = square_bb(E6);
    Bitboard atk = attacks::rook(E4, occ);
    EXPECT_TRUE(atk & square_bb(E5));
    EXPECT_TRUE(atk & square_bb(E6));
    EXPECT_FALSE(atk & square_bb(E7));
}

TEST_F(AttacksTest, QueenIsRookPlusBishop) {
    Bitboard occ = square_bb(G6) | square_bb(C2);
    Bitboard q = attacks::queen(E4, occ);
    Bitboard r = attacks::rook(E4, occ);
    Bitboard b = attacks::bishop(E4, occ);
    EXPECT_EQ(q, r | b);
}
