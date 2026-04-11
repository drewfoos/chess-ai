#include <gtest/gtest.h>
#include "core/bitboard.h"

TEST(Bitboard, SquareBB) {
    EXPECT_EQ(square_bb(A1), 1ULL);
    EXPECT_EQ(square_bb(H1), 1ULL << 7);
    EXPECT_EQ(square_bb(A8), 1ULL << 56);
    EXPECT_EQ(square_bb(H8), 1ULL << 63);
}

TEST(Bitboard, RankFileBB) {
    EXPECT_EQ(RANK_BB[0], 0xFFULL);
    EXPECT_EQ(RANK_BB[7], 0xFFULL << 56);
    EXPECT_EQ(FILE_BB[0], 0x0101010101010101ULL);
    EXPECT_EQ(FILE_BB[7], 0x8080808080808080ULL);
}

TEST(Bitboard, Popcount) {
    EXPECT_EQ(popcount(0ULL), 0);
    EXPECT_EQ(popcount(0xFFULL), 8);
    EXPECT_EQ(popcount(0xFFFFFFFFFFFFFFFFULL), 64);
    EXPECT_EQ(popcount(square_bb(E4)), 1);
}

TEST(Bitboard, LSB) {
    EXPECT_EQ(lsb(1ULL), A1);
    EXPECT_EQ(lsb(1ULL << 63), H8);
    EXPECT_EQ(lsb(0x0100000000ULL), A5);
}

TEST(Bitboard, PopLSB) {
    Bitboard bb = square_bb(A1) | square_bb(C3) | square_bb(H8);
    Square s1 = pop_lsb(bb);
    EXPECT_EQ(s1, A1);
    Square s2 = pop_lsb(bb);
    EXPECT_EQ(s2, C3);
    Square s3 = pop_lsb(bb);
    EXPECT_EQ(s3, H8);
    EXPECT_EQ(bb, 0ULL);
}

TEST(Bitboard, ShiftNorth) {
    Bitboard rank1 = RANK_BB[0];
    EXPECT_EQ(shift_bb<NORTH>(rank1), RANK_BB[1]);
}

TEST(Bitboard, ShiftSouth) {
    Bitboard rank2 = RANK_BB[1];
    EXPECT_EQ(shift_bb<SOUTH>(rank2), RANK_BB[0]);
}

TEST(Bitboard, ShiftEastWraparound) {
    Bitboard h_file = FILE_BB[7];
    EXPECT_EQ(shift_bb<EAST>(h_file), 0ULL);
}

TEST(Bitboard, ShiftWestWraparound) {
    Bitboard a_file = FILE_BB[0];
    EXPECT_EQ(shift_bb<WEST>(a_file), 0ULL);
}

TEST(Bitboard, MultipleBits) {
    Bitboard bb = square_bb(A1) | square_bb(B2) | square_bb(C3);
    EXPECT_EQ(popcount(bb), 3);
    EXPECT_TRUE(bb & square_bb(B2));
    EXPECT_FALSE(bb & square_bb(D4));
}
