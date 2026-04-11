#include <gtest/gtest.h>
#include "core/movegen.h"
#include "core/attacks.h"

class MoveGenTest : public ::testing::Test {
protected:
    void SetUp() override { attacks::init(); }

    int count_legal_moves(const std::string& fen) {
        Position pos;
        pos.set_fen(fen);
        Move moves[MAX_MOVES];
        return generate_legal_moves(pos, moves);
    }

    bool has_move(const std::string& fen, const std::string& uci) {
        Position pos;
        pos.set_fen(fen);
        Move moves[MAX_MOVES];
        int n = generate_legal_moves(pos, moves);
        for (int i = 0; i < n; ++i)
            if (moves[i].to_uci() == uci) return true;
        return false;
    }
};

TEST_F(MoveGenTest, StartingPosition) {
    EXPECT_EQ(count_legal_moves("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 20);
}

TEST_F(MoveGenTest, PawnDoublePush) {
    EXPECT_TRUE(has_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"));
}

TEST_F(MoveGenTest, PawnCapture) {
    EXPECT_TRUE(has_move("8/8/8/3p4/4P3/8/8/4K2k w - - 0 1", "e4d5"));
}

TEST_F(MoveGenTest, EnPassant) {
    EXPECT_TRUE(has_move("rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "e5f6"));
}

TEST_F(MoveGenTest, Promotion) {
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8q"));
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8r"));
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8b"));
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8n"));
}

TEST_F(MoveGenTest, CastlingKingside) {
    EXPECT_TRUE(has_move("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, CastlingQueenside) {
    EXPECT_TRUE(has_move("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1"));
}

TEST_F(MoveGenTest, CastlingBlockedByPiece) {
    EXPECT_FALSE(has_move("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3KB1R w KQkq - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, CastlingThroughCheck) {
    EXPECT_FALSE(has_move("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, CastlingOutOfCheck) {
    EXPECT_FALSE(has_move("4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, MustEscapeCheck) {
    int n = count_legal_moves("4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    EXPECT_GT(n, 0);
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    Move moves[MAX_MOVES];
    int count = generate_legal_moves(pos, moves);
    for (int i = 0; i < count; ++i)
        EXPECT_EQ(pos.piece_on(moves[i].from()), KING);
}

TEST_F(MoveGenTest, StalemateNoMoves) {
    EXPECT_EQ(count_legal_moves("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1"), 0);
}

TEST_F(MoveGenTest, PinnedPieceCannotMove) {
    EXPECT_FALSE(has_move("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1", "e2d4"));
    EXPECT_FALSE(has_move("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1", "e2f4"));
}
