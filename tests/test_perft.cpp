#include <gtest/gtest.h>
#include "core/movegen.h"
#include "core/attacks.h"

class PerftTest : public ::testing::Test {
protected:
    void SetUp() override { attacks::init(); }

    uint64_t perft(Position& pos, int depth) {
        if (depth == 0) return 1;
        Move moves[MAX_MOVES];
        int n = generate_legal_moves(pos, moves);
        if (depth == 1) return n;
        uint64_t nodes = 0;
        for (int i = 0; i < n; ++i) {
            UndoInfo undo;
            pos.make_move(moves[i], undo);
            nodes += perft(pos, depth - 1);
            pos.unmake_move(moves[i], undo);
        }
        return nodes;
    }
};

// Position 1: Starting position
TEST_F(PerftTest, StartingPos_Depth1) {
    Position pos; pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 1), 20ULL);
}
TEST_F(PerftTest, StartingPos_Depth2) {
    Position pos; pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 2), 400ULL);
}
TEST_F(PerftTest, StartingPos_Depth3) {
    Position pos; pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 3), 8902ULL);
}
TEST_F(PerftTest, StartingPos_Depth4) {
    Position pos; pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 4), 197281ULL);
}
TEST_F(PerftTest, StartingPos_Depth5) {
    Position pos; pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 5), 4865609ULL);
}

// Position 2: Kiwipete
TEST_F(PerftTest, Kiwipete_Depth1) {
    Position pos; pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 1), 48ULL);
}
TEST_F(PerftTest, Kiwipete_Depth2) {
    Position pos; pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 2), 2039ULL);
}
TEST_F(PerftTest, Kiwipete_Depth3) {
    Position pos; pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 3), 97862ULL);
}
TEST_F(PerftTest, Kiwipete_Depth4) {
    Position pos; pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 4), 4085603ULL);
}

// Position 3
TEST_F(PerftTest, Position3_Depth1) {
    Position pos; pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 1), 14ULL);
}
TEST_F(PerftTest, Position3_Depth2) {
    Position pos; pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 2), 191ULL);
}
TEST_F(PerftTest, Position3_Depth3) {
    Position pos; pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 3), 2812ULL);
}
TEST_F(PerftTest, Position3_Depth4) {
    Position pos; pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 4), 43238ULL);
}
TEST_F(PerftTest, Position3_Depth5) {
    Position pos; pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 5), 674624ULL);
}

// Position 4
TEST_F(PerftTest, Position4_Depth1) {
    Position pos; pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 1), 6ULL);
}
TEST_F(PerftTest, Position4_Depth2) {
    Position pos; pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 2), 264ULL);
}
TEST_F(PerftTest, Position4_Depth3) {
    Position pos; pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 3), 9467ULL);
}
TEST_F(PerftTest, Position4_Depth4) {
    Position pos; pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 4), 422333ULL);
}

// Position 5
TEST_F(PerftTest, Position5_Depth1) {
    Position pos; pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 1), 44ULL);
}
TEST_F(PerftTest, Position5_Depth2) {
    Position pos; pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 2), 1486ULL);
}
TEST_F(PerftTest, Position5_Depth3) {
    Position pos; pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 3), 62379ULL);
}
TEST_F(PerftTest, Position5_Depth4) {
    Position pos; pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 4), 2103487ULL);
}
