#include <gtest/gtest.h>
#include "uci/time_manager.h"

TEST(TimeManager, MovetimeConvertsToIterations) {
    uci::TimeControl tc;
    tc.movetime_ms = 1000;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_EQ(alloc.iterations, 500);
    EXPECT_EQ(alloc.soft_time_ms, 1000);
}

TEST(TimeManager, InfiniteReturnsLargeCount) {
    uci::TimeControl tc;
    tc.infinite = true;
    auto alloc = uci::allocate_time(tc, WHITE);
    EXPECT_GE(alloc.iterations, 100000);
}

TEST(TimeManager, MaxNodesPassedThrough) {
    uci::TimeControl tc;
    tc.max_nodes = 400;
    auto alloc = uci::allocate_time(tc, WHITE);
    EXPECT_EQ(alloc.iterations, 400);
}

TEST(TimeManager, ClockBasedAllocation) {
    uci::TimeControl tc;
    tc.wtime_ms = 60000;
    tc.winc_ms = 1000;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_GT(alloc.iterations, 500);
    EXPECT_LT(alloc.iterations, 3000);
}

TEST(TimeManager, NeverExceedsHalfRemainingTime) {
    uci::TimeControl tc;
    tc.wtime_ms = 1000;
    tc.winc_ms = 0;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_LE(alloc.soft_time_ms, 500);
}

TEST(TimeManager, BlackUsesBlackClock) {
    uci::TimeControl tc;
    tc.wtime_ms = 60000;
    tc.btime_ms = 10000;
    auto alloc = uci::allocate_time(tc, BLACK, 500);
    EXPECT_LT(alloc.iterations, 500);
}

// --- UCI Protocol Handler Tests ---

#include "uci/uci.h"
#include "core/attacks.h"

class UCITest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { attacks::init(); }
};

TEST_F(UCITest, UciCommandReturnsUciok) {
    std::istringstream in("uci\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("id name ChessAI"), std::string::npos);
    EXPECT_NE(out.str().find("uciok"), std::string::npos);
}

TEST_F(UCITest, IsReadyReturnsReadyok) {
    std::istringstream in("isready\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("readyok"), std::string::npos);
}

TEST_F(UCITest, PositionStartposAndGo) {
    std::istringstream in("position startpos\ngo nodes 50\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, PositionStartposMoves) {
    std::istringstream in("position startpos moves e2e4 e7e5\ngo nodes 50\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, PositionFen) {
    std::istringstream in(
        "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n"
        "go nodes 50\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, GoInfiniteAndStop) {
    std::istringstream in("position startpos\ngo infinite\nstop\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}
