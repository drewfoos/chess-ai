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
