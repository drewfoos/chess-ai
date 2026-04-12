#pragma once
#include "core/types.h"
#include <algorithm>

namespace uci {

struct TimeControl {
    int wtime_ms = 0;
    int btime_ms = 0;
    int winc_ms = 0;
    int binc_ms = 0;
    int movestogo = 0;
    int movetime_ms = 0;
    int max_depth = 0;
    int max_nodes = 0;
    bool infinite = false;
};

struct TimeAllocation {
    int iterations = 0;
    int soft_time_ms = 0;
};

inline TimeAllocation allocate_time(const TimeControl& tc, Color side_to_move,
                                     int nps_estimate = 500) {
    TimeAllocation alloc{};

    if (tc.infinite) {
        alloc.iterations = 1000000;
        return alloc;
    }
    if (tc.max_nodes > 0) {
        alloc.iterations = tc.max_nodes;
        return alloc;
    }
    if (tc.max_depth > 0) {
        alloc.iterations = tc.max_depth * 100;
        return alloc;
    }
    if (tc.movetime_ms > 0) {
        alloc.iterations = std::max(1, tc.movetime_ms * nps_estimate / 1000);
        alloc.soft_time_ms = tc.movetime_ms;
        return alloc;
    }

    int our_time = (side_to_move == WHITE) ? tc.wtime_ms : tc.btime_ms;
    int our_inc = (side_to_move == WHITE) ? tc.winc_ms : tc.binc_ms;
    int moves_left = (tc.movestogo > 0) ? tc.movestogo : 30;

    int time_per_move = our_time / moves_left + our_inc;
    time_per_move = time_per_move * 80 / 100;
    time_per_move = std::max(50, std::min(time_per_move, our_time / 2));

    alloc.soft_time_ms = time_per_move;
    alloc.iterations = std::max(1, time_per_move * nps_estimate / 1000);
    return alloc;
}

} // namespace uci
