#pragma once
#include "core/types.h"
#include <algorithm>
#include <limits>

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
    // `go ponder`: search runs speculatively assuming opponent plays our
    // expected reply. No deadline is enforced and bestmove must NOT emit
    // until `stop` (abandon ponder) or `ponderhit` (opponent did play it,
    // convert to a normal timed search using the original tc).
    bool ponder = false;
};

struct TimeAllocation {
    int iterations = 0;
    int soft_time_ms = 0;
};

inline TimeAllocation allocate_time(const TimeControl& tc, Color side_to_move,
                                     int nps_estimate = 500) {
    TimeAllocation alloc{};

    if (tc.infinite) {
        // `go infinite` per spec must run until stop. A 1M cap would let a
        // fast backend (TRT) finish on its own and emit unsolicited bestmove.
        // INT_MAX is effectively unreachable even at >1 Bnps for years.
        alloc.iterations = std::numeric_limits<int>::max();
        return alloc;
    }
    if (tc.max_nodes > 0) {
        // In MCTS, one "iteration" expands one leaf, which becomes one node
        // in the tree (plus siblings via edges). Mapping `go nodes N` to N
        // iterations is an approximation — actual node count will be close
        // but can be slightly higher as siblings are materialized on demand.
        // Matches the convention used by Lc0 and other MCTS engines.
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

    // Degenerate: GUI gave us no clock and no inc. 100ms gets at least one
    // NN forward pass on a slow net — better than emitting `bestmove 0000`.
    if (our_time <= 0 && our_inc <= 0) {
        alloc.soft_time_ms = 100;
        alloc.iterations = std::max(1, 100 * nps_estimate / 1000);
        return alloc;
    }
    // Increment-only (e.g. `go winc 2000` with no wtime, occasionally seen
    // from buggy GUIs or test harnesses): use 80% of the increment.
    if (our_time <= 0) {
        alloc.soft_time_ms = std::max(10, our_inc * 8 / 10);
        alloc.iterations = std::max(1, alloc.soft_time_ms * nps_estimate / 1000);
        return alloc;
    }

    // Emergency reserve: keep a slice of the clock untouched so a single
    // slow batch or GUI latency spike can't flag us. Larger of 100ms (covers
    // typical NN inference + lichess RTT) or 5% of the clock (scales with
    // total time so blitz games don't reserve so much that play stalls).
    int emergency_reserve = std::max(100, our_time / 20);
    int usable_time = std::max(0, our_time - emergency_reserve);

    int budget;
    if (tc.movestogo > 0) {
        // Tournament time control (e.g. 40/90+30): split usable_time across
        // remaining moves in the period, plus most of the per-move increment.
        int divisor = std::max(1, tc.movestogo);
        budget = usable_time / divisor + (our_inc * 8 / 10);
    } else if (our_inc > 0) {
        // Sudden death WITH increment: increment refills the clock each move,
        // so we can spend a fair share now without permanently shrinking it.
        // Assume ~30 moves remaining and consume ~80% of the increment.
        budget = usable_time / 30 + (our_inc * 8 / 10);
    } else {
        // Sudden death NO increment: each move's spend permanently shrinks
        // the clock, so be more conservative — assume 40 remaining moves.
        budget = usable_time / 40;
    }

    // Hard per-move cap: never blow more than 25% of the remaining clock on
    // one move, regardless of how favorable the inc-aware math looks. The
    // engine has no move-stability signal yet (PV stability would justify
    // exceeding this), so this prevents pathological allocations like
    // wtime=1000 + inc=100000 from burning the whole base clock at once.
    int hard_cap = our_time / 4;
    budget = std::min(budget, hard_cap);

    // Floor at 10ms so even bullet scrambles get one usable batch.
    budget = std::max(10, budget);

    alloc.soft_time_ms = budget;
    alloc.iterations = std::max(1, budget * nps_estimate / 1000);
    return alloc;
}

} // namespace uci
