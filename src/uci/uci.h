#pragma once
#include "mcts/search.h"
#include "neural/position_history.h"
#include "uci/time_manager.h"
#include "uci/responder.h"
#include "uci/stdout_responder.h"
#include "core/move_parser.h"
#include <string>
#include <atomic>
#include <thread>
#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <cstdint>

namespace uci {

class UCIHandler {
public:
    // Default ctor: writes protocol output to `out` via an internally-owned
    // StdoutResponder. Used by main.cpp and tests — they pass std::cout or a
    // std::ostringstream respectively.
    UCIHandler(mcts::Evaluator& evaluator,
               const mcts::SearchParams& base_params = {},
               std::istream& in = std::cin,
               std::ostream& out = std::cout);

    // Custom-responder ctor: lets callers inject a Responder (e.g. a ponder
    // forwarder in Phase E, or a capturing mock in tests). `responder` must
    // outlive the UCIHandler.
    UCIHandler(mcts::Evaluator& evaluator,
               Responder& responder,
               const mcts::SearchParams& base_params = {},
               std::istream& in = std::cin);

    ~UCIHandler();

    void loop();

    // Block until the currently-running search (if any) completes naturally and
    // emits bestmove. Does NOT set stop_flag — useful in tests that feed a
    // bounded `go nodes N` without a trailing `quit`/`stop`, so the search has
    // a chance to run to completion instead of being aborted by cleanup.
    void wait_for_search();

private:
    void handle_uci();
    void handle_isready();
    void handle_ucinewgame();
    void handle_position(std::istringstream& args);
    void handle_go(std::istringstream& args);
    void handle_stop();
    void handle_ponderhit();
    void handle_setoption(std::istringstream& args);

    void start_search(const TimeControl& tc);
    void stop_search();

    mcts::Evaluator& evaluator_;
    std::unique_ptr<StdoutResponder> owned_responder_;  // non-null only when using the stream ctor
    Responder& responder_;
    mcts::SearchParams base_params_;
    std::istream& in_;

    Position current_pos_;
    neural::PositionHistory history_;

    std::thread search_thread_;
    std::atomic<bool> stop_flag_{false};
    int nps_estimate_ = 500;
    int move_overhead_ms_ = 50;  // safety buffer subtracted from allocated wall-clock time
    bool safety_filter_ = true;  // reject bestmoves that allow opponent mate-in-1
    bool show_wdl_ = false;      // emit per-mille W/D/L in info lines when enabled
    int multi_pv_ = 1;           // number of top moves to report; >1 emits alternatives

    // Pondering state. `pondering_` is read by handle_ponderhit/handle_stop on
    // the main thread; written by start_search (main) and cleared by ponderhit.
    // `hard_deadline_ms_` is read by the search-thread info callback every poll
    // and written by main thread (start_search initial value, ponderhit update).
    // Atomic so the search thread can pick up a ponderhit-installed deadline
    // without taking a lock. Stored as ms-since-search_start_; 0 = no deadline.
    std::atomic<bool> pondering_{false};
    std::atomic<int64_t> hard_deadline_ms_{0};
    TimeControl pending_tc_{};                                // saved on `go ponder` for ponderhit
    std::chrono::steady_clock::time_point search_start_{};    // set by start_search before launching thread
};

}  // namespace uci
