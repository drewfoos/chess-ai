#include "uci/uci.h"
#include "uci/safety_filter.h"
#include "syzygy/syzygy.h"
#include <chrono>
#include <algorithm>
#include <numeric>

namespace uci {

UCIHandler::UCIHandler(mcts::Evaluator& evaluator,
                       const mcts::SearchParams& base_params,
                       std::istream& in,
                       std::ostream& out)
    : evaluator_(evaluator)
    , owned_responder_(std::make_unique<StdoutResponder>(out))
    , responder_(*owned_responder_)
    , base_params_(base_params)
    , in_(in)
{
    current_pos_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    history_.reset(current_pos_);
}

UCIHandler::UCIHandler(mcts::Evaluator& evaluator,
                       Responder& responder,
                       const mcts::SearchParams& base_params,
                       std::istream& in)
    : evaluator_(evaluator)
    , owned_responder_(nullptr)
    , responder_(responder)
    , base_params_(base_params)
    , in_(in)
{
    current_pos_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    history_.reset(current_pos_);
}

UCIHandler::~UCIHandler() {
    stop_search();
}

void UCIHandler::loop() {
    std::string line;
    while (std::getline(in_, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd.empty()) continue;

        // Any handler may throw on malformed input (notably parse_uci_move on
        // illegal/garbage moves). Catch here so the engine keeps running and
        // the GUI gets a diagnostic instead of silent process exit.
        try {
            if (cmd == "uci") {
                handle_uci();
            } else if (cmd == "isready") {
                handle_isready();
            } else if (cmd == "ucinewgame") {
                handle_ucinewgame();
            } else if (cmd == "position") {
                handle_position(iss);
            } else if (cmd == "go") {
                handle_go(iss);
            } else if (cmd == "stop") {
                handle_stop();
            } else if (cmd == "ponderhit") {
                handle_ponderhit();
            } else if (cmd == "setoption") {
                handle_setoption(iss);
            } else if (cmd == "quit") {
                stop_search();
                return;
            }
            // Unknown commands are silently ignored per UCI spec
        } catch (const std::exception& e) {
            responder_.OnString(std::string("error parsing '") + cmd +
                                "': " + e.what());
        }
    }
}

void UCIHandler::handle_uci() {
    responder_.OnRaw("id name ChessAI");
    responder_.OnRaw("id author drew");
    responder_.OnRaw("option name Iterations type spin default 800 min 1 max 100000");
    // Accept common GUI-set options even if we don't use them — Arena/Lichess
    // complain when they try to set an option the engine didn't advertise.
    responder_.OnRaw("option name Hash type spin default 256 min 1 max 32768");
    responder_.OnRaw("option name Threads type spin default 1 min 1 max 128");
    responder_.OnRaw("option name UCI_Chess960 type check default false");
    responder_.OnRaw("option name Ponder type check default false");
    responder_.OnRaw("option name Move Overhead type spin default 50 min 0 max 5000");
    responder_.OnRaw("option name SyzygyPath type string default <empty>");
    responder_.OnRaw("option name SafetyFilter type check default true");
    responder_.OnRaw("option name UCI_ShowWDL type check default false");
    responder_.OnRaw("option name MultiPV type spin default 1 min 1 max 500");
    responder_.OnRaw("uciok");
}

void UCIHandler::handle_isready() {
    responder_.OnRaw("readyok");
}

void UCIHandler::handle_ucinewgame() {
    stop_search();
    current_pos_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    history_.reset(current_pos_);
}

void UCIHandler::handle_position(std::istringstream& args) {
    // Atomic update: build the new position in locals, only commit if every
    // token parses. A GUI that sends an illegal/malformed move mid-sequence
    // (e.g. `position startpos moves e2e4 xxxx`) leaves the engine's state
    // exactly as it was — otherwise the partial apply would desync us from
    // the GUI by one ply. On failure we throw; loop() catches and emits an
    // info string diagnostic without killing the process.
    std::string token;
    args >> token;

    Position new_pos;
    neural::PositionHistory new_history;

    if (token == "startpos") {
        new_pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        new_history.reset(new_pos);
        args >> token; // consume "moves" if present
    } else if (token == "fen") {
        std::string fen;
        for (int i = 0; i < 6; i++) {
            std::string part;
            args >> part;
            if (i > 0) fen += " ";
            fen += part;
        }
        new_pos.set_fen(fen);
        new_history.reset(new_pos);
        args >> token; // consume "moves" if present
    } else {
        // Unknown sub-command — leave state alone, quietly ignore (spec says
        // position is always "startpos" or "fen").
        return;
    }

    // Parse moves into the temporary position; any parse_uci_move throw
    // aborts the whole update before current_pos_/history_ is touched.
    std::string move_str;
    while (args >> move_str) {
        Move m = parse_uci_move(new_pos, move_str);
        UndoInfo undo;
        new_pos.make_move(m, undo);
        new_history.push(new_pos);
    }

    // Commit.
    current_pos_ = new_pos;
    history_ = std::move(new_history);
}

void UCIHandler::handle_go(std::istringstream& args) {
    TimeControl tc;
    std::string token;
    while (args >> token) {
        if (token == "wtime") args >> tc.wtime_ms;
        else if (token == "btime") args >> tc.btime_ms;
        else if (token == "winc") args >> tc.winc_ms;
        else if (token == "binc") args >> tc.binc_ms;
        else if (token == "movestogo") args >> tc.movestogo;
        else if (token == "depth") args >> tc.max_depth;
        else if (token == "nodes") args >> tc.max_nodes;
        else if (token == "movetime") args >> tc.movetime_ms;
        else if (token == "infinite") tc.infinite = true;
        else if (token == "ponder") tc.ponder = true;
    }
    start_search(tc);
}

void UCIHandler::handle_stop() {
    // Stop while pondering means the GUI is abandoning ponder (opponent played
    // a different move). Clear the flag before joining so the search loop
    // exits cleanly and the post-search code emits bestmove just like a normal
    // stop. UCI requires a bestmove in response to stop, even mid-ponder.
    pondering_.store(false);
    stop_search();
}

void UCIHandler::handle_ponderhit() {
    // GUI is telling us the opponent played the move we were pondering on,
    // so the speculative search becomes the actual search. Convert the
    // saved `pending_tc_` into a real deadline relative to search_start_,
    // then let the search keep running until the deadline trips stop_flag.
    if (!pondering_.load(std::memory_order_acquire)) return;  // no-op if not pondering

    auto alloc = allocate_time(pending_tc_, current_pos_.side_to_move(), nps_estimate_);
    int64_t budget_ms = alloc.soft_time_ms;
    if (budget_ms > 0) {
        budget_ms = std::max<int64_t>(10, budget_ms - move_overhead_ms_);
    }

    if (budget_ms > 0) {
        // Deadline is "ms-since-search_start_". Search has already been
        // pondering for some time; that time is essentially "free" thinking,
        // so charge only the new budget against it (don't deduct elapsed).
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - search_start_).count();
        hard_deadline_ms_.store(elapsed_ms + budget_ms, std::memory_order_release);
    }
    // budget_ms <= 0 means the user gave us no clock (e.g. `go ponder` alone)
    // — treat as `go infinite` post-hit and rely on `stop` to terminate.

    pondering_.store(false, std::memory_order_release);
}

void UCIHandler::handle_setoption(std::istringstream& args) {
    // Grammar: setoption name <NAME...> value <VALUE...>
    // NAME and VALUE may each contain spaces; parse accordingly.
    std::string token;
    args >> token;
    if (token != "name") return;

    std::string name;
    std::string value;
    bool in_value = false;
    while (args >> token) {
        if (!in_value && token == "value") { in_value = true; continue; }
        if (in_value) {
            if (!value.empty()) value += " ";
            value += token;
        } else {
            if (!name.empty()) name += " ";
            name += token;
        }
    }

    // Options that mutate state the search thread reads (SyzygyPath reloads a
    // global tablebase handle) must not race with an in-flight search. UCI
    // spec says GUIs shouldn't send setoption mid-search, but we can't trust
    // that — stop first so the mutation is safe.
    stop_search();

    if (name == "Iterations") {
        try {
            int v = std::stoi(value);
            // Clamp to advertised min/max so neither a negative (search loops
            // zero times, bestmove = none) nor an absurd value (memory blowup)
            // can be injected.
            base_params_.num_iterations = std::max(1, std::min(100000, v));
        } catch (...) {}
    } else if (name == "Move Overhead") {
        try {
            int v = std::stoi(value);
            move_overhead_ms_ = std::max(0, std::min(5000, v));
        } catch (...) {}
    } else if (name == "SafetyFilter") {
        // UCI check-option values are the literal strings "true"/"false".
        safety_filter_ = (value == "true" || value == "True" || value == "TRUE");
    } else if (name == "UCI_ShowWDL") {
        show_wdl_ = (value == "true" || value == "True" || value == "TRUE");
    } else if (name == "MultiPV") {
        try {
            int v = std::stoi(value);
            multi_pv_ = std::max(1, std::min(500, v));
        } catch (...) {}
    } else if (name == "SyzygyPath") {
        // Empty / "<empty>" disables tablebases. Otherwise reload from path.
        if (value.empty() || value == "<empty>") {
            syzygy::TableBase::shutdown();
            responder_.OnString("Syzygy tablebases disabled");
        } else {
            int max_pieces = syzygy::TableBase::init(value);
            if (max_pieces > 0) {
                responder_.OnString("Syzygy tablebases loaded (max " +
                                    std::to_string(max_pieces) +
                                    " pieces) from " + value);
            } else {
                responder_.OnString("Syzygy: no tablebases found at " + value);
            }
        }
    }
    // Hash, Threads, UCI_Chess960, Ponder — accepted and silently ignored.
}

namespace {

// Approximate search depth as log2(iterations) — MCTS has no literal iterative-
// deepening depth, but GUIs expect a non-empty Depth column that grows with
// search effort. Matches Lc0's "pseudo-depth" approach.
int pseudo_depth(int iterations) {
    int d = 1;
    int it = iterations;
    while (it > 1) { d++; it >>= 1; }
    return d;
}

// Convert a root value in [-1, 1] to UCI centipawns, clamped to ±12800.
//
// Sign note: after backprop, `root->mean_value()` accumulates leaf values that
// have been flipped once per level — at the root it lands in the *opponent's*
// perspective, not side-to-move's. UCI `score cp` must be from side-to-move's
// perspective (positive = STM winning), so negate here. This is a display-only
// flip: SearchResult.root_value keeps its existing semantics for the Python
// self-play pipeline, which already treats it consistently.
int value_to_cp(float root_value) {
    int cp = static_cast<int>(-root_value * 128.0f);
    return std::max(-12800, std::min(12800, cp));
}

// Variant for values ALREADY in STM perspective (e.g. child_q_values, whose
// storage is one level further from root so the sign lands pre-flipped).
int stm_value_to_cp(float stm_value) {
    int cp = static_cast<int>(stm_value * 128.0f);
    return std::max(-12800, std::min(12800, cp));
}

// Synthesize a WDL per-mille triplet from an STM-perspective value in [-1, 1].
// Our network has a scalar value head, not a Lc0-style W/D/L head, so we
// reconstruct draw mass via a heuristic: draw rate peaks near value=0 and
// tapers to zero at ±1. The integer triplet is guaranteed to sum to exactly
// 1000 (rounding drift is absorbed into L).
WDL synthesize_wdl(float stm_value) {
    stm_value = std::max(-1.0f, std::min(1.0f, stm_value));
    float abs_v = std::abs(stm_value);
    // Peak draw rate 0.40 at v=0, tapering quadratically to 0 at |v|=1. Tuned
    // to feel reasonable for MCTS scores rather than match any ground-truth
    // distribution — without a WDL head we can't do better.
    float draw = 0.40f * (1.0f - abs_v * abs_v);
    // Expected score = W + D/2  =>  W = score - D/2
    float score = 0.5f * (stm_value + 1.0f);
    float w = std::max(0.0f, std::min(1.0f, score - 0.5f * draw));
    int W = static_cast<int>(w * 1000.0f + 0.5f);
    int D = static_cast<int>(draw * 1000.0f + 0.5f);
    if (W + D > 1000) D = 1000 - W;  // defensive
    int L = 1000 - W - D;
    return WDL{W, D, L};
}

}  // namespace

void UCIHandler::start_search(const TimeControl& tc) {
    stop_search();

    auto alloc = allocate_time(tc, current_pos_.side_to_move(), nps_estimate_);

    // When a wall-clock budget is given (movetime or clock time), let time
    // drive the stop — not the iteration count derived from a stale nps_estimate.
    // The first search calibrates nps_estimate_ to the engine's real speed;
    // without this override, move 1 finishes in a fraction of the requested time.
    int64_t initial_deadline_ms = alloc.soft_time_ms;
    if (initial_deadline_ms > 0) {
        // Reserve a safety buffer so the GUI gets our bestmove before flagging.
        initial_deadline_ms = std::max<int64_t>(10, initial_deadline_ms - move_overhead_ms_);
        alloc.iterations = 1'000'000;
    }

    if (tc.ponder) {
        // Ponder: speculative search with no deadline and effectively unlimited
        // iterations. The actual budget (if any) is installed by handle_ponderhit
        // when the GUI confirms the predicted move. Until then, only `stop` or
        // `ponderhit` can terminate the search. Save tc so we can derive the
        // post-hit deadline from the original wtime/btime/etc.
        pending_tc_ = tc;
        pondering_.store(true, std::memory_order_release);
        initial_deadline_ms = 0;
        alloc.iterations = std::numeric_limits<int>::max();
    } else {
        pondering_.store(false, std::memory_order_release);
    }

    hard_deadline_ms_.store(initial_deadline_ms, std::memory_order_release);
    search_start_ = std::chrono::steady_clock::now();
    stop_flag_.store(false);

    // Capture history by value so the search thread has its own copy
    neural::PositionHistory history_copy = history_;

    search_thread_ = std::thread([this, alloc, history_copy]() {
        mcts::SearchParams params = base_params_;
        params.num_iterations = alloc.iterations;
        params.add_noise = false;
        // Competitive play must avoid 3-folding won positions. Self-play keeps
        // Lc0-parity (3-fold only) via base_params_; override here so MCTS
        // prunes repeating lines as draws during UCI play.
        params.two_fold_draw = true;

        mcts::Search search(evaluator_, params);
        search.set_stop_flag(&stop_flag_);

        // Use search_start_ from the main thread (set in start_search before
        // launching us) so handle_ponderhit's deadline math is referenced to
        // the same anchor we check against here. A locally-captured "now" at
        // thread start would drift by a microsecond and complicate ponderhit.
        const auto start_time = search_start_;
        int last_info_iter = 0;
        auto last_info_time = start_time;

        search.set_info_callback([&](const mcts::SearchInfo& info) {
            auto now = std::chrono::steady_clock::now();
            auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_info_time).count();
            int iters_since = info.iterations - last_info_iter;

            auto elapsed_ms_now = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            // Re-read deadline each callback so a ponderhit-installed budget
            // takes effect immediately. 0 = no deadline (pondering, infinite,
            // node-budgeted, etc.).
            int64_t deadline = hard_deadline_ms_.load(std::memory_order_acquire);
            if (deadline > 0 && elapsed_ms_now >= deadline) {
                stop_flag_.store(true, std::memory_order_relaxed);
            }

            if (since_last >= 500 || iters_since >= 100) {
                ThinkingInfo ti;
                // Depth derived from total_nodes (matches final info line, so
                // GUIs don't see the column jump when the last line lands).
                ti.depth = pseudo_depth(info.total_nodes);
                ti.nodes = info.total_nodes;
                ti.time_ms = elapsed_ms_now;
                ti.nps = (elapsed_ms_now > 0) ? info.total_nodes * 1000 / elapsed_ms_now : 0;
                // Mid-search we don't yet have proven terminal status wired through
                // the callback, so report cp from current root value.
                ti.score_cp = value_to_cp(info.root_value);
                if (!info.best_move.is_none()) {
                    ti.pv.push_back(info.best_move.to_uci());
                }
                responder_.OnInfo(ti);

                last_info_iter = info.iterations;
                last_info_time = now;
            }
        });

        mcts::SearchResult result = search.run(history_copy);

        // UCI ponder: bestmove must NOT emerge until stop or ponderhit. If
        // search returned early (e.g. solver proved root terminal during
        // ponder, or finite iter budget), park here until the GUI signals.
        // Either handle_stop sets stop_flag, or handle_ponderhit clears the
        // pondering flag — both wake us. Cheap polling (10ms) is fine since
        // this branch is rare and only active mid-ponder.
        while (pondering_.load(std::memory_order_acquire) &&
               !stop_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // Only update nps_estimate_ on searches that ran to completion. A
        // stop-interrupted search captures ramp-up throughput and would poison
        // the next time-based iteration budget. Also require a minimum of
        // 100ms to filter out tiny warm-up runs where timer noise dominates.
        if (elapsed_ms >= 100 && !stop_flag_.load(std::memory_order_relaxed)) {
            nps_estimate_ = static_cast<int>(result.total_nodes * 1000 / elapsed_ms);
        }

        // Tactical safety filter: if the search's top choice hands the opponent
        // mate-in-1, walk down the visit-sorted candidates for the first that
        // doesn't. Mainly matters for pretrained-only networks, where policy
        // priors on "unnatural" killer replies can be low enough that MCTS
        // never visits the mate branch. Sub-millisecond per candidate checked.
        Move final_move = result.best_move;
        if (safety_filter_ && !result.moves.empty() && !result.best_move.is_none()) {
            Position filter_pos = history_copy.current();
            // Try result.best_move first (the search's declared pick — may be a
            // proof-tree move with few visits), then fall back to visit-greedy
            // order for any other candidates. Without this, a proven mate can
            // get overridden by an arbitrary pawn push that *also* doesn't
            // hang mate, just because the pawn push racked up more visits
            // before the solver closed the search.
            int best_idx = -1;
            std::vector<int> order;
            order.reserve(result.moves.size());
            for (size_t i = 0; i < result.moves.size(); i++) {
                if (result.moves[i] == result.best_move) {
                    best_idx = static_cast<int>(i);
                } else {
                    order.push_back(static_cast<int>(i));
                }
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return result.visit_counts[a] > result.visit_counts[b];
            });
            if (best_idx >= 0) order.insert(order.begin(), best_idx);
            for (int idx : order) {
                if (idx != best_idx && result.visit_counts[idx] == 0) break;
                Move cand = result.moves[idx];
                if (!hangs_mate(filter_pos, cand, 1)) {
                    if (!(cand == result.best_move)) {
                        responder_.OnString("safety filter: " +
                                            result.best_move.to_uci() +
                                            " hangs mate; playing " +
                                            cand.to_uci() + " instead");
                    }
                    final_move = cand;
                    break;
                }
            }
            // If every visited candidate hangs mate (we're already lost),
            // final_move stays as result.best_move — the GUI gets *a* move
            // and the game ends normally instead of forfeiting on time.
        }

        std::string best = final_move.is_none() ? "0000" : final_move.to_uci();

        // Final info line(s) before bestmove. The periodic callback is
        // rate-limited (every 500ms or 100 iters) so very short searches can
        // finish without emitting any info — GUIs like Arena expect at least
        // one. When MultiPV > 1, emit one line per top-N root move sorted by
        // visit count, with `multipv 1..N`. multipv 1 always corresponds to
        // final_move (possibly safety-filter-chosen); the remaining slots are
        // the next-most-visited root moves.
        {
            // Build the ordered candidate list. Slot 0 = final_move (may not
            // be the most-visited — safety filter can override). Remaining
            // slots = other moves by descending visit count, zero-visit moves
            // excluded so we don't advertise branches the engine never touched.
            int best_idx = -1;
            for (size_t i = 0; i < result.moves.size(); i++) {
                if (result.moves[i] == final_move) { best_idx = static_cast<int>(i); break; }
            }
            std::vector<int> pv_order;
            pv_order.reserve(result.moves.size());
            if (best_idx >= 0) pv_order.push_back(best_idx);
            {
                std::vector<int> rest;
                for (size_t i = 0; i < result.moves.size(); i++) {
                    if (static_cast<int>(i) == best_idx) continue;
                    if (result.visit_counts[i] == 0) continue;
                    rest.push_back(static_cast<int>(i));
                }
                std::sort(rest.begin(), rest.end(), [&](int a, int b) {
                    return result.visit_counts[a] > result.visit_counts[b];
                });
                for (int i : rest) pv_order.push_back(i);
            }

            const int emit_n = std::max(1, std::min(multi_pv_, static_cast<int>(pv_order.size())));

            for (int rank = 0; rank < emit_n; rank++) {
                int idx = pv_order[rank];
                ThinkingInfo ti;
                // Only stamp `multipv` when the user actually asked for
                // alternatives — default runs stay clean of the field.
                if (multi_pv_ > 1) ti.multipv = rank + 1;
                ti.depth = pseudo_depth(result.total_nodes);
                if (result.seldepth > 0) ti.seldepth = result.seldepth;
                ti.nodes = result.total_nodes;
                ti.time_ms = elapsed_ms;
                ti.nps = (elapsed_ms > 0) ? result.total_nodes * 1000 / elapsed_ms : 0;

                if (rank == 0) {
                    // Primary move gets the full proven-mate / contempt-adjusted
                    // score from the search result itself.
                    if (result.mate_distance_plies != 0) {
                        ti.mate_plies = result.mate_distance_plies;
                    } else {
                        ti.score_cp = value_to_cp(result.root_value);
                    }
                    if (show_wdl_) {
                        ti.wdl = synthesize_wdl(-result.root_value);
                    }
                    // PV: proper search PV walk when our chosen move matches
                    // the search's best; otherwise (safety filter overrode)
                    // fall back to just the chosen move.
                    if (!result.pv.empty() && final_move == result.best_move) {
                        for (const Move& m : result.pv) ti.pv.push_back(m.to_uci());
                    } else {
                        ti.pv.push_back(best);
                    }
                } else {
                    // Alternatives: score derived from the root child's Q
                    // (already in STM perspective, no flip needed). PV is the
                    // single-move visit — deeper walks would require re-
                    // entering the tree from that child.
                    float q = (idx < static_cast<int>(result.child_q_values.size()))
                                  ? result.child_q_values[idx] : 0.0f;
                    ti.score_cp = stm_value_to_cp(q);
                    if (show_wdl_) {
                        ti.wdl = synthesize_wdl(q);
                    }
                    ti.pv.push_back(result.moves[idx].to_uci());
                }

                // Only stamp hashfull / tbhits on the primary line to avoid
                // redundant / misleading totals on alternative lines.
                if (rank == 0) {
                    if (result.hashfull_permille > 0) ti.hashfull = result.hashfull_permille;
                    if (result.tb_hits > 0) ti.tbhits = result.tb_hits;
                }
                responder_.OnInfo(ti);
            }
        }

        BestMoveInfo bm;
        bm.bestmove = best;
        // Ponder move: second PV entry (opponent's expected reply). Only emit
        // when PV is the unfiltered one — a safety-filter override invalidates
        // the tail of the PV, so suppress ponder rather than mislead the GUI.
        if (result.pv.size() >= 2 && final_move == result.best_move) {
            bm.ponder = result.pv[1].to_uci();
        }
        responder_.OnBestMove(bm);
    });
}

void UCIHandler::stop_search() {
    stop_flag_.store(true);
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
}

void UCIHandler::wait_for_search() {
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
}

}  // namespace uci
