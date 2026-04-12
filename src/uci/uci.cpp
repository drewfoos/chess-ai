#include "uci/uci.h"
#include <chrono>
#include <algorithm>

namespace uci {

UCIHandler::UCIHandler(mcts::Evaluator& evaluator,
                       const mcts::SearchParams& base_params,
                       std::istream& in,
                       std::ostream& out)
    : evaluator_(evaluator)
    , base_params_(base_params)
    , in_(in)
    , out_(out)
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
        } else if (cmd == "setoption") {
            handle_setoption(iss);
        } else if (cmd == "quit") {
            stop_search();
            return;
        }
        // Unknown commands are silently ignored per UCI spec
    }
}

void UCIHandler::handle_uci() {
    send("id name ChessAI");
    send("id author drew");
    send("option name Iterations type spin default 800 min 1 max 100000");
    // Accept common GUI-set options even if we don't use them — Arena/Lichess
    // complain when they try to set an option the engine didn't advertise.
    send("option name Hash type spin default 256 min 1 max 32768");
    send("option name Threads type spin default 1 min 1 max 128");
    send("option name UCI_Chess960 type check default false");
    send("option name Ponder type check default false");
    send("option name Move Overhead type spin default 50 min 0 max 5000");
    send("uciok");
}

void UCIHandler::handle_isready() {
    send("readyok");
}

void UCIHandler::handle_ucinewgame() {
    stop_search();
    current_pos_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    history_.reset(current_pos_);
}

void UCIHandler::handle_position(std::istringstream& args) {
    std::string token;
    args >> token;

    if (token == "startpos") {
        current_pos_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        history_.reset(current_pos_);
        args >> token; // consume "moves" if present
    } else if (token == "fen") {
        std::string fen;
        for (int i = 0; i < 6; i++) {
            std::string part;
            args >> part;
            if (i > 0) fen += " ";
            fen += part;
        }
        current_pos_.set_fen(fen);
        history_.reset(current_pos_);
        args >> token; // consume "moves" if present
    }

    // Parse moves
    std::string move_str;
    while (args >> move_str) {
        Move m = parse_uci_move(current_pos_, move_str);
        UndoInfo undo;
        current_pos_.make_move(m, undo);
        history_.push(current_pos_);
    }
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
    }
    start_search(tc);
}

void UCIHandler::handle_stop() {
    stop_flag_.store(true);
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
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

    if (name == "Iterations") {
        try { base_params_.num_iterations = std::stoi(value); } catch (...) {}
    } else if (name == "Move Overhead") {
        try { move_overhead_ms_ = std::max(0, std::stoi(value)); } catch (...) {}
    }
    // Hash, Threads, UCI_Chess960, Ponder — accepted and silently ignored.
}

void UCIHandler::start_search(const TimeControl& tc) {
    stop_search();

    auto alloc = allocate_time(tc, current_pos_.side_to_move(), nps_estimate_);

    // When a wall-clock budget is given (movetime or clock time), let time
    // drive the stop — not the iteration count derived from a stale nps_estimate.
    // The first search calibrates nps_estimate_ to the engine's real speed;
    // without this override, move 1 finishes in a fraction of the requested time.
    int hard_deadline_ms = alloc.soft_time_ms;
    if (hard_deadline_ms > 0) {
        // Reserve a safety buffer so the GUI gets our bestmove before flagging.
        hard_deadline_ms = std::max(10, hard_deadline_ms - move_overhead_ms_);
        alloc.iterations = 1'000'000;
    }

    stop_flag_.store(false);
    searching_.store(true);

    // Capture history by value so the search thread has its own copy
    neural::PositionHistory history_copy = history_;

    search_thread_ = std::thread([this, alloc, hard_deadline_ms, history_copy]() {
        mcts::SearchParams params = base_params_;
        params.num_iterations = alloc.iterations;
        params.add_noise = false;

        mcts::Search search(evaluator_, params);
        search.set_stop_flag(&stop_flag_);

        auto start_time = std::chrono::steady_clock::now();
        int last_info_iter = 0;
        auto last_info_time = start_time;

        search.set_info_callback([&](const mcts::SearchInfo& info) {
            auto now = std::chrono::steady_clock::now();
            auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_info_time).count();
            int iters_since = info.iterations - last_info_iter;

            auto elapsed_ms_now = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            if (hard_deadline_ms > 0 && elapsed_ms_now >= hard_deadline_ms) {
                stop_flag_.store(true, std::memory_order_relaxed);
            }

            if (since_last >= 500 || iters_since >= 100) {
                auto elapsed_ms = elapsed_ms_now;
                int nps = (elapsed_ms > 0) ? static_cast<int>(info.iterations * 1000 / elapsed_ms) : 0;

                // Score in centipawns: root_value is [-1, +1], scale to cp
                int score_cp = static_cast<int>(info.root_value * 128.0f);
                score_cp = std::max(-12800, std::min(12800, score_cp));

                std::string pv = info.best_move.is_none() ? "0000" : info.best_move.to_uci();

                // MCTS has no literal search depth — approximate via log2(iterations)
                // so GUIs have a non-empty Depth column that grows with search effort.
                int pseudo_depth = 1;
                int it = info.iterations;
                while (it > 1) { pseudo_depth++; it >>= 1; }

                send("info depth " + std::to_string(pseudo_depth) +
                     " nodes " + std::to_string(info.total_nodes) +
                     " nps " + std::to_string(nps) +
                     " time " + std::to_string(elapsed_ms) +
                     " score cp " + std::to_string(score_cp) +
                     " pv " + pv);

                last_info_iter = info.iterations;
                last_info_time = now;
            }
        });

        mcts::SearchResult result = search.run(history_copy);

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (elapsed_ms > 0) {
            nps_estimate_ = static_cast<int>(result.total_nodes * 1000 / elapsed_ms);
        }

        std::string best = result.best_move.is_none() ? "0000" : result.best_move.to_uci();
        send("bestmove " + best);

        searching_.store(false);
    });
}

void UCIHandler::stop_search() {
    stop_flag_.store(true);
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
}

void UCIHandler::send(const std::string& msg) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    out_ << msg << "\n";
    out_.flush();
}

} // namespace uci
