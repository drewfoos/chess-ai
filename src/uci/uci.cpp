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
    std::string token;
    args >> token; // "name"
    if (token != "name") return;

    std::string name;
    args >> name;

    args >> token; // "value"
    if (token != "value") return;

    if (name == "Iterations") {
        int val;
        args >> val;
        base_params_.num_iterations = val;
    }
}

void UCIHandler::start_search(const TimeControl& tc) {
    stop_search();

    auto alloc = allocate_time(tc, current_pos_.side_to_move(), nps_estimate_);

    stop_flag_.store(false);
    searching_.store(true);

    // Capture history by value so the search thread has its own copy
    neural::PositionHistory history_copy = history_;

    search_thread_ = std::thread([this, alloc, history_copy]() {
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

            if (since_last >= 500 || iters_since >= 100) {
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                int nps = (elapsed_ms > 0) ? static_cast<int>(info.iterations * 1000 / elapsed_ms) : 0;

                // Score in centipawns: root_value is [-1, +1], scale to cp
                int score_cp = static_cast<int>(info.root_value * 128.0f);
                score_cp = std::max(-12800, std::min(12800, score_cp));

                std::string pv = info.best_move.is_none() ? "0000" : info.best_move.to_uci();

                send("info nodes " + std::to_string(info.total_nodes) +
                     " nps " + std::to_string(nps) +
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
