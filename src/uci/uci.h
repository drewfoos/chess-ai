#pragma once
#include "mcts/search.h"
#include "neural/position_history.h"
#include "uci/time_manager.h"
#include "core/move_parser.h"
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>
#include <sstream>

namespace uci {

class UCIHandler {
public:
    UCIHandler(mcts::Evaluator& evaluator,
               const mcts::SearchParams& base_params = {},
               std::istream& in = std::cin,
               std::ostream& out = std::cout);
    ~UCIHandler();

    void loop();

private:
    void handle_uci();
    void handle_isready();
    void handle_ucinewgame();
    void handle_position(std::istringstream& args);
    void handle_go(std::istringstream& args);
    void handle_stop();
    void handle_setoption(std::istringstream& args);

    void start_search(const TimeControl& tc);
    void stop_search();
    void send(const std::string& msg);

    mcts::Evaluator& evaluator_;
    mcts::SearchParams base_params_;
    std::istream& in_;
    std::ostream& out_;
    std::mutex output_mutex_;

    Position current_pos_;
    neural::PositionHistory history_;

    std::thread search_thread_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> searching_{false};
    int nps_estimate_ = 500;
};

} // namespace uci
