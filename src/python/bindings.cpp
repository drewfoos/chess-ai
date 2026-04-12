#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/types.h"
#include "core/position.h"
#include "core/movegen.h"
#include "core/attacks.h"
#include "mcts/search.h"
#include "mcts/game_manager.h"
#include "neural/neural_evaluator.h"
#include "neural/position_history.h"
#include "neural/policy_map.h"

#include <string>
#include <map>
#include <stdexcept>
#include <memory>
#include <array>

namespace py = pybind11;

namespace {

// Parse a UCI move string (e.g., "e2e4", "e7e8q") against the current position.
// We need to match against legal moves to get the correct MoveFlag.
Move parse_uci_move(const Position& pos, const std::string& uci) {
    if (uci.size() < 4 || uci.size() > 5) {
        throw std::runtime_error("Invalid UCI move format: " + uci);
    }

    int from = (uci[0] - 'a') + (uci[1] - '1') * 8;
    int to   = (uci[2] - 'a') + (uci[3] - '1') * 8;

    PieceType promo = NO_PIECE_TYPE;
    if (uci.size() == 5) {
        switch (uci[4]) {
            case 'q': promo = QUEEN;  break;
            case 'r': promo = ROOK;   break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
            default:
                throw std::runtime_error("Invalid promotion piece in UCI move: " + uci);
        }
    }

    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    for (int i = 0; i < n; i++) {
        if (moves[i].from() == Square(from) && moves[i].to() == Square(to)) {
            if (promo == NO_PIECE_TYPE || moves[i].promo_piece() == promo) {
                return moves[i];
            }
        }
    }
    throw std::runtime_error("Illegal move: " + uci);
}

// Convert std::array<float, 1858> to numpy array
py::array_t<float> make_numpy_array(const std::array<float, neural::POLICY_SIZE>& arr) {
    auto result = py::array_t<float>(neural::POLICY_SIZE);
    auto buf = result.mutable_unchecked<1>();
    for (int i = 0; i < neural::POLICY_SIZE; i++) {
        buf(i) = arr[i];
    }
    return result;
}

// Python-friendly search result
struct PySearchResult {
    std::string best_move;
    std::map<std::string, int> visit_counts;
    py::array_t<float> policy_target;
    py::array_t<float> raw_policy;
    float root_value;
    float raw_value;
    int total_nodes;
};

// Apply a config dict to SearchParams
void apply_config(mcts::SearchParams& params, const py::dict& config) {
    for (auto& item : config) {
        std::string key = py::str(item.first);

        if (key == "num_iterations")       params.num_iterations = item.second.cast<int>();
        else if (key == "c_puct_init")     params.c_puct_init = item.second.cast<float>();
        else if (key == "c_puct_base")     params.c_puct_base = item.second.cast<float>();
        else if (key == "c_puct_factor")   params.c_puct_factor = item.second.cast<float>();
        else if (key == "fpu_reduction_root") params.fpu_reduction_root = item.second.cast<float>();
        else if (key == "fpu_reduction")   params.fpu_reduction = item.second.cast<float>();
        else if (key == "dirichlet_alpha") params.dirichlet_alpha = item.second.cast<float>();
        else if (key == "dirichlet_epsilon") params.dirichlet_epsilon = item.second.cast<float>();
        else if (key == "add_noise")       params.add_noise = item.second.cast<bool>();
        else if (key == "policy_softmax_temp") params.policy_softmax_temp = item.second.cast<float>();
        else if (key == "batch_size")      params.batch_size = item.second.cast<int>();
        else if (key == "smart_pruning")   params.smart_pruning = item.second.cast<bool>();
        else if (key == "smart_pruning_factor") params.smart_pruning_factor = item.second.cast<float>();
        else if (key == "two_fold_draw")   params.two_fold_draw = item.second.cast<bool>();
        else if (key == "shaped_dirichlet") params.shaped_dirichlet = item.second.cast<bool>();
        else if (key == "uncertainty_weight") params.uncertainty_weight = item.second.cast<float>();
        else if (key == "variance_scaling") params.variance_scaling = item.second.cast<bool>();
        else if (key == "contempt")        params.contempt = item.second.cast<float>();
        else if (key == "sibling_blending") params.sibling_blending = item.second.cast<bool>();
        else if (key == "nn_cache_size")   params.nn_cache_size = item.second.cast<int>();
        else {
            throw std::runtime_error("Unknown config key: " + key);
        }
    }
}

// The main SearchEngine wrapper exposed to Python
class SearchEngine {
public:
    SearchEngine(const std::string& model_path, const std::string& device,
                 py::dict config = py::dict()) {
        params_ = mcts::SearchParams{};
        apply_config(params_, config);

        evaluator_ = std::make_unique<neural::NeuralEvaluator>(
            model_path, device, params_.policy_softmax_temp);
        search_ = std::make_unique<mcts::Search>(*evaluator_, params_);
    }

    PySearchResult search(const std::string& fen,
                          const std::vector<std::string>& moves = {}) {
        // Parse the starting FEN
        Position pos;
        pos.set_fen(fen);

        // Build position history by replaying moves
        neural::PositionHistory history;
        history.reset(pos);

        for (const auto& uci_str : moves) {
            Move m = parse_uci_move(pos, uci_str);
            UndoInfo undo;
            pos.make_move(m, undo);
            history.push(pos);
        }

        // Run C++ MCTS search
        mcts::SearchResult result = search_->run(history);

        // Convert to Python-friendly result
        PySearchResult py_result;
        py_result.best_move = result.best_move.to_uci();
        py_result.root_value = result.root_value;
        py_result.raw_value = result.raw_value;
        py_result.total_nodes = result.total_nodes;
        py_result.policy_target = make_numpy_array(result.policy_target);
        py_result.raw_policy = make_numpy_array(result.raw_policy);

        // Build visit_counts dict: UCI string -> count
        for (size_t i = 0; i < result.moves.size(); i++) {
            py_result.visit_counts[result.moves[i].to_uci()] = result.visit_counts[i];
        }

        return py_result;
    }

    void set_config(const py::dict& config) {
        apply_config(params_, config);
        // Recreate search with updated params
        search_ = std::make_unique<mcts::Search>(*evaluator_, params_);
    }

private:
    mcts::SearchParams params_;
    std::unique_ptr<neural::NeuralEvaluator> evaluator_;
    std::unique_ptr<mcts::Search> search_;
};

// The GameManager wrapper exposed to Python — runs N games with cross-game batching
class PyGameManager {
public:
    PyGameManager(const std::string& model_path, const std::string& device,
                  py::dict config = py::dict()) {
        params_ = mcts::SearchParams{};
        apply_config(params_, config);

        evaluator_ = std::make_unique<neural::NeuralEvaluator>(
            model_path, device, params_.policy_softmax_temp);
        manager_ = std::make_unique<mcts::GameManager>(*evaluator_, params_);
    }

    void init_games(int num_games, int num_sims) {
        manager_->init_games(num_games, num_sims);
    }

    void init_games_from_fen(const std::vector<std::string>& fens,
                             const std::vector<std::vector<std::string>>& move_histories,
                             int num_sims) {
        manager_->init_games_from_fen(fens, move_histories, num_sims);
    }

    int step() {
        return manager_->step();
    }

    bool all_complete() const {
        return manager_->all_complete();
    }

    bool is_complete(int idx) const {
        return manager_->is_complete(idx);
    }

    PySearchResult get_result(int idx) {
        mcts::SearchResult result = manager_->get_result(idx);

        PySearchResult py_result;
        py_result.best_move = result.best_move.to_uci();
        py_result.root_value = result.root_value;
        py_result.raw_value = result.raw_value;
        py_result.total_nodes = result.total_nodes;
        py_result.policy_target = make_numpy_array(result.policy_target);
        py_result.raw_policy = make_numpy_array(result.raw_policy);

        for (size_t i = 0; i < result.moves.size(); i++) {
            py_result.visit_counts[result.moves[i].to_uci()] = result.visit_counts[i];
        }

        return py_result;
    }

    int num_games() const {
        return manager_->num_games();
    }

private:
    mcts::SearchParams params_;
    std::unique_ptr<neural::NeuralEvaluator> evaluator_;
    std::unique_ptr<mcts::GameManager> manager_;
};

} // anonymous namespace

PYBIND11_MODULE(chess_mcts, m) {
    m.doc() = "C++ MCTS search engine for chess AI";

    // Initialize attack tables once at module load
    attacks::init();

    // Expose SearchResult
    py::class_<PySearchResult>(m, "SearchResult")
        .def_readonly("best_move", &PySearchResult::best_move)
        .def_readonly("visit_counts", &PySearchResult::visit_counts)
        .def_readonly("policy_target", &PySearchResult::policy_target)
        .def_readonly("raw_policy", &PySearchResult::raw_policy)
        .def_readonly("root_value", &PySearchResult::root_value)
        .def_readonly("raw_value", &PySearchResult::raw_value)
        .def_readonly("total_nodes", &PySearchResult::total_nodes);

    // Expose SearchEngine
    py::class_<SearchEngine>(m, "SearchEngine")
        .def(py::init<const std::string&, const std::string&, py::dict>(),
             py::arg("model_path"),
             py::arg("device") = "cpu",
             py::arg("config") = py::dict())
        .def("search", &SearchEngine::search,
             py::arg("fen"),
             py::arg("moves") = std::vector<std::string>{},
             "Run MCTS search from a FEN position with optional move history")
        .def("set_config", &SearchEngine::set_config,
             py::arg("config"),
             "Update search configuration parameters");

    // Expose GameManager for cross-game batched search
    py::class_<PyGameManager>(m, "GameManager")
        .def(py::init<const std::string&, const std::string&, py::dict>(),
             py::arg("model_path"),
             py::arg("device") = "cpu",
             py::arg("config") = py::dict())
        .def("init_games", &PyGameManager::init_games,
             py::arg("num_games"),
             py::arg("num_sims"),
             "Initialize N games from the starting position")
        .def("init_games_from_fen", &PyGameManager::init_games_from_fen,
             py::arg("fens"),
             py::arg("move_histories"),
             py::arg("num_sims"),
             "Initialize games from FEN strings and UCI move histories")
        .def("step", &PyGameManager::step,
             "Run one step of cross-game batched search. Returns number of newly completed games.")
        .def("all_complete", &PyGameManager::all_complete,
             "Check if all games have completed their search")
        .def("is_complete", &PyGameManager::is_complete,
             py::arg("idx"),
             "Check if a specific game has completed its search")
        .def("get_result", &PyGameManager::get_result,
             py::arg("idx"),
             "Get the search result for a completed game")
        .def("num_games", &PyGameManager::num_games,
             "Get the number of games");
}
