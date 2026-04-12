#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/types.h"
#include "core/position.h"
#include "core/movegen.h"
#include "core/attacks.h"
#include "core/move_parser.h"
#include "mcts/search.h"
#include "mcts/game_manager.h"
#include "neural/neural_evaluator.h"
#include "neural/position_history.h"
#include "neural/policy_map.h"
#include "neural/encoder.h"
#ifdef HAS_TENSORRT
#include "neural/trt_evaluator.h"
#endif

#include <string>
#include <map>
#include <stdexcept>
#include <memory>
#include <array>

namespace py = pybind11;

namespace {

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
        else if (key == "max_collapse_visits") params.max_collapse_visits = item.second.cast<int>();
        else if (key == "fpu_absolute_root") params.fpu_absolute_root = item.second.cast<bool>();
        else if (key == "fpu_absolute_root_value") params.fpu_absolute_root_value = item.second.cast<float>();
        else if (key == "mlh_weight")      params.mlh_weight = item.second.cast<float>();
        else if (key == "mlh_cap")         params.mlh_cap = item.second.cast<float>();
        else if (key == "mlh_q_threshold") params.mlh_q_threshold = item.second.cast<float>();
        else {
            throw std::runtime_error("Unknown config key: " + key);
        }
    }
}

// The main SearchEngine wrapper exposed to Python
class SearchEngine {
public:
    SearchEngine(const std::string& model_path, const std::string& device,
                 py::dict config = py::dict(), bool use_fp16 = false,
                 bool use_trt = false, const std::string& trt_engine_path = "") {
        params_ = mcts::SearchParams{};
        apply_config(params_, config);

        if (use_trt) {
#ifdef HAS_TENSORRT
            const std::string& eng = trt_engine_path.empty() ? model_path : trt_engine_path;
            trt_evaluator_ = std::make_unique<neural::TRTEvaluator>(
                eng, params_.policy_softmax_temp, std::max(1, params_.batch_size));
            search_ = std::make_unique<mcts::Search>(*trt_evaluator_, params_);
#else
            throw std::runtime_error("SearchEngine: use_trt=True but build lacks HAS_TENSORRT");
#endif
        } else {
            evaluator_ = std::make_unique<neural::NeuralEvaluator>(
                model_path, device, params_.policy_softmax_temp, use_fp16);
            search_ = std::make_unique<mcts::Search>(*evaluator_, params_);
        }
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
        if (evaluator_) {
            search_ = std::make_unique<mcts::Search>(*evaluator_, params_);
        }
#ifdef HAS_TENSORRT
        else if (trt_evaluator_) {
            search_ = std::make_unique<mcts::Search>(*trt_evaluator_, params_);
        }
#endif
    }

private:
    mcts::SearchParams params_;
    std::unique_ptr<neural::NeuralEvaluator> evaluator_;
#ifdef HAS_TENSORRT
    std::unique_ptr<neural::TRTEvaluator> trt_evaluator_;
#endif
    std::unique_ptr<mcts::Search> search_;
};

// The GameManager wrapper exposed to Python — runs N games with cross-game batching
class PyGameManager {
public:
    PyGameManager(const std::string& model_path, const std::string& device,
                  py::dict config = py::dict(), bool use_fp16 = false,
                  bool use_trt = false, const std::string& trt_engine_path = "") {
        params_ = mcts::SearchParams{};
        apply_config(params_, config);

        if (use_trt) {
#ifdef HAS_TENSORRT
            const std::string& eng = trt_engine_path.empty() ? model_path : trt_engine_path;
            trt_evaluator_ = std::make_unique<neural::TRTEvaluator>(
                eng, params_.policy_softmax_temp, std::max(1, params_.batch_size));
            manager_ = std::make_unique<mcts::GameManager>(*trt_evaluator_, params_);
#else
            throw std::runtime_error("GameManager: use_trt=True but build lacks HAS_TENSORRT");
#endif
        } else {
            evaluator_ = std::make_unique<neural::NeuralEvaluator>(
                model_path, device, params_.policy_softmax_temp, use_fp16);
            manager_ = std::make_unique<mcts::GameManager>(*evaluator_, params_);
        }
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
#ifdef HAS_TENSORRT
    std::unique_ptr<neural::TRTEvaluator> trt_evaluator_;
#endif
    std::unique_ptr<mcts::GameManager> manager_;
};

} // anonymous namespace

PYBIND11_MODULE(_core, m) {
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
        .def(py::init<const std::string&, const std::string&, py::dict, bool, bool, const std::string&>(),
             py::arg("model_path"),
             py::arg("device") = "cpu",
             py::arg("config") = py::dict(),
             py::arg("use_fp16") = false,
             py::arg("use_trt") = false,
             py::arg("trt_engine_path") = "")
        .def("search", &SearchEngine::search,
             py::arg("fen"),
             py::arg("moves") = std::vector<std::string>{},
             "Run MCTS search from a FEN position with optional move history")
        .def("set_config", &SearchEngine::set_config,
             py::arg("config"),
             "Update search configuration parameters");

    // Module-level encode_packed: replays uci_moves from start_fen and encodes
    // the resulting position (with 8-step history) into Lc0-style bitpacked
    // planes + a few bytes of metadata. Returns a tuple:
    //   (bitboards: np.ndarray(uint64)[104], stm: bool,
    //    castling: int, rule50: int, fullmove: int).
    // Castling bits: 0=STM-K, 1=STM-Q, 2=OPP-K, 3=OPP-Q.
    m.def("encode_packed",
        [](const std::string& start_fen,
           const std::vector<std::string>& uci_moves) -> py::tuple {
            Position pos;
            pos.set_fen(start_fen);
            neural::PositionHistory hist;
            hist.reset(pos);
            for (const auto& uci : uci_moves) {
                Move mv = parse_uci_move(pos, uci);
                UndoInfo undo;
                pos.make_move(mv, undo);
                hist.push(pos);
            }
            neural::PackedPosition pp;
            neural::encode_position_packed(hist, pp);

            auto bb = py::array_t<uint64_t>(neural::PACKED_PLANES);
            auto buf = bb.mutable_unchecked<1>();
            for (int i = 0; i < neural::PACKED_PLANES; i++) {
                buf(i) = pp.planes[i];
            }
            return py::make_tuple(
                bb,
                py::bool_(pp.stm != 0),
                py::int_(pp.castling),
                py::int_(pp.rule50),
                py::int_(pp.fullmove)
            );
        },
        py::arg("start_fen"),
        py::arg("uci_moves") = std::vector<std::string>{},
        "Encode position + 8-step history into bitpacked planes. "
        "Returns (bitboards[104] uint64, stm bool, castling int, rule50 int, fullmove int).");

    // Expose GameManager for cross-game batched search
    py::class_<PyGameManager>(m, "GameManager")
        .def(py::init<const std::string&, const std::string&, py::dict, bool, bool, const std::string&>(),
             py::arg("model_path"),
             py::arg("device") = "cpu",
             py::arg("config") = py::dict(),
             py::arg("use_fp16") = false,
             py::arg("use_trt") = false,
             py::arg("trt_engine_path") = "")
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
