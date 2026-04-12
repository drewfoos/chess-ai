#pragma once
#include "core/types.h"
#include "core/position.h"
#include "mcts/node.h"
#include "mcts/nn_cache.h"
#include "neural/position_history.h"
#include "neural/policy_map.h"
#include <vector>
#include <array>
#include <memory>
#include <atomic>
#include <functional>

namespace mcts {

// Result of evaluating a position
struct EvalResult {
    std::vector<float> policy;  // Prior probability per legal move (same order as moves array)
    float value;                // Position evaluation from side-to-move perspective: [-1, +1]
};

// Request for batch evaluation
struct BatchEvalRequest {
    Position position;
    Move legal_moves[MAX_MOVES];
    int num_legal_moves;
};

// Abstract evaluator interface — neural network plugs in here later
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) = 0;

    // Batch evaluation — default falls back to individual calls.
    // NeuralEvaluator overrides with single GPU forward pass.
    virtual std::vector<EvalResult> evaluate_batch(const std::vector<BatchEvalRequest>& requests) {
        std::vector<EvalResult> results;
        results.reserve(requests.size());
        for (const auto& req : requests) {
            results.push_back(evaluate(req.position, req.legal_moves, req.num_legal_moves));
        }
        return results;
    }
};

// Stub evaluator for testing MCTS without a neural network
// Returns uniform policy over legal moves and material-based value
class RandomEvaluator : public Evaluator {
public:
    EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) override;
};

// Search parameters
struct SearchParams {
    int num_iterations = 800;
    // Dynamic c_puct: c_init + c_factor * log((N + c_base) / c_base)
    // Lc0 current defaults
    float c_puct_init = 3.0f;
    float c_puct_base = 19652.0f;
    float c_puct_factor = 2.0f;
    float fpu_reduction_root = 1.2f;
    float fpu_reduction = 1.2f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    bool add_noise = true;    // Add Dirichlet noise at root (for self-play)
    float policy_softmax_temp = 2.2f;  // Temperature applied to NN policy logits

    // Batched search config
    int batch_size = 128;
    bool smart_pruning = true;
    float smart_pruning_factor = 1.33f;
    bool two_fold_draw = true;
    bool shaped_dirichlet = true;
    float uncertainty_weight = 0.15f;
    bool variance_scaling = true;
    float contempt = 0.0f;
    bool sibling_blending = true;
    int nn_cache_size = 200000;
};

// Search result
struct SearchResult {
    Move best_move;
    std::vector<Move> moves;           // Legal moves at root
    std::vector<int> visit_counts;     // Visit count per move
    float root_value;                  // Value estimate at root
    int total_nodes;                   // Total nodes in tree

    // New fields for training data
    std::array<float, neural::POLICY_SIZE> policy_target = {};   // Normalized visit distribution mapped to 1858 policy indices
    std::array<float, neural::POLICY_SIZE> raw_policy = {};      // NN policy before MCTS
    float raw_value = 0.0f;                                       // NN value before MCTS
};

struct SearchInfo {
    int iterations;       // Simulations completed
    int total_nodes;      // Nodes in tree
    float root_value;     // Current root value [-1, +1]
    Move best_move;       // Current best move
};

using InfoCallback = std::function<void(const SearchInfo&)>;

class Search {
public:
    Search(Evaluator& evaluator, const SearchParams& params = SearchParams{});

    // Backward-compatible: wraps pos in a 1-entry PositionHistory
    SearchResult run(const Position& pos);

    // Full history for repetition detection
    SearchResult run(const neural::PositionHistory& history);

    // Temperature-based move selection for self-play
    // temperature = 1.0: proportional to visit counts
    // temperature → 0: greedy (pick most visited)
    static Move select_move_with_temperature(const SearchResult& result, float temperature);

    void set_stop_flag(std::atomic<bool>* stop);
    void set_info_callback(InfoCallback cb);

private:
    std::atomic<bool>* stop_flag_ = nullptr;
    InfoCallback info_callback_;
    Evaluator& evaluator_;
    SearchParams params_;
    NNCache cache_;

    // Batched search internals
    struct PendingEval {
        Node* leaf;
        Position position;
        std::vector<Move> path_moves;      // Moves from root to leaf for position tracking
        std::vector<uint64_t> path_hashes; // Hashes along selection path
        std::vector<Node*> path_nodes;     // Nodes with virtual loss applied (for reverting)
    };

    // Selection with all features: solver, variance scaling, uncertainty, sibling blending
    Node* select_child_advanced(Node* node, bool is_root);

    // Gather phase: select leaves with virtual loss
    void gather_leaf(Node* root, const neural::PositionHistory& history,
                     std::vector<PendingEval>& batch);

    // Expand a node with evaluation results
    void expand_node(Node* node, const Position& pos, const EvalResult& eval_result);

    // Backpropagate value up the tree
    void backpropagate(Node* node, float value);

    // Propagate terminal status up the tree
    void propagate_terminal(Node* node);

    // Dirichlet noise (shaped or uniform)
    void add_dirichlet_noise(Node* root);
    void add_shaped_dirichlet_noise(Node* root);

    float dynamic_cpuct(int parent_visits) const;

    // Build position by replaying moves from root
    void replay_moves(const Position& root_pos, const std::vector<Move>& moves, Position& out_pos);

    // Check for two-fold repetition during selection
    bool is_two_fold_repetition(uint64_t hash, const std::vector<uint64_t>& path_hashes,
                                 const neural::PositionHistory& history) const;

    // Build the final result with contempt
    SearchResult build_result(Node* root, const Position& root_pos,
                              float raw_value, const std::array<float, neural::POLICY_SIZE>& raw_policy);
};

} // namespace mcts
