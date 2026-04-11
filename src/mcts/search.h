#pragma once
#include "core/types.h"
#include "core/position.h"
#include "mcts/node.h"
#include <vector>
#include <memory>

namespace mcts {

// Result of evaluating a position
struct EvalResult {
    std::vector<float> policy;  // Prior probability per legal move (same order as moves array)
    float value;                // Position evaluation from side-to-move perspective: [-1, +1]
};

// Abstract evaluator interface — neural network plugs in here later
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) = 0;
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
    float c_puct = 2.5f;
    float fpu_reduction_root = 0.44f;
    float fpu_reduction = 0.25f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    bool add_noise = true;    // Add Dirichlet noise at root (for self-play)
};

// Search result
struct SearchResult {
    Move best_move;
    std::vector<Move> moves;           // Legal moves at root
    std::vector<int> visit_counts;     // Visit count per move
    float root_value;                  // Value estimate at root
    int total_nodes;                   // Total nodes in tree
};

class Search {
public:
    Search(Evaluator& evaluator, const SearchParams& params = SearchParams{});

    SearchResult run(const Position& pos);

    // Temperature-based move selection for self-play
    // temperature = 1.0: proportional to visit counts
    // temperature → 0: greedy (pick most visited)
    static Move select_move_with_temperature(const SearchResult& result, float temperature);

private:
    Evaluator& evaluator_;
    SearchParams params_;

    Node* select(Node* root);
    void expand(Node* node, const Position& pos);
    float evaluate(Node* node, const Position& pos);
    void backpropagate(Node* node, float value);
    void add_dirichlet_noise(Node* root);

    // Position tracking during selection
    void apply_moves_to_root(const Position& root_pos, Node* node, Position& out_pos);
};

} // namespace mcts
