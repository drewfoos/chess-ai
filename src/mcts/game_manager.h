#pragma once
#ifdef HAS_LIBTORCH

#include "core/types.h"
#include "core/position.h"
#include "core/movegen.h"
#include "mcts/node.h"
#include "mcts/search.h"
#include "mcts/nn_cache.h"
#include "neural/neural_evaluator.h"
#include "neural/position_history.h"
#include "neural/encoder.h"
#include "neural/policy_map.h"
#include <vector>
#include <memory>
#include <string>

namespace mcts {

class GameManager {
public:
    GameManager(neural::NeuralEvaluator& evaluator, const SearchParams& params);

    // Initialize a game at index with a position and history
    void init_game(int idx, const neural::PositionHistory& history, int num_sims);

    // Initialize N games all from starting position
    void init_games(int num_games, int num_sims);

    // Initialize games from FEN strings and move histories
    void init_games_from_fen(const std::vector<std::string>& fens,
                             const std::vector<std::vector<std::string>>& move_histories,
                             int num_sims);

    // Run one step of cross-game batching. Returns number of newly completed games.
    int step();

    // Check if all games are done
    bool all_complete() const;

    // Check if specific game is done
    bool is_complete(int idx) const;

    // Get result for a completed game
    SearchResult get_result(int idx) const;

    // Get number of games
    int num_games() const { return static_cast<int>(games_.size()); }

private:
    struct GameState {
        neural::PositionHistory history;
        Node* root = nullptr;  // Pool-managed
        int sims_done = 0;
        int target_sims = 400;
        bool search_complete = false;
        float raw_value = 0.0f;
        std::array<float, neural::POLICY_SIZE> raw_policy = {};
        bool root_expanded = false;
    };

    // Pending evaluation across games
    struct PendingLeaf {
        int game_idx;
        Node* leaf;
        Position position;
        std::vector<Node*> path_nodes;  // For virtual loss reversal
    };

    neural::NeuralEvaluator& evaluator_;
    SearchParams params_;
    NNCache cache_;
    NodePool pool_;
    std::vector<GameState> games_;

    // Pre-allocated buffers (reused across step() calls)
    std::vector<float> flat_encode_buffer_;
    std::vector<std::vector<Move>> legal_moves_vec_;
    std::vector<int> num_legal_moves_vec_;

    // Gather a leaf from one game (returns true if a leaf was queued for eval)
    bool gather_leaf_from_game(int game_idx, std::vector<PendingLeaf>& batch);

    // Expand root for a game using single-position evaluation
    void expand_root(GameState& game);

    // Selection with advanced features (duplicated from Search)
    Node* select_child_advanced(Node* node, bool is_root);

    // Backpropagate value up the tree
    void backpropagate(Node* node, float value);

    // Propagate terminal status up the tree
    void propagate_terminal(Node* node);

    // Dirichlet noise (shaped or uniform)
    void add_dirichlet_noise(Node* root);
    void add_shaped_dirichlet_noise(Node* root);

    // Build result from completed game
    SearchResult build_result(const GameState& game) const;

    float dynamic_cpuct(int parent_visits) const;

    // Check for two-fold repetition during selection
    bool is_two_fold_repetition(uint64_t hash, const std::vector<uint64_t>& path_hashes,
                                 const neural::PositionHistory& history) const;

    // Replay moves from root position to build a child position
    void replay_moves(const Position& root_pos, const std::vector<Move>& moves, Position& out_pos);

    // Expand a node with evaluation results
    void expand_node(Node* node, const Position& pos, const EvalResult& eval_result);

    // Check smart pruning for a game
    bool should_prune(const GameState& game) const;
};

} // namespace mcts

#endif // HAS_LIBTORCH
