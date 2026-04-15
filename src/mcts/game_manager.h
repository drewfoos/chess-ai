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
#include <array>

namespace mcts {

// Per-root search statistics returned by GameManager::step_stats() after a batch
// of MCTS iterations. Lets Python orchestrate move selection, recording, and
// adjudication without committing a move in C++.
//
// Conventions:
//  - q_per_child[i] is parent-POV (already flipped from the child's stored Q,
//    which is from the child side-to-move's perspective).
//  - root_wdl is parent-POV and derived from the root node's mean_value() (a
//    placeholder until WDL flows through RawBatchEvaluator end-to-end).
//  - terminal_status: 0=ongoing, 1=win-for-side-to-move, -1=loss-for-stm, 2=draw.
struct RootStats {
    int game_idx = 0;
    bool game_complete = false;
    int terminal_status = 0;
    int n_legal = 0;
    std::vector<Move> legal_moves;      // size n_legal
    std::vector<int> visits;            // size n_legal
    std::vector<float> q_per_child;     // size n_legal, parent-POV (already flipped)
    int best_child_idx = -1;            // argmax(visits); -1 if root has no edges
    std::array<float, 3> root_wdl{0.0f, 0.0f, 0.0f};
    std::vector<float> raw_nn_policy;   // size POLICY_SIZE (1858)
    std::array<float, 3> raw_nn_value{0.0f, 0.0f, 0.0f};
    float raw_nn_mlh = 0.0f;
    int sims_done = 0;
};

class GameManager {
public:
    GameManager(neural::RawBatchEvaluator& evaluator, const SearchParams& params);

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

    // New API (Stage 1, Lc0-parity self-play refactor): run MCTS until each game
    // reaches target_sims[i] (or becomes terminal), then return per-game RootStats
    // *without* committing a move. Caller (Python orchestrator) chooses the move
    // via apply_move(). Does not alter step()'s behavior — runs alongside it.
    std::vector<RootStats> step_stats(const std::vector<int>& target_sims);

    // Commit the move at legal-move index `move_idx` for game `game_idx`:
    // pushes the move onto history, resets the search tree (no subtree reuse,
    // matching Lc0 self-play default), and clears per-step flags.
    void apply_move(int game_idx, int move_idx);

    // Current FEN of a game.
    std::string get_fen(int game_idx) const;

    // Current ply count of a game (number of moves played since init).
    int get_ply(int game_idx) const;

    // Check if all games are done
    bool all_complete() const;

    // Check if specific game is done
    bool is_complete(int idx) const;

    // Get result for a completed game
    SearchResult get_result(int idx) const;

    // Get number of games
    int num_games() const { return static_cast<int>(games_.size()); }

    // Test-only: access the raw root Node for a game. Used by q-flip POV tests
    // to compare RootStats.q_per_child against the child's stored mean_value.
    // Not part of the stable public API — may change without notice.
    Node* test_root(int idx) const {
        if (idx < 0 || idx >= static_cast<int>(games_.size())) return nullptr;
        return games_[idx].root;
    }

private:
    struct GameState {
        neural::PositionHistory history;
        Node* root = nullptr;  // Pool-managed
        int sims_done = 0;
        int target_sims = 400;
        int ply = 0;           // Number of moves played since init (for get_ply / Python orchestration)
        bool search_complete = false;
        float raw_value = 0.0f;
        float raw_mlh = 0.0f;  // Root NN moves-left estimate (pre-search) — surfaced in RootStats
        std::array<float, neural::POLICY_SIZE> raw_policy = {};
        bool root_expanded = false;
    };

    // Pending evaluation across games
    struct PendingLeaf {
        int game_idx;
        Node* leaf;
        Position position;
        std::vector<Node*> path_nodes;  // For virtual loss reversal
        int multivisit = 1;             // Number of collapsed visits (multivisit optimization)
    };

    neural::RawBatchEvaluator& evaluator_;
    SearchParams params_;
    NNCache cache_;
    NodePool pool_;
    std::vector<GameState> games_;

    // Pre-allocated buffers (reused across step() calls)
    std::vector<float> flat_encode_buffer_;
    std::vector<std::vector<Move>> legal_moves_vec_;
    std::vector<int> num_legal_moves_vec_;

    // Gather a leaf from one game. Returns the number of simulations this descent
    // contributed (0 = early-exit without NN queue, 1 = standard, N = multivisit collapse).
    int gather_leaf_from_game(int game_idx, std::vector<PendingLeaf>& batch);

    // Expand root for a game using single-position evaluation
    void expand_root(GameState& game);

    // Selection with advanced features (duplicated from Search). Optionally returns edge index.
    Node* select_child_advanced(Node* node, bool is_root);
    Node* select_child_advanced(Node* node, bool is_root, int* out_idx);

    // Compute how many additional visits PUCT would route to the selected child
    // before switching to a sibling. Returns value in [1, max_collapse_visits].
    int compute_collapse_visits(Node* parent, int best_idx, bool is_root);

    // Backpropagate value up the tree. n > 1 folds N multivisit updates into one walk.
    void backpropagate(Node* node, float value, int n = 1);

    // Propagate terminal status up the tree
    void propagate_terminal(Node* node);

    // Dirichlet noise (shaped or uniform)
    void add_dirichlet_noise(Node* root);
    void add_shaped_dirichlet_noise(Node* root);

    // Build result from completed game
    SearchResult build_result(const GameState& game) const;

    // Build the public RootStats snapshot for a single game.
    RootStats build_root_stats(int idx) const;

    // Shared batched-evaluate-then-backprop used by both step() and step_stats().
    // Expands each pending leaf with the returned (policy,value,mlh), caches the
    // entry, then reverts virtual loss + backpropagates.
    void evaluate_and_backprop_batch(std::vector<PendingLeaf>& batch);

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
