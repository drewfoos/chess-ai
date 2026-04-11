#include "mcts/search.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mcts {

// --- RandomEvaluator ---

EvalResult RandomEvaluator::evaluate(const Position& pos, const Move* moves, int num_moves) {
    EvalResult result;

    // Terminal position
    if (num_moves == 0) {
        if (pos.in_check()) {
            result.value = -1.0f; // Checkmated — loss for side to move
        } else {
            result.value = 0.0f;  // Stalemate — draw
        }
        return result;
    }

    // Uniform policy over legal moves
    float uniform = 1.0f / num_moves;
    result.policy.assign(num_moves, uniform);

    // Simple material-based evaluation
    Color us = pos.side_to_move();
    Color them = ~us;

    static constexpr float piece_values[] = {
        1.0f,   // PAWN
        3.0f,   // KNIGHT
        3.0f,   // BISHOP
        5.0f,   // ROOK
        9.0f,   // QUEEN
        0.0f    // KING
    };

    float material = 0.0f;
    for (int pt = PAWN; pt <= QUEEN; pt++) {
        material += piece_values[pt] * popcount(pos.pieces(us, PieceType(pt)));
        material -= piece_values[pt] * popcount(pos.pieces(them, PieceType(pt)));
    }

    // Squash material advantage into [-1, 1] using tanh-like scaling
    result.value = material / (std::abs(material) + 3.0f);

    return result;
}

// --- Search ---

// Helper: collect path of moves from root to a given node
static void collect_path(Node* node, std::vector<Move>& path) {
    path.clear();
    std::vector<Move> reversed;
    Node* cur = node;
    while (cur->parent() != nullptr) {
        reversed.push_back(cur->move());
        cur = cur->parent();
    }
    for (int i = static_cast<int>(reversed.size()) - 1; i >= 0; i--) {
        path.push_back(reversed[i]);
    }
}

void Search::apply_moves_to_root(const Position& root_pos, Node* node, Position& out_pos) {
    std::vector<Move> path;
    collect_path(node, path);

    out_pos = root_pos;
    UndoInfo undo;
    for (Move m : path) {
        out_pos.make_move(m, undo);
    }
}

Search::Search(Evaluator& evaluator, const SearchParams& params)
    : evaluator_(evaluator), params_(params) {}

Node* Search::select(Node* node) {
    while (!node->is_leaf()) {
        bool is_root = (node->parent() == nullptr);
        float fpu_reduction = is_root ? params_.fpu_reduction_root : params_.fpu_reduction;
        float parent_q = node->mean_value();
        float fpu_value = parent_q - fpu_reduction;

        node = node->select_child(params_.c_puct, fpu_value);
    }
    return node;
}

void Search::expand(Node* node, const Position& pos) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 0) return; // Terminal node

    EvalResult eval_result = evaluator_.evaluate(pos, moves, num_moves);

    for (int i = 0; i < num_moves; i++) {
        node->add_child(moves[i], eval_result.policy[i]);
    }
}

float Search::evaluate(Node* node, const Position& pos) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    EvalResult result = evaluator_.evaluate(pos, moves, num_moves);
    return result.value;
}

void Search::backpropagate(Node* node, float value) {
    while (node != nullptr) {
        node->update(value);
        value = -value;
        node = node->parent();
    }
}

void Search::add_dirichlet_noise(Node* root) {
    if (root->is_leaf()) return;

    int num_children = root->num_children();
    std::vector<float> noise(num_children);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(params_.dirichlet_alpha, 1.0f);

    float noise_sum = 0.0f;
    for (int i = 0; i < num_children; i++) {
        noise[i] = gamma(gen);
        noise_sum += noise[i];
    }
    for (int i = 0; i < num_children; i++) {
        noise[i] /= noise_sum;
    }

    float eps = params_.dirichlet_epsilon;
    for (int i = 0; i < num_children; i++) {
        float old_prior = root->child(i)->prior();
        float new_prior = (1.0f - eps) * old_prior + eps * noise[i];
        root->child(i)->set_prior(new_prior);
    }
}

SearchResult Search::run(const Position& pos) {
    auto root = std::make_unique<Node>();

    // Expand root
    expand(root.get(), pos);

    if (root->is_leaf()) {
        // No legal moves — terminal position
        SearchResult result;
        result.best_move = Move::none();
        result.root_value = evaluate(root.get(), pos);
        result.total_nodes = 1;
        return result;
    }

    // Initial root evaluation and visit
    float root_value = evaluate(root.get(), pos);
    root->update(root_value);

    // Add Dirichlet noise at root for exploration
    if (params_.add_noise) {
        add_dirichlet_noise(root.get());
    }

    // Main MCTS loop
    for (int iter = 0; iter < params_.num_iterations; iter++) {
        // 1. SELECT — descend to a leaf
        Node* leaf = select(root.get());

        // 2. Get position at leaf
        Position leaf_pos;
        apply_moves_to_root(pos, leaf, leaf_pos);

        // Check if leaf is a terminal node (already visited, no children = terminal)
        if (leaf->visit_count() > 0) {
            // Already visited leaf — evaluate and backprop without expanding.
            // Negate value: evaluate() returns value from side-to-move-at-leaf perspective,
            // but the leaf node stores value from the parent's (mover's) perspective.
            float value = evaluate(leaf, leaf_pos);
            backpropagate(leaf, -value);
            continue;
        }

        // 3. EVALUATE — get value for this leaf
        float value = evaluate(leaf, leaf_pos);

        // 4. EXPAND — create children
        expand(leaf, leaf_pos);

        // 5. BACKPROPAGATE — update all nodes on path.
        // Negate value: evaluate() returns value from side-to-move-at-leaf perspective,
        // but the leaf node stores value from the parent's (mover's) perspective.
        backpropagate(leaf, -value);
    }

    // Build result
    SearchResult result;
    result.best_move = root->best_move();
    result.root_value = root->mean_value();

    int num_children = root->num_children();
    result.moves.resize(num_children);
    result.visit_counts.resize(num_children);
    for (int i = 0; i < num_children; i++) {
        result.moves[i] = root->child(i)->move();
        result.visit_counts[i] = root->child(i)->visit_count();
    }

    result.total_nodes = root->visit_count();

    return result;
}

Move Search::select_move_with_temperature(const SearchResult& result, float temperature) {
    if (result.moves.empty()) return Move::none();

    // Temperature ≈ 0: greedy selection
    if (temperature < 0.01f) {
        int best_idx = 0;
        for (int i = 1; i < static_cast<int>(result.visit_counts.size()); i++) {
            if (result.visit_counts[i] > result.visit_counts[best_idx]) {
                best_idx = i;
            }
        }
        return result.moves[best_idx];
    }

    // Temperature-based sampling: π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)
    int num_moves = static_cast<int>(result.moves.size());
    std::vector<float> probs(num_moves);
    float inv_temp = 1.0f / temperature;

    float sum = 0.0f;
    for (int i = 0; i < num_moves; i++) {
        probs[i] = std::pow(static_cast<float>(result.visit_counts[i]), inv_temp);
        sum += probs[i];
    }

    if (sum <= 0.0f) {
        // Fallback: uniform random
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, num_moves - 1);
        return result.moves[dist(gen)];
    }

    for (int i = 0; i < num_moves; i++) {
        probs[i] /= sum;
    }

    // Sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return result.moves[dist(gen)];
}

} // namespace mcts
