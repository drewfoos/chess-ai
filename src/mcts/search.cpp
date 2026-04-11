#include "mcts/search.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>

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

// --- Search (stub implementations, filled in Task 4) ---

Search::Search(Evaluator& evaluator, const SearchParams& params)
    : evaluator_(evaluator), params_(params) {}

SearchResult Search::run(const Position& /*pos*/) {
    return SearchResult{};
}

Node* Search::select(Node* /*root*/) { return nullptr; }
void Search::expand(Node* /*node*/, const Position& /*pos*/) {}
float Search::evaluate(Node* /*node*/, const Position& /*pos*/) { return 0.0f; }
void Search::backpropagate(Node* /*node*/, float /*value*/) {}
void Search::add_dirichlet_noise(Node* /*root*/) {}
void Search::apply_moves_to_root(const Position& /*root_pos*/, Node* /*node*/, Position& /*out_pos*/) {}

} // namespace mcts
