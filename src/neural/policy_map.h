#pragma once
#include "core/types.h"

namespace neural {

constexpr int POLICY_SIZE = 1858;

// Low-level: from/to already in side-to-move perspective.
// promo = NO_PIECE_TYPE for non-underpromotion (including queen promo).
// promo = KNIGHT/BISHOP/ROOK for underpromotion.
// Returns index in [0, 1857] or -1.
int move_to_policy_index(Square from_sq, Square to_sq, PieceType promo);

// High-level: extracts from/to/promo from Move, mirrors for Black.
int move_to_policy_index(Move move, Color side_to_move);

} // namespace neural
