#pragma once
#include "core/position.h"
#include "neural/position_history.h"

namespace neural {

constexpr int INPUT_PLANES = 112;
constexpr int BOARD_SIZE = 8;
constexpr int TENSOR_SIZE = INPUT_PLANES * BOARD_SIZE * BOARD_SIZE;  // 7168

// Encodes a chess position into a 112×8×8 float tensor (layout: [plane][rank][file]).
// Planes 0-103 : 8 time steps × 13 planes each
//   0-5   : current player's pieces (pawn, knight, bishop, rook, queen, king)
//   6-11  : opponent's pieces (same order)
//   12    : repetition count (placeholder, always 0)
// Planes 104-111 : constant feature planes
//   104  : side to move (1.0 = white, 0.0 = black)
//   105  : fullmove number / 200
//   106  : STM kingside castling
//   107  : STM queenside castling
//   108  : opponent kingside castling
//   109  : opponent queenside castling
//   110  : halfmove clock / 100
//   111  : bias (all ones)
void encode_position(const Position& pos, float* output);

// Encode from history (fills all 8 time steps with real positions)
void encode_position(const PositionHistory& history, float* output);

} // namespace neural
