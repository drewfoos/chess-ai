#pragma once
#include "core/position.h"
#include "neural/position_history.h"
#include <cstdint>

namespace neural {

constexpr int INPUT_PLANES = 112;
constexpr int BOARD_SIZE = 8;
constexpr int TENSOR_SIZE = INPUT_PLANES * BOARD_SIZE * BOARD_SIZE;  // 7168
constexpr int PACKED_PLANES = 104;  // 8 history steps × 13 planes

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

// Bitpacked training-data representation. Planes 0-103 are binary piece-
// occupancy planes from encode_position, packed as uint64 bitboards in LERF
// (bit i = square i, a1 = LSB, h8 = MSB). Rank-flip for STM is baked in, so
// the bitboards are in STM-canonical orientation exactly like the dense form.
// The 8 scalar planes (104-111) are reduced to a few bytes of metadata here;
// plane 111 (ones) is always constant and regenerated at expand time.
struct PackedPosition {
    uint64_t planes[PACKED_PLANES];  // 8 history × 13 planes, 0 for unset
    uint8_t  castling;               // bits: 0=STM-K, 1=STM-Q, 2=OPP-K, 3=OPP-Q
    uint8_t  rule50;                 // halfmove clock, 0..100
    uint16_t fullmove;               // fullmove number
    uint8_t  stm;                    // 1 = white to move, 0 = black (stored as byte for ABI stability)
};

void encode_position_packed(const PositionHistory& history, PackedPosition& out);
void encode_position_packed(const Position& pos, PackedPosition& out);

} // namespace neural
