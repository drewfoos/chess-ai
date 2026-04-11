#pragma once
#include "core/bitboard.h"

namespace attacks {

void init();

Bitboard knight(Square s);
Bitboard king(Square s);
Bitboard pawn(Color c, Square s);

Bitboard bishop(Square s, Bitboard occupied);
Bitboard rook(Square s, Bitboard occupied);
Bitboard queen(Square s, Bitboard occupied);

} // namespace attacks
