#include "core/attacks.h"

namespace attacks {

static Bitboard knight_attacks[NUM_SQUARES];
static Bitboard king_attacks[NUM_SQUARES];
static Bitboard pawn_attacks[NUM_COLORS][NUM_SQUARES];

static Bitboard ray_attacks(Square s, Direction d, Bitboard occupied) {
    Bitboard attacks = 0;
    Bitboard bb = square_bb(s);

    while (true) {
        if (d == NORTH)           bb <<= 8;
        else if (d == SOUTH)      bb >>= 8;
        else if (d == EAST)       { if (bb & FILE_H_BB) break; bb <<= 1; }
        else if (d == WEST)       { if (bb & FILE_A_BB) break; bb >>= 1; }
        else if (d == NORTH_EAST) { if (bb & FILE_H_BB) break; bb <<= 9; }
        else if (d == NORTH_WEST) { if (bb & FILE_A_BB) break; bb <<= 7; }
        else if (d == SOUTH_EAST) { if (bb & FILE_H_BB) break; bb >>= 7; }
        else if (d == SOUTH_WEST) { if (bb & FILE_A_BB) break; bb >>= 9; }

        if (bb == 0) break;
        attacks |= bb;
        if (bb & occupied) break;
    }

    return attacks;
}

static void init_knight_attacks() {
    const int offsets[8][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard bb = 0;
        int r = rank_of(Square(sq));
        int f = file_of(Square(sq));
        for (auto& [dr, df] : offsets) {
            int nr = r + dr, nf = f + df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                bb |= square_bb(make_square(nf, nr));
        }
        knight_attacks[sq] = bb;
    }
}

static void init_king_attacks() {
    const int offsets[8][2] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
    };
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard bb = 0;
        int r = rank_of(Square(sq));
        int f = file_of(Square(sq));
        for (auto& [dr, df] : offsets) {
            int nr = r + dr, nf = f + df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                bb |= square_bb(make_square(nf, nr));
        }
        king_attacks[sq] = bb;
    }
}

static void init_pawn_attacks() {
    for (int sq = 0; sq < 64; ++sq) {
        int r = rank_of(Square(sq));
        int f = file_of(Square(sq));

        Bitboard white_atk = 0, black_atk = 0;

        if (r < 7) {
            if (f > 0) white_atk |= square_bb(make_square(f - 1, r + 1));
            if (f < 7) white_atk |= square_bb(make_square(f + 1, r + 1));
        }

        if (r > 0) {
            if (f > 0) black_atk |= square_bb(make_square(f - 1, r - 1));
            if (f < 7) black_atk |= square_bb(make_square(f + 1, r - 1));
        }

        pawn_attacks[WHITE][sq] = white_atk;
        pawn_attacks[BLACK][sq] = black_atk;
    }
}

void init() {
    init_knight_attacks();
    init_king_attacks();
    init_pawn_attacks();
}

Bitboard knight(Square s) { return knight_attacks[s]; }
Bitboard king(Square s)   { return king_attacks[s]; }
Bitboard pawn(Color c, Square s) { return pawn_attacks[c][s]; }

Bitboard bishop(Square s, Bitboard occupied) {
    return ray_attacks(s, NORTH_EAST, occupied)
         | ray_attacks(s, NORTH_WEST, occupied)
         | ray_attacks(s, SOUTH_EAST, occupied)
         | ray_attacks(s, SOUTH_WEST, occupied);
}

Bitboard rook(Square s, Bitboard occupied) {
    return ray_attacks(s, NORTH, occupied)
         | ray_attacks(s, SOUTH, occupied)
         | ray_attacks(s, EAST,  occupied)
         | ray_attacks(s, WEST,  occupied);
}

Bitboard queen(Square s, Bitboard occupied) {
    return bishop(s, occupied) | rook(s, occupied);
}

} // namespace attacks
