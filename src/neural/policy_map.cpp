#include "neural/policy_map.h"
#include <unordered_map>

namespace neural {

// Direction vectors: (file_delta, rank_delta) — same order as Python encoder
static constexpr int QUEEN_DIRS[8][2] = {
    { 0,  1},  // N
    { 1,  1},  // NE
    { 1,  0},  // E
    { 1, -1},  // SE
    { 0, -1},  // S
    {-1, -1},  // SW
    {-1,  0},  // W
    {-1,  1},  // NW
};

static constexpr int KNIGHT_DELTAS[8][2] = {
    { 1,  2}, { 2,  1}, { 2, -1}, { 1, -2},
    {-1, -2}, {-2, -1}, {-2,  1}, {-1,  2},
};

// Underpromotion directions: capture-left, forward, capture-right
static constexpr int PROMO_DIRS[3][2] = {
    {-1, 1},  // capture left
    { 0, 1},  // forward
    { 1, 1},  // capture right
};

// Underpromotion pieces in order: KNIGHT(1), BISHOP(2), ROOK(3)
static constexpr PieceType PROMO_PIECES[3] = { KNIGHT, BISHOP, ROOK };

// Key encoding: (from_sq << 16) | (to_sq << 8) | promo_code
// promo_code: 0 = NO_PIECE_TYPE (non-underpromotion), 1=KNIGHT, 2=BISHOP, 3=ROOK
static uint32_t make_key(int from_sq, int to_sq, int promo_code) {
    return (uint32_t(from_sq) << 16) | (uint32_t(to_sq) << 8) | uint32_t(promo_code);
}

static std::unordered_map<uint32_t, int>& get_table() {
    static std::unordered_map<uint32_t, int> table;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        int idx = 0;
        for (int from_sq = 0; from_sq < 64; ++from_sq) {
            int f = from_sq & 7;
            int r = from_sq >> 3;

            // Queen-like moves: 8 directions × up to 7 distances
            for (int d = 0; d < 8; ++d) {
                int df = QUEEN_DIRS[d][0];
                int dr = QUEEN_DIRS[d][1];
                for (int dist = 1; dist <= 7; ++dist) {
                    int nf = f + df * dist;
                    int nr = r + dr * dist;
                    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                        int to_sq = nr * 8 + nf;
                        table[make_key(from_sq, to_sq, 0)] = idx;
                        ++idx;
                    }
                }
            }

            // Knight moves
            for (int k = 0; k < 8; ++k) {
                int nf = f + KNIGHT_DELTAS[k][0];
                int nr = r + KNIGHT_DELTAS[k][1];
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                    int to_sq = nr * 8 + nf;
                    table[make_key(from_sq, to_sq, 0)] = idx;
                    ++idx;
                }
            }

            // Underpromotions: only from rank 6 (0-indexed), to rank 7
            if (r == 6) {
                for (int p = 0; p < 3; ++p) {
                    int nf = f + PROMO_DIRS[p][0];
                    int nr = r + PROMO_DIRS[p][1];  // = 7
                    if (nf >= 0 && nf < 8 && nr == 7) {
                        int to_sq = nr * 8 + nf;
                        for (int piece_idx = 0; piece_idx < 3; ++piece_idx) {
                            int promo_code = int(PROMO_PIECES[piece_idx]);  // 1=N, 2=B, 3=R
                            table[make_key(from_sq, to_sq, promo_code)] = idx;
                            ++idx;
                        }
                    }
                }
            }
        }
    }
    return table;
}

int move_to_policy_index(Square from_sq, Square to_sq, PieceType promo) {
    if (from_sq == to_sq) return -1;

    // promo_code: 0 for non-underpromotion (includes NO_PIECE_TYPE and queen promo)
    // 1=KNIGHT, 2=BISHOP, 3=ROOK for underpromotion
    int promo_code = 0;
    if (promo == KNIGHT || promo == BISHOP || promo == ROOK) {
        promo_code = int(promo);
    }

    auto& table = get_table();
    auto it = table.find(make_key(int(from_sq), int(to_sq), promo_code));
    if (it == table.end()) return -1;
    return it->second;
}

int move_to_policy_index(Move move, Color side_to_move) {
    Square from = move.from();
    Square to   = move.to();

    // Mirror squares for Black (rank 0 <-> rank 7)
    if (side_to_move == BLACK) {
        from = make_square(file_of(from), 7 - rank_of(from));
        to   = make_square(file_of(to),   7 - rank_of(to));
    }

    // Determine promo piece type
    PieceType promo = NO_PIECE_TYPE;
    if (move.is_promotion()) {
        PieceType pp = move.promo_piece();
        // Queen promotion → NO_PIECE_TYPE (uses normal queen-move encoding)
        // Knight/Bishop/Rook → underpromotion encoding
        if (pp == KNIGHT || pp == BISHOP || pp == ROOK) {
            promo = pp;
        }
        // else pp == QUEEN → promo stays NO_PIECE_TYPE
    }

    return move_to_policy_index(from, to, promo);
}

} // namespace neural
