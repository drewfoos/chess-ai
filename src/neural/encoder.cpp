#include "neural/encoder.h"
#include "core/types.h"
#include <cstring>

namespace neural {

static void fill_plane(float* output, int plane, float value) {
    float* start = output + plane * 64;
    for (int i = 0; i < 64; i++) start[i] = value;
}

void encode_position(const Position& pos, float* output) {
    std::memset(output, 0, TENSOR_SIZE * sizeof(float));

    bool is_white = (pos.side_to_move() == WHITE);

    // Encode current position (time step 0, planes 0-11 = pieces, plane 12 = repetition)
    for (int sq = 0; sq < 64; sq++) {
        PieceType pt = pos.piece_on(Square(sq));
        if (pt == NO_PIECE_TYPE) continue;

        Color c = pos.color_on(Square(sq));

        // Flip square for black to move
        int actual_sq;
        if (!is_white) {
            int sq_file = sq & 7;
            int sq_rank = sq >> 3;
            actual_sq = (7 - sq_rank) * 8 + sq_file;
        } else {
            actual_sq = sq;
        }

        int plane;
        if (c == pos.side_to_move()) {
            plane = int(pt);           // 0-5: our pieces
        } else {
            plane = 6 + int(pt);      // 6-11: opponent pieces
        }

        int rank = actual_sq >> 3;
        int file = actual_sq & 7;
        output[plane * 64 + rank * 8 + file] = 1.0f;
    }

    // Plane 12: repetition count (zero for now — no history tracking in C++ yet)
    // Already zeroed by memset.

    // Copy time step 0 (13 planes * 64 = 832 floats) to steps 1-7
    for (int t = 1; t < 8; t++) {
        std::memcpy(output + t * 13 * 64, output, 13 * 64 * sizeof(float));
    }

    // Constant planes 104-111
    // Plane 104: color (1.0 if white to move, 0.0 if black)
    if (is_white) fill_plane(output, 104, 1.0f);

    // Plane 105: move count (fullmove_number / 200)
    fill_plane(output, 105, pos.fullmove_number() / 200.0f);

    // Planes 106-109: castling rights from STM perspective
    CastlingRight cr = pos.castling_rights();
    if (is_white) {
        if (cr & WHITE_OO)  fill_plane(output, 106, 1.0f);
        if (cr & WHITE_OOO) fill_plane(output, 107, 1.0f);
        if (cr & BLACK_OO)  fill_plane(output, 108, 1.0f);
        if (cr & BLACK_OOO) fill_plane(output, 109, 1.0f);
    } else {
        if (cr & BLACK_OO)  fill_plane(output, 106, 1.0f);
        if (cr & BLACK_OOO) fill_plane(output, 107, 1.0f);
        if (cr & WHITE_OO)  fill_plane(output, 108, 1.0f);
        if (cr & WHITE_OOO) fill_plane(output, 109, 1.0f);
    }

    // Plane 110: halfmove clock / 100
    fill_plane(output, 110, pos.halfmove_clock() / 100.0f);

    // Plane 111: bias plane (all ones)
    fill_plane(output, 111, 1.0f);
}

} // namespace neural
