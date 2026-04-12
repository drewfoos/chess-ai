#include "neural/encoder.h"
#include "core/types.h"
#include <cstring>

namespace neural {

static void fill_plane(float* output, int plane, float value) {
    float* start = output + plane * 64;
    for (int i = 0; i < 64; i++) start[i] = value;
}

// Encode pieces from a single position into a specific time step's planes.
// base_plane is the starting plane index for this time step (t * 13).
// is_white indicates whether the current player (at time step 0) is white,
// used for board flipping consistency.
static void encode_pieces(const Position& pos, float* output, int base_plane, bool is_white) {
    for (int sq = 0; sq < 64; sq++) {
        PieceType pt = pos.piece_on(Square(sq));
        if (pt == NO_PIECE_TYPE) continue;

        Color c = pos.color_on(Square(sq));

        // Flip square for black to move (from current player's perspective)
        int actual_sq;
        if (!is_white) {
            int sq_file = sq & 7;
            int sq_rank = sq >> 3;
            actual_sq = (7 - sq_rank) * 8 + sq_file;
        } else {
            actual_sq = sq;
        }

        // "Our" pieces are always the current player's (history.current().side_to_move()),
        // regardless of whose turn it was at this historical position.
        int plane;
        if (c == (is_white ? WHITE : BLACK)) {
            plane = base_plane + int(pt);           // 0-5: current player's pieces
        } else {
            plane = base_plane + 6 + int(pt);      // 6-11: opponent pieces
        }

        int rank = actual_sq >> 3;
        int file = actual_sq & 7;
        output[plane * 64 + rank * 8 + file] = 1.0f;
    }
}

void encode_position(const PositionHistory& history, float* output) {
    std::memset(output, 0, TENSOR_SIZE * sizeof(float));

    const Position& current = history.current();
    bool is_white = (current.side_to_move() == WHITE);

    // Encode 8 time steps with real historical positions
    for (int t = 0; t < 8; t++) {
        const Position& pos = history.at(t);
        int base_plane = t * 13;

        // Encode piece planes (0-11 within time step)
        encode_pieces(pos, output, base_plane, is_white);

        // Plane 12: repetition indicator
        // Check if this historical position appears more than once up to this point
        // For simplicity: check if the position at step t matches any other position in history
        if (t == 0 && history.is_repetition(2)) {
            fill_plane(output, base_plane + 12, 1.0f);
        } else if (t > 0) {
            // For older time steps, check if that position was repeated at its point in history
            // We check if positions_[idx] hash matches any earlier position
            int idx = history.length() - 1 - t;
            if (idx > 0) {
                // Use a simple approach: the position at step t is a repetition if
                // it appears elsewhere in the first idx+1 positions
                // We approximate by comparing with the current position's repetition status
                // For now, only mark step 0's repetition (matching Python encoder behavior)
            }
        }
    }

    // Constant planes 104-111 (from current position)
    // Plane 104: color (1.0 if white to move, 0.0 if black)
    if (is_white) fill_plane(output, 104, 1.0f);

    // Plane 105: move count (fullmove_number / 200)
    fill_plane(output, 105, current.fullmove_number() / 200.0f);

    // Planes 106-109: castling rights from STM perspective
    CastlingRight cr = current.castling_rights();
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
    fill_plane(output, 110, current.halfmove_clock() / 100.0f);

    // Plane 111: bias plane (all ones)
    fill_plane(output, 111, 1.0f);
}

void encode_position(const Position& pos, float* output) {
    // Wrap single position into a 1-position history and call the history overload
    PositionHistory hist;
    hist.reset(pos);
    encode_position(hist, output);
}

// Bitpacked variant. Mirrors the logic of encode_position but writes one
// uint64 per piece plane (bit per square) instead of 64 floats. The scalar
// feature planes are reduced to a few bytes of metadata.
static void encode_pieces_packed(const Position& pos, uint64_t* step_planes, bool is_white) {
    for (int sq = 0; sq < 64; sq++) {
        PieceType pt = pos.piece_on(Square(sq));
        if (pt == NO_PIECE_TYPE) continue;
        Color c = pos.color_on(Square(sq));
        int plane_in_step;
        if (c == (is_white ? WHITE : BLACK)) {
            plane_in_step = int(pt);        // 0-5 : current player's pieces
        } else {
            plane_in_step = 6 + int(pt);    // 6-11 : opponent pieces
        }
        int actual_sq;
        if (!is_white) {
            int sq_file = sq & 7;
            int sq_rank = sq >> 3;
            actual_sq = (7 - sq_rank) * 8 + sq_file;
        } else {
            actual_sq = sq;
        }
        step_planes[plane_in_step] |= (uint64_t(1) << actual_sq);
    }
}

void encode_position_packed(const PositionHistory& history, PackedPosition& out) {
    std::memset(&out, 0, sizeof(PackedPosition));

    const Position& current = history.current();
    bool is_white = (current.side_to_move() == WHITE);
    out.stm = is_white ? 1 : 0;

    for (int t = 0; t < 8; t++) {
        const Position& pos = history.at(t);
        uint64_t* step = out.planes + t * 13;
        encode_pieces_packed(pos, step, is_white);
        // Plane 12: repetition indicator. Matches dense encoder: only step 0
        // marks a 2-fold repetition; other steps are left zero.
        if (t == 0 && history.is_repetition(2)) {
            step[12] = ~uint64_t(0);
        }
    }

    // Metadata derived from current position (STM-canonical castling order).
    out.fullmove = static_cast<uint16_t>(current.fullmove_number());
    out.rule50   = static_cast<uint8_t>(current.halfmove_clock());

    CastlingRight cr = current.castling_rights();
    uint8_t castling = 0;
    if (is_white) {
        if (cr & WHITE_OO)  castling |= 0x1;
        if (cr & WHITE_OOO) castling |= 0x2;
        if (cr & BLACK_OO)  castling |= 0x4;
        if (cr & BLACK_OOO) castling |= 0x8;
    } else {
        if (cr & BLACK_OO)  castling |= 0x1;
        if (cr & BLACK_OOO) castling |= 0x2;
        if (cr & WHITE_OO)  castling |= 0x4;
        if (cr & WHITE_OOO) castling |= 0x8;
    }
    out.castling = castling;
}

void encode_position_packed(const Position& pos, PackedPosition& out) {
    PositionHistory hist;
    hist.reset(pos);
    encode_position_packed(hist, out);
}

} // namespace neural
