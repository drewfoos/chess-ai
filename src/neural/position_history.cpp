#include "neural/position_history.h"

namespace neural {

void PositionHistory::reset(const Position& pos) {
    positions_.clear();
    hashes_.clear();
    positions_.push_back(pos);
    hashes_.push_back(compute_hash(pos));
}

void PositionHistory::push(const Position& pos) {
    positions_.push_back(pos);
    hashes_.push_back(compute_hash(pos));
}

const Position& PositionHistory::current() const {
    return positions_.back();
}

const Position& PositionHistory::at(int steps_back) const {
    int idx = static_cast<int>(positions_.size()) - 1 - steps_back;
    if (idx < 0) idx = 0;
    return positions_[idx];
}

bool PositionHistory::is_repetition(int count) const {
    if (hashes_.empty()) return false;
    uint64_t current_hash = hashes_.back();
    int occurrences = 0;
    for (size_t i = 0; i < hashes_.size(); i++) {
        if (hashes_[i] == current_hash) {
            occurrences++;
            if (occurrences >= count) return true;
        }
    }
    return false;
}

uint64_t PositionHistory::compute_hash(const Position& pos) {
    uint64_t h = 0;

    // XOR all 12 piece bitboards with mixing
    for (int c = 0; c < NUM_COLORS; c++) {
        for (int pt = 0; pt < NUM_PIECE_TYPES; pt++) {
            uint64_t bb = pos.pieces(Color(c), PieceType(pt));
            // Mix: multiply by distinct large primes per (color, piece) to avoid collisions
            uint64_t mixed = bb * (0x9E3779B97F4A7C15ULL + uint64_t(c * 6 + pt) * 0x517CC1B727220A95ULL);
            h ^= mixed;
        }
    }

    // Mix in castling rights
    uint64_t cr = uint64_t(pos.castling_rights());
    h ^= cr * 0x6C62272E07BB0142ULL;

    // Mix in en passant square
    uint64_t ep = uint64_t(pos.ep_square());
    h ^= ep * 0xBF58476D1CE4E5B9ULL;

    // Mix in side to move
    if (pos.side_to_move() == BLACK) {
        h ^= 0x94D049BB133111EBULL;
    }

    // Final mix to reduce patterns
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;

    return h;
}

} // namespace neural
