#pragma once
#include "core/position.h"
#include <vector>
#include <cstdint>

namespace neural {

class PositionHistory {
public:
    // Clear history and set initial position
    void reset(const Position& pos);

    // Add position after a move was made
    void push(const Position& pos);

    // Return the current (most recent) position
    const Position& current() const;

    // Return position at steps_back (0=current, 1=one move ago, etc.)
    // Clamps to the first position if out of bounds.
    const Position& at(int steps_back) const;

    int length() const { return static_cast<int>(positions_.size()); }

    // Check if current position has been seen 'count' times in the history
    bool is_repetition(int count = 2) const;

    // Compute a hash from position state (piece placement + castling + EP + STM)
    static uint64_t compute_hash(const Position& pos);

    // Access to hash history for repetition detection during search
    const std::vector<uint64_t>& hashes() const { return hashes_; }

private:
    std::vector<Position> positions_;
    std::vector<uint64_t> hashes_;
};

} // namespace neural
