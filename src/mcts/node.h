#pragma once
#include "core/types.h"
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>

namespace mcts {

class Node {
public:
    Node();
    Node(Move move, float prior);

    // Tree structure
    bool is_leaf() const { return children_.empty(); }
    int num_children() const { return static_cast<int>(children_.size()); }
    Node* child(int i) { assert(i >= 0 && i < num_children()); return children_[i].get(); }
    const Node* child(int i) const { assert(i >= 0 && i < num_children()); return children_[i].get(); }
    Node* parent() const { return parent_; }

    // Statistics
    int visit_count() const { return visit_count_; }
    float total_value() const { return total_value_; }
    float prior() const { return half_to_float(prior_bits_); }
    Move move() const { return move_; }

    float mean_value() const {
        return visit_count_ > 0 ? total_value_ / visit_count_ : 0.0f;
    }

    // Value variance for uncertainty estimation
    float value_variance() const {
        if (visit_count_ < 2) return 0.0f;
        float mean = total_value_ / visit_count_;
        float mean_sq = sum_sq_value_ / visit_count_;
        return std::max(0.0f, mean_sq - mean * mean);
    }

    // Virtual loss for batched MCTS
    void apply_virtual_loss() { visit_count_++; pending_evals_++; }
    void revert_virtual_loss() { visit_count_--; pending_evals_--; }
    int pending_evals() const { return pending_evals_; }

    // Terminal status for MCTS-solver
    int8_t terminal_status() const { return terminal_status_; }
    void set_terminal_status(int8_t s) { terminal_status_ = s; }

    // Modification
    void add_child(Move move, float prior);
    void update(float value);

    // Selection
    Node* select_child(float c_puct, float fpu_value) const;
    Move best_move() const;

    // For Dirichlet noise
    void set_prior(float p) { prior_bits_ = float_to_half(p); }

    // Sort children by prior descending (for cache locality)
    void sort_children_by_prior() {
        std::sort(children_.begin(), children_.end(),
            [](const std::unique_ptr<Node>& a, const std::unique_ptr<Node>& b) {
                return a->prior() > b->prior();
            });
    }

    // Float16 conversion helpers (public for testing)
    static uint16_t float_to_half(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        uint16_t sign = (bits >> 16) & 0x8000;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFF;
        if (exponent <= 0) return sign;  // underflow to zero
        if (exponent >= 31) return sign | 0x7C00;  // overflow to inf
        return sign | (exponent << 10) | (mantissa >> 13);
    }

    static float half_to_float(uint16_t half) {
        uint32_t sign = (half & 0x8000) << 16;
        uint32_t exponent = (half >> 10) & 0x1F;
        uint32_t mantissa = half & 0x3FF;
        if (exponent == 0) {
            if (mantissa == 0) { float f; uint32_t bits = sign; std::memcpy(&f, &bits, sizeof(f)); return f; }
            // Denormalized
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            exponent++; mantissa &= ~0x400;
        } else if (exponent == 31) {
            uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
            float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        }
        uint32_t bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    }

private:
    Move move_;
    uint16_t prior_bits_ = 0;

    int visit_count_ = 0;
    float total_value_ = 0.0f;
    float sum_sq_value_ = 0.0f;

    int8_t terminal_status_ = 0;
    int16_t pending_evals_ = 0;

    Node* parent_ = nullptr;
    std::vector<std::unique_ptr<Node>> children_;
};

} // namespace mcts
