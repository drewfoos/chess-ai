#pragma once
#include "core/types.h"
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <deque>
#include <numeric>

namespace mcts {

class NodePool;  // Forward declaration

static_assert(sizeof(uint16_t) == 2, "uint16_t must be 2 bytes");

struct Edge {
    uint16_t move_bits = 0;   // Raw Move data
    uint16_t prior_bits = 0;  // FP16 prior

    Move move() const { Move m; m.data = move_bits; return m; }
    float prior() const;      // Defined after Node (uses half_to_float)
    void set_prior(float p);  // Defined after Node (uses float_to_half)
};
static_assert(sizeof(Edge) == 4, "Edge must be 4 bytes for memory efficiency");

class Node {
public:
    Node();
    ~Node();

    // Tree structure
    bool is_leaf() const { return num_edges_ == 0; }
    int num_children() const { return num_edges_; }
    int num_edges() const { return num_edges_; }
    Edge& edge(int i) { assert(i >= 0 && i < num_edges_); return edges_[i]; }
    const Edge& edge(int i) const { assert(i >= 0 && i < num_edges_); return edges_[i]; }
    Node* child_node(int i) const { assert(i >= 0 && i < num_edges_); return child_nodes_[i]; }
    bool has_child_node(int i) const { assert(i >= 0 && i < num_edges_); return child_nodes_[i] != nullptr; }
    Node* ensure_child(int i, NodePool* pool = nullptr);
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

    // Virtual loss for batched MCTS. n > 1 supports multivisit collapse
    // (claim N visits in a single descent when PUCT would pick the same child N times).
    void apply_virtual_loss(int n = 1) { visit_count_ += n; pending_evals_ += static_cast<int16_t>(n); }
    void revert_virtual_loss(int n = 1) { visit_count_ -= n; pending_evals_ -= static_cast<int16_t>(n); }
    int pending_evals() const { return pending_evals_; }

    // Terminal status for MCTS-solver
    int8_t terminal_status() const { return terminal_status_; }
    void set_terminal_status(int8_t s) { terminal_status_ = s; }

    // Moves-left head output (network's estimate of remaining plies from this position).
    // 0 until the node is expanded via the evaluator. Used as a PUCT bonus term.
    float mlh() const { return mlh_; }
    void set_mlh(float m) { mlh_ = m; }

    // Edge/child management
    void create_edges(const Move* moves, const float* priors, int count);
    void sort_edges_by_prior();

    // Modification. n > 1 applies N visits at once (multivisit backprop).
    void update(float value, int n = 1);

    // For backward compat — sets the node's own prior
    void set_prior(float p) { prior_bits_ = float_to_half(p); }

    // Pool management
    bool pool_managed() const { return pool_managed_; }
    void set_pool_managed(bool v) { pool_managed_ = v; }

    // Selection — NOTE: may return nullptr for unvisited edges.
    // Real search code uses select_child_advanced() + ensure_child() instead.
    Node* select_child(float c_puct, float fpu_value) const;
    Move best_move() const;

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
    friend class NodePool;  // NodePool needs access to reset()

    // Reset node to initial state — only for pool reuse.
    // Frees edge/child_nodes arrays but does NOT delete child Node objects
    // (assumes pool owns them). Do not call on heap-allocated nodes with
    // heap-allocated children — use the destructor instead.
    void reset() {
        move_ = Move::none();
        prior_bits_ = 0;
        visit_count_ = 0;
        total_value_ = 0.0f;
        sum_sq_value_ = 0.0f;
        terminal_status_ = 0;
        pending_evals_ = 0;
        mlh_ = 0.0f;
        parent_ = nullptr;
        if (edges_) {
            delete[] edges_;
            edges_ = nullptr;
        }
        if (child_nodes_) {
            delete[] child_nodes_;
            child_nodes_ = nullptr;
        }
        num_edges_ = 0;
        pool_managed_ = false;
    }

    Move move_;
    uint16_t prior_bits_ = 0;

    int visit_count_ = 0;
    float total_value_ = 0.0f;
    float sum_sq_value_ = 0.0f;

    int8_t terminal_status_ = 0;
    int16_t pending_evals_ = 0;
    float mlh_ = 0.0f;

    Node* parent_ = nullptr;

    Edge* edges_ = nullptr;
    Node** child_nodes_ = nullptr;
    uint16_t num_edges_ = 0;
    bool pool_managed_ = false;
};

// Edge inline methods that depend on Node
inline float Edge::prior() const { return Node::half_to_float(prior_bits); }
inline void Edge::set_prior(float p) { prior_bits = Node::float_to_half(p); }

// Arena allocator for MCTS nodes
class NodePool {
public:
    explicit NodePool(size_t initial_capacity = 65536);
    Node* allocate();      // Returns zeroed Node
    void reset();          // Reuse pool for next search
    size_t used() const { return next_free_; }
private:
    std::deque<Node> nodes_;
    size_t next_free_ = 0;
    size_t capacity_ = 0;
};

} // namespace mcts
