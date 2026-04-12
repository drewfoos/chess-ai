#include "mcts/node.h"
#include <cassert>
#include <limits>

namespace mcts {

Node::Node() : move_(Move::none()), prior_bits_(0), parent_(nullptr),
               edges_(nullptr), child_nodes_(nullptr), num_edges_(0), pool_managed_(false) {}

Node::~Node() {
    if (edges_) {
        if (!pool_managed_) {
            // Delete heap-allocated children
            for (int i = 0; i < num_edges_; i++) {
                delete child_nodes_[i];  // nullptr-safe
            }
        }
        delete[] edges_;
        delete[] child_nodes_;
    }
}

void Node::create_edges(const Move* moves, const float* priors, int count) {
    assert(num_edges_ == 0);  // Should only be called once
    num_edges_ = static_cast<uint16_t>(count);
    edges_ = new Edge[count];
    child_nodes_ = new Node*[count]();  // Zero-init (all nullptr)
    for (int i = 0; i < count; i++) {
        edges_[i].move_bits = moves[i].data;
        edges_[i].set_prior(priors[i]);
    }
}

void Node::sort_edges_by_prior() {
    if (num_edges_ <= 1) return;
    // Build index array, sort by prior descending, reorder both arrays
    std::vector<int> idx(num_edges_);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [this](int a, int b) {
        return half_to_float(edges_[a].prior_bits) > half_to_float(edges_[b].prior_bits);
    });
    std::vector<Edge> sorted_edges(num_edges_);
    std::vector<Node*> sorted_nodes(num_edges_);
    for (int i = 0; i < num_edges_; i++) {
        sorted_edges[i] = edges_[idx[i]];
        sorted_nodes[i] = child_nodes_[idx[i]];
    }
    std::copy(sorted_edges.begin(), sorted_edges.end(), edges_);
    std::copy(sorted_nodes.begin(), sorted_nodes.end(), child_nodes_);
}

Node* Node::ensure_child(int i, NodePool* pool) {
    assert(i >= 0 && i < num_edges_);
    if (!child_nodes_[i]) {
        Node* child;
        if (pool) {
            child = pool->allocate();
            child->pool_managed_ = true;
        } else {
            child = new Node();
        }
        child->move_ = edges_[i].move();
        child->prior_bits_ = edges_[i].prior_bits;
        child->parent_ = this;
        child_nodes_[i] = child;
    }
    return child_nodes_[i];
}

void Node::update(float value, int n) {
    visit_count_ += n;
    total_value_ += static_cast<float>(n) * value;
    sum_sq_value_ += static_cast<float>(n) * value * value;
}

Node* Node::select_child(float c_puct, float fpu_value) const {
    assert(!is_leaf());

    int parent_visits = visit_count_;
    float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));

    int best_idx = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_edges_; i++) {
        Node* child = child_nodes_[i];
        float q = (child && child->visit_count_ > 0) ? child->mean_value() : fpu_value;
        int child_visits = child ? child->visit_count_ : 0;
        float u = c_puct * edges_[i].prior() * sqrt_parent / (1.0f + child_visits);
        float score = q + u;

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    // Return existing child node or nullptr for unvisited edges
    return (best_idx >= 0) ? child_nodes_[best_idx] : nullptr;
}

Move Node::best_move() const {
    assert(!is_leaf());

    int best_idx = 0;
    int best_visits = -1;

    for (int i = 0; i < num_edges_; i++) {
        int visits = child_nodes_[i] ? child_nodes_[i]->visit_count_ : 0;
        if (visits > best_visits) {
            best_visits = visits;
            best_idx = i;
        }
    }
    return edges_[best_idx].move();
}

// --- NodePool ---

NodePool::NodePool(size_t initial_capacity) : next_free_(0), capacity_(initial_capacity) {
    nodes_.resize(initial_capacity);
}

Node* NodePool::allocate() {
    if (next_free_ >= capacity_) {
        // Grow by doubling
        size_t new_cap = capacity_ * 2;
        nodes_.resize(new_cap);
        capacity_ = new_cap;
    }
    Node* node = &nodes_[next_free_++];
    // Reset node to clean state
    node->reset();
    return node;
}

void NodePool::reset() {
    // Just reset the free pointer — nodes will be reset when allocated
    next_free_ = 0;
}

} // namespace mcts
