#pragma once
#include "core/types.h"
#include <vector>
#include <memory>
#include <cmath>

namespace mcts {

class Node {
public:
    Node();
    Node(Move move, float prior);

    // Tree structure
    bool is_leaf() const { return children_.empty(); }
    int num_children() const { return static_cast<int>(children_.size()); }
    Node* child(int i) { return children_[i].get(); }
    const Node* child(int i) const { return children_[i].get(); }
    Node* parent() const { return parent_; }

    // Statistics
    int visit_count() const { return visit_count_; }
    float total_value() const { return total_value_; }
    float prior() const { return prior_; }
    Move move() const { return move_; }

    float mean_value() const {
        return visit_count_ > 0 ? total_value_ / visit_count_ : 0.0f;
    }

    // Modification
    void add_child(Move move, float prior);
    void update(float value);

    // Selection
    Node* select_child(float c_puct, float fpu_value) const;
    Move best_move() const;

    // For Dirichlet noise
    void set_prior(float p) { prior_ = p; }

private:
    Move move_;
    float prior_ = 0.0f;

    int visit_count_ = 0;
    float total_value_ = 0.0f;

    Node* parent_ = nullptr;
    std::vector<std::unique_ptr<Node>> children_;
};

} // namespace mcts
