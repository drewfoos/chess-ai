#include "mcts/node.h"
#include <cassert>
#include <limits>

namespace mcts {

Node::Node() : move_(Move::none()), prior_bits_(0), parent_(nullptr) {}

Node::Node(Move move, float prior) : move_(move), prior_bits_(float_to_half(prior)), parent_(nullptr) {}

void Node::add_child(Move move, float prior) {
    auto child = std::make_unique<Node>(move, prior);
    child->parent_ = this;
    children_.push_back(std::move(child));
}

void Node::update(float value) {
    visit_count_++;
    total_value_ += value;
    sum_sq_value_ += value * value;
}

Node* Node::select_child(float c_puct, float fpu_value) const {
    assert(!is_leaf());

    int parent_visits = visit_count_;
    float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));

    Node* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();

    for (const auto& child : children_) {
        float q = child->visit_count_ > 0 ? child->mean_value() : fpu_value;
        float u = c_puct * child->prior() * sqrt_parent / (1.0f + child->visit_count_);
        float score = q + u;

        if (score > best_score) {
            best_score = score;
            best = child.get();
        }
    }
    return best;
}

Move Node::best_move() const {
    assert(!is_leaf());

    const Node* best = nullptr;
    int best_visits = -1;

    for (const auto& child : children_) {
        if (child->visit_count_ > best_visits) {
            best_visits = child->visit_count_;
            best = child.get();
        }
    }
    return best ? best->move() : Move::none();
}

} // namespace mcts
