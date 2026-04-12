#include "mcts/search.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace mcts {

// --- RandomEvaluator ---

EvalResult RandomEvaluator::evaluate(const Position& pos, const Move* moves, int num_moves) {
    EvalResult result;

    // Terminal position
    if (num_moves == 0) {
        if (pos.in_check()) {
            result.value = -1.0f; // Checkmated — loss for side to move
        } else {
            result.value = 0.0f;  // Stalemate — draw
        }
        return result;
    }

    // Uniform policy over legal moves
    float uniform = 1.0f / num_moves;
    result.policy.assign(num_moves, uniform);

    // Simple material-based evaluation
    Color us = pos.side_to_move();
    Color them = ~us;

    static constexpr float piece_values[] = {
        1.0f,   // PAWN
        3.0f,   // KNIGHT
        3.0f,   // BISHOP
        5.0f,   // ROOK
        9.0f,   // QUEEN
        0.0f    // KING
    };

    float material = 0.0f;
    for (int pt = PAWN; pt <= QUEEN; pt++) {
        material += piece_values[pt] * popcount(pos.pieces(us, PieceType(pt)));
        material -= piece_values[pt] * popcount(pos.pieces(them, PieceType(pt)));
    }

    // Squash material advantage into [-1, 1] using tanh-like scaling
    result.value = material / (std::abs(material) + 3.0f);

    return result;
}

// --- Search ---

Search::Search(Evaluator& evaluator, const SearchParams& params)
    : evaluator_(evaluator), params_(params), cache_(params.nn_cache_size) {}

void Search::set_stop_flag(std::atomic<bool>* stop) { stop_flag_ = stop; }
void Search::set_info_callback(InfoCallback cb) { info_callback_ = std::move(cb); }

float Search::dynamic_cpuct(int parent_visits) const {
    return params_.c_puct_init + params_.c_puct_factor * std::log(
        (parent_visits + params_.c_puct_base) / params_.c_puct_base
    );
}

void Search::replay_moves(const Position& root_pos, const std::vector<Move>& moves, Position& out_pos) {
    out_pos = root_pos;
    UndoInfo undo;
    for (Move m : moves) {
        out_pos.make_move(m, undo);
    }
}

void Search::backpropagate(Node* node, float value) {
    while (node != nullptr) {
        node->update(value);
        value = -value;
        node = node->parent();
    }
}

void Search::propagate_terminal(Node* node) {
    // Walk up from node, trying to resolve terminal status
    Node* cur = node->parent();
    while (cur != nullptr) {
        if (cur->terminal_status() != 0 || cur->is_leaf()) {
            cur = cur->parent();
            continue;
        }

        bool all_resolved = true;
        bool any_draw = false;
        bool found_child_loss = false; // child loses = we win

        for (int i = 0; i < cur->num_children(); i++) {
            int8_t cs = cur->child(i)->terminal_status();
            if (cs == 0) {
                all_resolved = false;
                break;
            }
            if (cs == 1) {
                // Child loses (from child's perspective) = we win
                found_child_loss = true;
            }
            if (cs == 2) {
                any_draw = true;
            }
        }

        if (found_child_loss) {
            // At least one child is a loss for the opponent = win for us
            cur->set_terminal_status(-1);
        } else if (all_resolved) {
            if (any_draw) {
                cur->set_terminal_status(2); // Best we can get is a draw
            } else {
                cur->set_terminal_status(1); // All children win for opponent = we lose
            }
        } else {
            break; // Can't resolve further up
        }

        cur = cur->parent();
    }
}

Node* Search::select_child_advanced(Node* node, bool is_root) {
    assert(!node->is_leaf());

    float fpu_red = is_root ? params_.fpu_reduction_root : params_.fpu_reduction;
    float fpu_value = node->mean_value() - fpu_red;
    float c_puct = dynamic_cpuct(node->visit_count());

    // Variance-scaled cPUCT
    if (params_.variance_scaling && node->visit_count() > 1) {
        float var = node->value_variance();
        float scale = std::sqrt(var) / 0.5f;
        scale = std::max(0.5f, std::min(2.0f, scale));
        c_puct *= scale;
    }

    float sqrt_parent = std::sqrt(static_cast<float>(node->visit_count()));

    // Sibling blending: compute average value of visited siblings for FPU
    float sibling_fpu = fpu_value;
    if (params_.sibling_blending) {
        float visited_sum = 0.0f;
        int visited_count = 0;
        for (int i = 0; i < node->num_children(); i++) {
            Node* child = node->child(i);
            if (child->visit_count() > 0) {
                visited_sum += child->mean_value();
                visited_count++;
            }
        }
        if (visited_count > 0) {
            sibling_fpu = visited_sum / visited_count;
            // Blend with FPU: still apply some reduction
            sibling_fpu = std::min(sibling_fpu, fpu_value + fpu_red * 0.5f);
        }
    }

    Node* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < node->num_children(); i++) {
        Node* child = node->child(i);

        // MCTS-solver: child terminal_status == 1 means child loses (from child's perspective)
        // which means WE win by going here. Pick immediately.
        if (child->terminal_status() == 1) {
            return child;
        }
        // Child wins (from child's perspective) = we lose. Skip.
        if (child->terminal_status() == -1) {
            continue;
        }

        float score;
        if (child->visit_count() == 0) {
            // Use sibling blending FPU for unvisited children
            float child_fpu = sibling_fpu;

            // Per-child sibling blending: use average of visited siblings with similar priors
            if (params_.sibling_blending) {
                float child_prior = child->prior();
                float sim_sum = 0.0f;
                int sim_count = 0;
                for (int j = 0; j < node->num_children(); j++) {
                    Node* sib = node->child(j);
                    if (sib->visit_count() > 0 && std::abs(sib->prior() - child_prior) < 0.10f) {
                        sim_sum += sib->mean_value();
                        sim_count++;
                    }
                }
                if (sim_count > 0) {
                    child_fpu = sim_sum / sim_count;
                }
            }

            score = child_fpu + c_puct * child->prior() * sqrt_parent / 1.0f;
        } else {
            // Q + U
            float q = child->mean_value();
            float u = c_puct * child->prior() * sqrt_parent / (1.0f + child->visit_count());
            score = q + u;

            // Uncertainty boosting
            if (params_.uncertainty_weight > 0.0f && child->visit_count() > 1) {
                score += params_.uncertainty_weight * std::sqrt(child->value_variance());
            }
        }

        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }

    // If all children are proven losses (terminal_status == -1), return first child
    if (best == nullptr) {
        best = node->child(0);
    }

    return best;
}

bool Search::is_two_fold_repetition(uint64_t hash, const std::vector<uint64_t>& path_hashes,
                                     const neural::PositionHistory& history) const {
    // Check against selection path hashes
    for (uint64_t h : path_hashes) {
        if (h == hash) return true;
    }
    // Check against game history
    const auto& hist_hashes = history.hashes();
    for (uint64_t h : hist_hashes) {
        if (h == hash) return true;
    }
    return false;
}

void Search::expand_node(Node* node, const Position& pos, const EvalResult& eval_result) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 0) {
        // Terminal node
        if (pos.in_check()) {
            node->set_terminal_status(1); // This node is a loss for the side to move
        } else {
            node->set_terminal_status(2); // Draw (stalemate)
        }
        return;
    }

    for (int i = 0; i < num_moves; i++) {
        float prior = (i < static_cast<int>(eval_result.policy.size())) ? eval_result.policy[i] : 0.0f;
        node->add_child(moves[i], prior);
    }
    node->sort_children_by_prior();
}

// Helper: revert virtual loss on all nodes in the path
static void revert_virtual_loss_path(const std::vector<Node*>& path_nodes) {
    for (Node* n : path_nodes) {
        n->revert_virtual_loss();
    }
}

void Search::gather_leaf(Node* root, const neural::PositionHistory& history,
                          std::vector<PendingEval>& batch) {
    // Select from root to a leaf, applying virtual loss
    Node* node = root;
    std::vector<Move> path_moves;
    std::vector<uint64_t> path_hashes;
    std::vector<Node*> path_nodes; // Track all nodes with virtual loss

    // Add root position hash
    path_hashes.push_back(neural::PositionHistory::compute_hash(history.current()));

    while (!node->is_leaf()) {
        bool is_root = (node->parent() == nullptr);
        Node* child = select_child_advanced(node, is_root);
        path_moves.push_back(child->move());

        // Apply virtual loss
        child->apply_virtual_loss();
        path_nodes.push_back(child);

        // Compute hash for two-fold detection
        if (params_.two_fold_draw) {
            Position child_pos;
            replay_moves(history.current(), path_moves, child_pos);
            uint64_t child_hash = neural::PositionHistory::compute_hash(child_pos);

            if (is_two_fold_repetition(child_hash, path_hashes, history)) {
                // Treat as draw: revert all virtual losses, backprop 0.0, set terminal
                revert_virtual_loss_path(path_nodes);
                child->set_terminal_status(2);
                backpropagate(child, 0.0f);
                propagate_terminal(child);
                return;
            }

            path_hashes.push_back(child_hash);
        }

        node = child;
    }

    // node is now a leaf
    // Build position at the leaf
    Position leaf_pos;
    replay_moves(history.current(), path_moves, leaf_pos);

    // Check if terminal (already visited leaf with no children = terminal)
    if (node->visit_count() > 0 && node->terminal_status() != 0) {
        float value = 0.0f;
        if (node->terminal_status() == 1) value = 1.0f;
        else if (node->terminal_status() == -1) value = -1.0f;

        revert_virtual_loss_path(path_nodes);
        backpropagate(node, -value);
        return;
    }

    // Check NN cache
    uint64_t pos_hash = neural::PositionHistory::compute_hash(leaf_pos);
    const CacheEntry* cached = cache_.get(pos_hash);
    if (cached) {
        // Expand from cache
        EvalResult eval_result;
        eval_result.policy = cached->policy;
        eval_result.value = cached->value;

        expand_node(node, leaf_pos, eval_result);

        // If terminal after expansion (no children added)
        if (node->is_leaf()) {
            float value = 0.0f;
            if (node->terminal_status() == 1) value = 1.0f;
            else if (node->terminal_status() == -1) value = -1.0f;
            revert_virtual_loss_path(path_nodes);
            backpropagate(node, -value);
            propagate_terminal(node);
            return;
        }

        // Revert virtual loss and backprop
        revert_virtual_loss_path(path_nodes);
        backpropagate(node, -eval_result.value);
        return;
    }

    // Queue for evaluation
    PendingEval pe;
    pe.leaf = node;
    pe.position = leaf_pos;
    pe.path_moves = std::move(path_moves);
    pe.path_hashes = std::move(path_hashes);
    pe.path_nodes = std::move(path_nodes);
    batch.push_back(std::move(pe));
}

void Search::add_dirichlet_noise(Node* root) {
    if (root->is_leaf()) return;

    if (params_.shaped_dirichlet) {
        add_shaped_dirichlet_noise(root);
        return;
    }

    int num_children = root->num_children();
    std::vector<float> noise(num_children);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(params_.dirichlet_alpha, 1.0f);

    float noise_sum = 0.0f;
    for (int i = 0; i < num_children; i++) {
        noise[i] = gamma(gen);
        noise_sum += noise[i];
    }
    if (noise_sum > 0.0f) {
        for (int i = 0; i < num_children; i++) {
            noise[i] /= noise_sum;
        }
    }

    float eps = params_.dirichlet_epsilon;
    for (int i = 0; i < num_children; i++) {
        float old_prior = root->child(i)->prior();
        float new_prior = (1.0f - eps) * old_prior + eps * noise[i];
        root->child(i)->set_prior(new_prior);
    }
}

void Search::add_shaped_dirichlet_noise(Node* root) {
    if (root->is_leaf()) return;

    int num_children = root->num_children();

    // Collect priors and compute log-space thresholds
    std::vector<float> priors(num_children);
    std::vector<float> log_priors(num_children);
    float max_log = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_children; i++) {
        priors[i] = root->child(i)->prior();
        log_priors[i] = std::log(priors[i] + 1e-8f);
        max_log = std::max(max_log, log_priors[i]);
    }

    float threshold = max_log - 2.0f;

    // Compute per-move weights based on prior strength
    std::vector<float> weights(num_children, 0.5f);
    float weight_sum = 0.0f;
    for (int i = 0; i < num_children; i++) {
        if (log_priors[i] > threshold) {
            weights[i] += 0.5f * (log_priors[i] - threshold) / 2.0f;
        }
        weight_sum += weights[i];
    }
    if (weight_sum > 0.0f) {
        for (int i = 0; i < num_children; i++) {
            weights[i] /= weight_sum;
        }
    }

    // Scale alpha per-move
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<float> noise(num_children);
    float noise_sum = 0.0f;
    for (int i = 0; i < num_children; i++) {
        float scaled_alpha = std::max(0.01f, params_.dirichlet_alpha * weights[i] * num_children);
        std::gamma_distribution<float> gamma(scaled_alpha, 1.0f);
        noise[i] = gamma(gen);
        noise_sum += noise[i];
    }
    if (noise_sum > 0.0f) {
        for (int i = 0; i < num_children; i++) {
            noise[i] /= noise_sum;
        }
    }

    float eps = params_.dirichlet_epsilon;
    for (int i = 0; i < num_children; i++) {
        float old_prior = root->child(i)->prior();
        float new_prior = (1.0f - eps) * old_prior + eps * noise[i];
        root->child(i)->set_prior(new_prior);
    }
}

SearchResult Search::run(const Position& pos) {
    neural::PositionHistory history;
    history.reset(pos);
    return run(history);
}

SearchResult Search::run(const neural::PositionHistory& history) {
    const Position& root_pos = history.current();
    auto root = std::make_unique<Node>();

    // Generate legal moves and evaluate root
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(root_pos, moves);

    if (num_moves == 0) {
        // Terminal position
        SearchResult result;
        result.best_move = Move::none();
        if (root_pos.in_check()) {
            result.root_value = -1.0f; // Checkmate
        } else {
            result.root_value = 0.0f;  // Stalemate
        }
        result.total_nodes = 1;
        result.raw_value = result.root_value;
        return result;
    }

    // Evaluate root position
    EvalResult root_eval = evaluator_.evaluate(root_pos, moves, num_moves);
    float raw_value = root_eval.value;

    // Build raw_policy in 1858-dim space
    std::array<float, neural::POLICY_SIZE> raw_policy = {};
    Color stm = root_pos.side_to_move();
    for (int i = 0; i < num_moves; i++) {
        int idx = neural::move_to_policy_index(moves[i], stm);
        if (idx >= 0 && idx < neural::POLICY_SIZE) {
            raw_policy[idx] = root_eval.policy[i];
        }
    }

    // Expand root
    for (int i = 0; i < num_moves; i++) {
        root->add_child(moves[i], root_eval.policy[i]);
    }
    root->sort_children_by_prior();

    // Initial root visit
    root->update(root_eval.value);

    // Cache root evaluation
    uint64_t root_hash = neural::PositionHistory::compute_hash(root_pos);
    CacheEntry root_entry;
    root_entry.policy = root_eval.policy;
    root_entry.value = root_eval.value;
    root_entry.num_moves = num_moves;
    cache_.put(root_hash, std::move(root_entry));

    // Add Dirichlet noise at root for exploration
    if (params_.add_noise) {
        add_dirichlet_noise(root.get());
    }

    // Main batched MCTS loop
    int sims_done = 0;
    int batch_size = std::max(1, params_.batch_size);

    while (sims_done < params_.num_iterations) {
        if (stop_flag_ && stop_flag_->load(std::memory_order_relaxed)) {
            break;
        }

        // Smart pruning check
        if (params_.smart_pruning && root->num_children() >= 2 && sims_done > batch_size) {
            int remaining = params_.num_iterations - sims_done;
            // Find best and second best visit counts
            int best_visits = 0, second_visits = 0;
            for (int i = 0; i < root->num_children(); i++) {
                int vc = root->child(i)->visit_count();
                if (vc > best_visits) {
                    second_visits = best_visits;
                    best_visits = vc;
                } else if (vc > second_visits) {
                    second_visits = vc;
                }
            }
            if (second_visits + remaining < static_cast<int>(best_visits * params_.smart_pruning_factor)) {
                break;
            }
        }

        // Gather phase: collect leaves for batch evaluation
        std::vector<PendingEval> batch;
        batch.reserve(batch_size);

        int gather_count = std::min(batch_size, params_.num_iterations - sims_done);
        for (int i = 0; i < gather_count; i++) {
            gather_leaf(root.get(), history, batch);
        }

        // Evaluate phase: build batch requests and evaluate all at once
        // Generate legal moves for each pending leaf
        std::vector<int> leaf_num_moves(batch.size());
        std::vector<BatchEvalRequest> eval_requests;
        eval_requests.reserve(batch.size());

        for (size_t i = 0; i < batch.size(); i++) {
            BatchEvalRequest req;
            req.position = batch[i].position;
            req.num_legal_moves = generate_legal_moves(batch[i].position, req.legal_moves);
            leaf_num_moves[i] = req.num_legal_moves;
            eval_requests.push_back(std::move(req));
        }

        // Single batched evaluation (one GPU forward pass for NeuralEvaluator)
        auto eval_results = evaluator_.evaluate_batch(eval_requests);

        // Scatter results: expand nodes, backpropagate
        for (size_t i = 0; i < batch.size(); i++) {
            auto& pe = batch[i];
            auto& eval_result = eval_results[i];

            expand_node(pe.leaf, pe.position, eval_result);

            if (pe.leaf->is_leaf()) {
                float value = 0.0f;
                if (pe.leaf->terminal_status() == 1) value = 1.0f;
                else if (pe.leaf->terminal_status() == -1) value = -1.0f;
                revert_virtual_loss_path(pe.path_nodes);
                backpropagate(pe.leaf, -value);
                propagate_terminal(pe.leaf);
            } else {
                revert_virtual_loss_path(pe.path_nodes);
                backpropagate(pe.leaf, -eval_result.value);

                uint64_t pos_hash = neural::PositionHistory::compute_hash(pe.position);
                CacheEntry entry;
                entry.policy = eval_result.policy;
                entry.value = eval_result.value;
                entry.num_moves = leaf_num_moves[i];
                cache_.put(pos_hash, std::move(entry));

                propagate_terminal(pe.leaf);
            }
        }

        sims_done += gather_count;

        if (info_callback_ && root->num_children() > 0) {
            SearchInfo sinfo;
            sinfo.iterations = sims_done;
            sinfo.total_nodes = root->visit_count();
            sinfo.root_value = root->mean_value();
            sinfo.best_move = root->best_move();
            info_callback_(sinfo);
        }

        // Check if root is resolved
        if (root->terminal_status() != 0) break;
    }

    return build_result(root.get(), root_pos, raw_value, raw_policy);
}

SearchResult Search::build_result(Node* root, const Position& root_pos,
                                   float raw_value, const std::array<float, neural::POLICY_SIZE>& raw_policy) {
    SearchResult result;
    result.best_move = root->is_leaf() ? Move::none() : root->best_move();
    result.root_value = root->mean_value();
    result.raw_value = raw_value;
    result.raw_policy = raw_policy;

    // Apply contempt
    if (params_.contempt > 0.0f && std::abs(result.root_value) < 1.0f) {
        float sign = result.root_value >= 0.0f ? 1.0f : -1.0f;
        float shift = params_.contempt * (1.0f - std::abs(result.root_value));
        result.root_value = std::max(-1.0f, std::min(1.0f, result.root_value + sign * shift));
    }

    int num_children = root->num_children();
    result.moves.resize(num_children);
    result.visit_counts.resize(num_children);

    int total_child_visits = 0;
    Color stm = root_pos.side_to_move();

    for (int i = 0; i < num_children; i++) {
        result.moves[i] = root->child(i)->move();
        result.visit_counts[i] = root->child(i)->visit_count();
        total_child_visits += result.visit_counts[i];
    }

    // Build policy_target: normalized visit distribution in 1858-dim space
    result.policy_target.fill(0.0f);
    if (total_child_visits > 0) {
        for (int i = 0; i < num_children; i++) {
            if (result.visit_counts[i] > 0) {
                int idx = neural::move_to_policy_index(result.moves[i], stm);
                if (idx >= 0 && idx < neural::POLICY_SIZE) {
                    result.policy_target[idx] = static_cast<float>(result.visit_counts[i]) / total_child_visits;
                }
            }
        }
    }

    result.total_nodes = root->visit_count();

    return result;
}

Move Search::select_move_with_temperature(const SearchResult& result, float temperature) {
    if (result.moves.empty()) return Move::none();

    // Temperature ≈ 0: greedy selection
    if (temperature < 0.01f) {
        int best_idx = 0;
        for (int i = 1; i < static_cast<int>(result.visit_counts.size()); i++) {
            if (result.visit_counts[i] > result.visit_counts[best_idx]) {
                best_idx = i;
            }
        }
        return result.moves[best_idx];
    }

    // Temperature-based sampling: π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)
    int num_moves = static_cast<int>(result.moves.size());
    std::vector<float> probs(num_moves);
    float inv_temp = 1.0f / temperature;

    float sum = 0.0f;
    for (int i = 0; i < num_moves; i++) {
        probs[i] = std::pow(static_cast<float>(result.visit_counts[i]), inv_temp);
        sum += probs[i];
    }

    if (sum <= 0.0f) {
        // Fallback: uniform random
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, num_moves - 1);
        return result.moves[dist(gen)];
    }

    for (int i = 0; i < num_moves; i++) {
        probs[i] /= sum;
    }

    // Sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return result.moves[dist(gen)];
}

} // namespace mcts
