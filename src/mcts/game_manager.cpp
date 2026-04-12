#ifdef HAS_LIBTORCH

#include "mcts/game_manager.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstring>

namespace mcts {

// Helper: revert virtual loss on all nodes in a path
static void revert_virtual_loss_path(const std::vector<Node*>& path_nodes) {
    for (Node* n : path_nodes) {
        n->revert_virtual_loss();
    }
}

// Helper: parse a UCI move string against a position
static Move parse_uci_move_internal(const Position& pos, const std::string& uci) {
    if (uci.size() < 4 || uci.size() > 5) {
        throw std::runtime_error("Invalid UCI move format: " + uci);
    }

    int from = (uci[0] - 'a') + (uci[1] - '1') * 8;
    int to   = (uci[2] - 'a') + (uci[3] - '1') * 8;

    PieceType promo = NO_PIECE_TYPE;
    if (uci.size() == 5) {
        switch (uci[4]) {
            case 'q': promo = QUEEN;  break;
            case 'r': promo = ROOK;   break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
            default:
                throw std::runtime_error("Invalid promotion piece in UCI move: " + uci);
        }
    }

    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    for (int i = 0; i < n; i++) {
        if (moves[i].from() == Square(from) && moves[i].to() == Square(to)) {
            if (promo == NO_PIECE_TYPE || moves[i].promo_piece() == promo) {
                return moves[i];
            }
        }
    }
    throw std::runtime_error("Illegal move: " + uci);
}

GameManager::GameManager(neural::NeuralEvaluator& evaluator, const SearchParams& params)
    : evaluator_(evaluator), params_(params), cache_(params.nn_cache_size) {}

void GameManager::init_game(int idx, const neural::PositionHistory& history, int num_sims) {
    if (idx < 0 || idx >= static_cast<int>(games_.size())) {
        throw std::runtime_error("GameManager::init_game: invalid index");
    }
    auto& game = games_[idx];
    game.history = history;
    game.root = std::make_unique<Node>();
    game.sims_done = 0;
    game.target_sims = num_sims;
    game.search_complete = false;
    game.raw_value = 0.0f;
    game.raw_policy.fill(0.0f);
    game.root_expanded = false;
}

void GameManager::init_games(int num_games, int num_sims) {
    games_.resize(num_games);
    for (int i = 0; i < num_games; i++) {
        Position start_pos;
        start_pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        neural::PositionHistory history;
        history.reset(start_pos);

        games_[i].history = history;
        games_[i].root = std::make_unique<Node>();
        games_[i].sims_done = 0;
        games_[i].target_sims = num_sims;
        games_[i].search_complete = false;
        games_[i].raw_value = 0.0f;
        games_[i].raw_policy.fill(0.0f);
        games_[i].root_expanded = false;
    }
}

void GameManager::init_games_from_fen(const std::vector<std::string>& fens,
                                       const std::vector<std::vector<std::string>>& move_histories,
                                       int num_sims) {
    int num_games = static_cast<int>(fens.size());
    games_.resize(num_games);

    for (int i = 0; i < num_games; i++) {
        Position pos;
        pos.set_fen(fens[i]);
        neural::PositionHistory history;
        history.reset(pos);

        // Replay move history if provided
        if (i < static_cast<int>(move_histories.size())) {
            for (const auto& uci_str : move_histories[i]) {
                Move m = parse_uci_move_internal(pos, uci_str);
                UndoInfo undo;
                pos.make_move(m, undo);
                history.push(pos);
            }
        }

        games_[i].history = history;
        games_[i].root = std::make_unique<Node>();
        games_[i].sims_done = 0;
        games_[i].target_sims = num_sims;
        games_[i].search_complete = false;
        games_[i].raw_value = 0.0f;
        games_[i].raw_policy.fill(0.0f);
        games_[i].root_expanded = false;
    }
}

void GameManager::expand_root(GameState& game) {
    const Position& root_pos = game.history.current();
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(root_pos, moves);

    if (num_moves == 0) {
        // Terminal root
        game.search_complete = true;
        if (root_pos.in_check()) {
            game.raw_value = -1.0f;
        } else {
            game.raw_value = 0.0f;
        }
        game.root_expanded = true;
        return;
    }

    // Encode and evaluate root
    std::vector<float> encode_buf(neural::TENSOR_SIZE);
    neural::encode_position(game.history, encode_buf.data());

    neural::BatchRequest req;
    req.encoded_planes = encode_buf.data();
    req.legal_moves = moves;
    req.num_legal_moves = num_moves;

    auto results = evaluator_.evaluate_batch({req});
    const auto& br = results[0];

    game.raw_value = br.value;

    // Build raw_policy in 1858-dim space
    Color stm = root_pos.side_to_move();
    game.raw_policy.fill(0.0f);
    for (int i = 0; i < num_moves; i++) {
        int idx = neural::move_to_policy_index(moves[i], stm);
        if (idx >= 0 && idx < neural::POLICY_SIZE) {
            game.raw_policy[idx] = br.policy[i];
        }
    }

    // Expand root node
    for (int i = 0; i < num_moves; i++) {
        float prior = (i < static_cast<int>(br.policy.size())) ? br.policy[i] : 0.0f;
        game.root->add_child(moves[i], prior);
    }
    game.root->sort_children_by_prior();
    game.root->update(br.value);

    // Cache root evaluation
    uint64_t root_hash = neural::PositionHistory::compute_hash(root_pos);
    CacheEntry entry;
    entry.policy = br.policy;
    entry.value = br.value;
    entry.num_moves = num_moves;
    cache_.put(root_hash, std::move(entry));

    // Add Dirichlet noise
    if (params_.add_noise) {
        add_dirichlet_noise(game.root.get());
    }

    game.root_expanded = true;
}

float GameManager::dynamic_cpuct(int parent_visits) const {
    return params_.c_puct_init + params_.c_puct_factor * std::log(
        (parent_visits + params_.c_puct_base) / params_.c_puct_base
    );
}

void GameManager::replay_moves(const Position& root_pos, const std::vector<Move>& moves, Position& out_pos) {
    out_pos = root_pos;
    UndoInfo undo;
    for (Move m : moves) {
        out_pos.make_move(m, undo);
    }
}

void GameManager::backpropagate(Node* node, float value) {
    while (node != nullptr) {
        node->update(value);
        value = -value;
        node = node->parent();
    }
}

void GameManager::propagate_terminal(Node* node) {
    Node* cur = node->parent();
    while (cur != nullptr) {
        if (cur->terminal_status() != 0 || cur->is_leaf()) {
            cur = cur->parent();
            continue;
        }

        bool all_resolved = true;
        bool any_draw = false;
        bool found_child_loss = false;

        for (int i = 0; i < cur->num_children(); i++) {
            int8_t cs = cur->child(i)->terminal_status();
            if (cs == 0) {
                all_resolved = false;
                break;
            }
            if (cs == 1) {
                found_child_loss = true;
            }
            if (cs == 2) {
                any_draw = true;
            }
        }

        if (found_child_loss) {
            cur->set_terminal_status(-1);
        } else if (all_resolved) {
            if (any_draw) {
                cur->set_terminal_status(2);
            } else {
                cur->set_terminal_status(1);
            }
        } else {
            break;
        }

        cur = cur->parent();
    }
}

Node* GameManager::select_child_advanced(Node* node, bool is_root) {
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

    // Sibling blending
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
            sibling_fpu = std::min(sibling_fpu, fpu_value + fpu_red * 0.5f);
        }
    }

    Node* best = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < node->num_children(); i++) {
        Node* child = node->child(i);

        // MCTS-solver
        if (child->terminal_status() == 1) {
            return child;
        }
        if (child->terminal_status() == -1) {
            continue;
        }

        float score;
        if (child->visit_count() == 0) {
            float child_fpu = sibling_fpu;

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
            float q = child->mean_value();
            float u = c_puct * child->prior() * sqrt_parent / (1.0f + child->visit_count());
            score = q + u;

            if (params_.uncertainty_weight > 0.0f && child->visit_count() > 1) {
                score += params_.uncertainty_weight * std::sqrt(child->value_variance());
            }
        }

        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }

    if (best == nullptr) {
        best = node->child(0);
    }

    return best;
}

bool GameManager::is_two_fold_repetition(uint64_t hash, const std::vector<uint64_t>& path_hashes,
                                          const neural::PositionHistory& history) const {
    for (uint64_t h : path_hashes) {
        if (h == hash) return true;
    }
    const auto& hist_hashes = history.hashes();
    for (uint64_t h : hist_hashes) {
        if (h == hash) return true;
    }
    return false;
}

void GameManager::expand_node(Node* node, const Position& pos, const EvalResult& eval_result) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 0) {
        if (pos.in_check()) {
            node->set_terminal_status(1);
        } else {
            node->set_terminal_status(2);
        }
        return;
    }

    for (int i = 0; i < num_moves; i++) {
        float prior = (i < static_cast<int>(eval_result.policy.size())) ? eval_result.policy[i] : 0.0f;
        node->add_child(moves[i], prior);
    }
    node->sort_children_by_prior();
}

bool GameManager::gather_leaf_from_game(int game_idx, std::vector<PendingLeaf>& batch) {
    auto& game = games_[game_idx];
    Node* node = game.root.get();
    std::vector<Move> path_moves;
    std::vector<uint64_t> path_hashes;
    std::vector<Node*> path_nodes;

    path_hashes.push_back(neural::PositionHistory::compute_hash(game.history.current()));

    while (!node->is_leaf()) {
        bool is_root = (node->parent() == nullptr);
        Node* child = select_child_advanced(node, is_root);
        path_moves.push_back(child->move());

        child->apply_virtual_loss();
        path_nodes.push_back(child);

        // Two-fold repetition check
        if (params_.two_fold_draw) {
            Position child_pos;
            replay_moves(game.history.current(), path_moves, child_pos);
            uint64_t child_hash = neural::PositionHistory::compute_hash(child_pos);

            if (is_two_fold_repetition(child_hash, path_hashes, game.history)) {
                revert_virtual_loss_path(path_nodes);
                child->set_terminal_status(2);
                backpropagate(child, 0.0f);
                propagate_terminal(child);
                return false;
            }

            path_hashes.push_back(child_hash);
        }

        node = child;
    }

    // node is a leaf
    Position leaf_pos;
    replay_moves(game.history.current(), path_moves, leaf_pos);

    // Already-visited terminal
    if (node->visit_count() > 0 && node->terminal_status() != 0) {
        float value = 0.0f;
        if (node->terminal_status() == 1) value = 1.0f;
        else if (node->terminal_status() == -1) value = -1.0f;
        revert_virtual_loss_path(path_nodes);
        backpropagate(node, -value);
        return false;
    }

    // NN cache check
    uint64_t pos_hash = neural::PositionHistory::compute_hash(leaf_pos);
    const CacheEntry* cached = cache_.get(pos_hash);
    if (cached) {
        EvalResult eval_result;
        eval_result.policy = cached->policy;
        eval_result.value = cached->value;

        expand_node(node, leaf_pos, eval_result);

        if (node->is_leaf()) {
            float value = 0.0f;
            if (node->terminal_status() == 1) value = 1.0f;
            else if (node->terminal_status() == -1) value = -1.0f;
            revert_virtual_loss_path(path_nodes);
            backpropagate(node, -value);
            propagate_terminal(node);
            return false;
        }

        revert_virtual_loss_path(path_nodes);
        backpropagate(node, -eval_result.value);
        return false;
    }

    // Queue for batch evaluation
    PendingLeaf pl;
    pl.game_idx = game_idx;
    pl.leaf = node;
    pl.position = leaf_pos;
    pl.path_nodes = std::move(path_nodes);
    batch.push_back(std::move(pl));
    return true;
}

bool GameManager::should_prune(const GameState& game) const {
    if (!params_.smart_pruning) return false;
    if (game.root->num_children() < 2) return false;
    if (game.sims_done < params_.batch_size) return false;

    int remaining = game.target_sims - game.sims_done;
    int best_visits = 0, second_visits = 0;
    for (int i = 0; i < game.root->num_children(); i++) {
        int vc = game.root->child(i)->visit_count();
        if (vc > best_visits) {
            second_visits = best_visits;
            best_visits = vc;
        } else if (vc > second_visits) {
            second_visits = vc;
        }
    }
    return second_visits + remaining < static_cast<int>(best_visits * params_.smart_pruning_factor);
}

void GameManager::add_dirichlet_noise(Node* root) {
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

void GameManager::add_shaped_dirichlet_noise(Node* root) {
    if (root->is_leaf()) return;

    int num_children = root->num_children();

    std::vector<float> priors(num_children);
    std::vector<float> log_priors(num_children);
    float max_log = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_children; i++) {
        priors[i] = root->child(i)->prior();
        log_priors[i] = std::log(priors[i] + 1e-8f);
        max_log = std::max(max_log, log_priors[i]);
    }

    float threshold = max_log - 2.0f;

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

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<float> noise(num_children);
    float noise_sum = 0.0f;
    for (int i = 0; i < num_children; i++) {
        float scaled_alpha = std::max(0.01f, params_.dirichlet_alpha * weights[i] * num_children);
        std::gamma_distribution<float> gamma_dist(scaled_alpha, 1.0f);
        noise[i] = gamma_dist(gen);
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

int GameManager::step() {
    int newly_completed = 0;

    // Expand roots that haven't been expanded yet
    for (auto& game : games_) {
        if (!game.root_expanded && !game.search_complete) {
            expand_root(game);
        }
    }

    // Count active games
    int num_active = 0;
    for (const auto& game : games_) {
        if (!game.search_complete) num_active++;
    }
    if (num_active == 0) return 0;

    // Determine how many leaves to gather per game
    int total_batch = std::max(1, params_.batch_size);
    int per_game = std::max(1, total_batch / std::max(1, num_active));

    // Gather phase: collect leaves from all active games
    std::vector<PendingLeaf> batch;
    batch.reserve(total_batch);

    for (int g = 0; g < static_cast<int>(games_.size()); g++) {
        auto& game = games_[g];
        if (game.search_complete) continue;

        // Check smart pruning
        if (should_prune(game)) {
            game.search_complete = true;
            newly_completed++;
            continue;
        }

        // Check if root is resolved
        if (game.root->terminal_status() != 0) {
            game.search_complete = true;
            newly_completed++;
            continue;
        }

        int leaves_this_game = std::min(per_game, game.target_sims - game.sims_done);
        for (int i = 0; i < leaves_this_game; i++) {
            gather_leaf_from_game(g, batch);
        }

        // Count sims for this game (including cache hits / terminals handled in gather)
        game.sims_done += leaves_this_game;

        // Check completion
        if (game.sims_done >= game.target_sims) {
            game.search_complete = true;
            newly_completed++;
        }
    }

    // Batch evaluate all pending leaves
    if (!batch.empty()) {
        int batch_size = static_cast<int>(batch.size());

        // Encode all positions
        std::vector<std::vector<float>> encode_buffers(batch_size, std::vector<float>(neural::TENSOR_SIZE));
        std::vector<std::vector<Move>> legal_moves_vec(batch_size);
        std::vector<int> num_legal_moves(batch_size);

        for (int b = 0; b < batch_size; b++) {
            neural::encode_position(batch[b].position, encode_buffers[b].data());

            Move moves[MAX_MOVES];
            int nm = generate_legal_moves(batch[b].position, moves);
            legal_moves_vec[b].assign(moves, moves + nm);
            num_legal_moves[b] = nm;
        }

        // Build batch requests
        std::vector<neural::BatchRequest> requests(batch_size);
        for (int b = 0; b < batch_size; b++) {
            requests[b].encoded_planes = encode_buffers[b].data();
            requests[b].legal_moves = legal_moves_vec[b].data();
            requests[b].num_legal_moves = num_legal_moves[b];
        }

        // Run batch inference
        auto results = evaluator_.evaluate_batch(requests);

        // Scatter results back
        for (int b = 0; b < batch_size; b++) {
            auto& pl = batch[b];

            EvalResult eval_result;
            eval_result.policy = results[b].policy;
            eval_result.value = results[b].value;

            // Expand the leaf
            expand_node(pl.leaf, pl.position, eval_result);

            if (pl.leaf->is_leaf()) {
                // Terminal after expansion
                float value = 0.0f;
                if (pl.leaf->terminal_status() == 1) value = 1.0f;
                else if (pl.leaf->terminal_status() == -1) value = -1.0f;
                revert_virtual_loss_path(pl.path_nodes);
                backpropagate(pl.leaf, -value);
                propagate_terminal(pl.leaf);
            } else {
                revert_virtual_loss_path(pl.path_nodes);
                backpropagate(pl.leaf, -eval_result.value);

                // Cache
                uint64_t pos_hash = neural::PositionHistory::compute_hash(pl.position);
                CacheEntry entry;
                entry.policy = eval_result.policy;
                entry.value = eval_result.value;
                entry.num_moves = num_legal_moves[b];
                cache_.put(pos_hash, std::move(entry));

                propagate_terminal(pl.leaf);
            }
        }
    }

    return newly_completed;
}

bool GameManager::all_complete() const {
    for (const auto& game : games_) {
        if (!game.search_complete) return false;
    }
    return true;
}

bool GameManager::is_complete(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(games_.size())) return false;
    return games_[idx].search_complete;
}

SearchResult GameManager::get_result(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(games_.size())) {
        throw std::runtime_error("GameManager::get_result: invalid index");
    }
    return build_result(games_[idx]);
}

SearchResult GameManager::build_result(const GameState& game) const {
    SearchResult result;
    Node* root = game.root.get();
    const Position& root_pos = game.history.current();

    result.best_move = root->is_leaf() ? Move::none() : root->best_move();
    result.root_value = root->mean_value();
    result.raw_value = game.raw_value;
    result.raw_policy = game.raw_policy;

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

    // Build policy_target
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

} // namespace mcts

#endif // HAS_LIBTORCH
