#ifdef HAS_LIBTORCH

#include "mcts/game_manager.h"
#include "core/movegen.h"
#include "core/bitboard.h"
#include "core/move_parser.h"
#include "syzygy/syzygy.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstring>

namespace mcts {

// Helper: revert virtual loss on all nodes in a path. n counts multivisit claims.
static void revert_virtual_loss_path(const std::vector<Node*>& path_nodes, int n = 1) {
    for (Node* node : path_nodes) {
        node->revert_virtual_loss(n);
    }
}

GameManager::GameManager(neural::RawBatchEvaluator& evaluator, const SearchParams& params)
    : evaluator_(evaluator), params_(params), cache_(params.nn_cache_size) {
    int max_batch = std::max(1, params.batch_size);
    flat_encode_buffer_.resize(max_batch * neural::TENSOR_SIZE);
    legal_moves_vec_.resize(max_batch);
    num_legal_moves_vec_.resize(max_batch);
}

void GameManager::init_game(int idx, const neural::PositionHistory& history, int num_sims) {
    if (idx < 0 || idx >= static_cast<int>(games_.size())) {
        throw std::runtime_error("GameManager::init_game: invalid index");
    }
    auto& game = games_[idx];
    game.history = history;
    game.root = pool_.allocate();
    game.root->set_pool_managed(true);
    game.sims_done = 0;
    game.target_sims = num_sims;
    game.search_complete = false;
    game.raw_value = 0.0f;
    game.raw_policy.fill(0.0f);
    game.root_expanded = false;
}

void GameManager::init_games(int num_games, int num_sims) {
    pool_.reset();
    games_.resize(num_games);
    for (int i = 0; i < num_games; i++) {
        Position start_pos;
        start_pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        neural::PositionHistory history;
        history.reset(start_pos);

        games_[i].history = history;
        games_[i].root = pool_.allocate();
        games_[i].root->set_pool_managed(true);
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
    pool_.reset();
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
                Move m = parse_uci_move(pos, uci_str);
                UndoInfo undo;
                pos.make_move(m, undo);
                history.push(pos);
            }
        }

        games_[i].history = history;
        games_[i].root = pool_.allocate();
        games_[i].root->set_pool_managed(true);
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

    auto results = evaluator_.evaluate_batch_raw({req});
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

    // Expand root node with edges
    game.root->create_edges(moves, br.policy.data(), num_moves);
    game.root->sort_edges_by_prior();
    game.root->set_mlh(br.mlh);
    game.root->update(br.value);

    // Cache root evaluation
    uint64_t root_hash = neural::PositionHistory::compute_hash(root_pos);
    CacheEntry entry;
    entry.policy = br.policy;
    entry.value = br.value;
    entry.num_moves = num_moves;
    entry.mlh = br.mlh;
    cache_.put(root_hash, std::move(entry));

    // Add Dirichlet noise
    if (params_.add_noise) {
        add_dirichlet_noise(game.root);
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

void GameManager::backpropagate(Node* node, float value, int n) {
    while (node != nullptr) {
        node->update(value, n);
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

        for (int i = 0; i < cur->num_edges(); i++) {
            Node* child = cur->child_node(i);
            if (!child) {
                all_resolved = false;
                break;
            }
            int8_t cs = child->terminal_status();
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
    return select_child_advanced(node, is_root, nullptr);
}

Node* GameManager::select_child_advanced(Node* node, bool is_root, int* out_idx) {
    assert(!node->is_leaf());

    bool absolute_fpu_at_root = is_root && params_.fpu_absolute_root;
    float fpu_red = is_root ? params_.fpu_reduction_root : params_.fpu_reduction;
    float fpu_value = absolute_fpu_at_root
        ? params_.fpu_absolute_root_value
        : node->mean_value() - fpu_red;
    float c_puct = dynamic_cpuct(node->visit_count());

    // Variance-scaled cPUCT
    if (params_.variance_scaling && node->visit_count() > 1) {
        float var = node->value_variance();
        float scale = std::sqrt(var) / 0.5f;
        scale = std::max(0.5f, std::min(2.0f, scale));
        c_puct *= scale;
    }

    float sqrt_parent = std::sqrt(static_cast<float>(node->visit_count()));

    // Sibling blending — skip at root when absolute-FPU is on: the whole point
    // is to force visiting every root child, which sibling blending would defeat.
    float sibling_fpu = fpu_value;
    if (params_.sibling_blending && !absolute_fpu_at_root) {
        float visited_sum = 0.0f;
        int visited_count = 0;
        for (int i = 0; i < node->num_edges(); i++) {
            Node* child = node->child_node(i);
            if (child && child->visit_count() > 0) {
                visited_sum += child->mean_value();
                visited_count++;
            }
        }
        if (visited_count > 0) {
            sibling_fpu = visited_sum / visited_count;
            sibling_fpu = std::min(sibling_fpu, fpu_value + fpu_red * 0.5f);
        }
    }

    int best_idx = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < node->num_edges(); i++) {
        Node* child = node->child_node(i);
        float edge_prior = node->edge(i).prior();

        // MCTS-solver
        if (child && child->terminal_status() == 1) {
            if (out_idx) *out_idx = i;
            return node->ensure_child(i, &pool_);
        }
        if (child && child->terminal_status() == -1) {
            continue;
        }

        float score;
        if (!child || child->visit_count() == 0) {
            float child_fpu = sibling_fpu;

            if (params_.sibling_blending && !absolute_fpu_at_root) {
                float child_prior = edge_prior;
                float sim_sum = 0.0f;
                int sim_count = 0;
                for (int j = 0; j < node->num_edges(); j++) {
                    Node* sib = node->child_node(j);
                    if (sib && sib->visit_count() > 0 && std::abs(node->edge(j).prior() - child_prior) < 0.10f) {
                        sim_sum += sib->mean_value();
                        sim_count++;
                    }
                }
                if (sim_count > 0) {
                    child_fpu = sim_sum / sim_count;
                }
            }

            score = child_fpu + c_puct * edge_prior * sqrt_parent / 1.0f;
        } else {
            float q = child->mean_value();
            float u = c_puct * edge_prior * sqrt_parent / (1.0f + child->visit_count());
            score = q + u;

            if (params_.uncertainty_weight > 0.0f && child->visit_count() > 1) {
                score += params_.uncertainty_weight * std::sqrt(child->value_variance());
            }

            // Lc0-style MLH bonus: when the parent is confident (|q| > threshold),
            // prefer shorter wins / longer losses by rewarding the child whose
            // moves-left is below the parent's. delta_m = parent.m - child.m - 1
            // (the -1 accounts for the move we just played).
            if (params_.mlh_weight > 0.0f && std::abs(q) > params_.mlh_q_threshold) {
                float delta_m = node->mlh() - child->mlh() - 1.0f;
                float bonus = params_.mlh_weight * (q >= 0.0f ? 1.0f : -1.0f) * delta_m;
                bonus = std::max(-params_.mlh_cap, std::min(params_.mlh_cap, bonus));
                score += bonus;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        best_idx = 0;
    }

    if (out_idx) *out_idx = best_idx;
    return node->ensure_child(best_idx, &pool_);
}

// Project how many additional visits the already-selected child (best_idx) would
// receive before PUCT switches to a different sibling. Single-level lookahead at
// the leaf's parent; caller ensures this is not the root and the selected child
// is non-terminal.
int GameManager::compute_collapse_visits(Node* parent, int best_idx, bool is_root) {
    int cap = params_.max_collapse_visits;
    if (cap <= 1 || is_root) return 1;
    assert(!parent->is_leaf());
    assert(best_idx >= 0 && best_idx < parent->num_edges());

    Node* selected = parent->child_node(best_idx);
    if (selected && selected->terminal_status() != 0) return 1;

    // Recompute parent-level knobs (mirrors select_child_advanced but with +m visits on selected)
    float fpu_red = is_root ? params_.fpu_reduction_root : params_.fpu_reduction;
    float fpu_value = parent->mean_value() - fpu_red;

    // Sibling blending FPU for unvisited siblings (computed from current state, doesn't change with m)
    float sibling_fpu = fpu_value;
    if (params_.sibling_blending) {
        float visited_sum = 0.0f;
        int visited_count = 0;
        for (int i = 0; i < parent->num_edges(); i++) {
            Node* child = parent->child_node(i);
            if (child && child->visit_count() > 0) {
                visited_sum += child->mean_value();
                visited_count++;
            }
        }
        if (visited_count > 0) {
            sibling_fpu = visited_sum / visited_count;
            sibling_fpu = std::min(sibling_fpu, fpu_value + fpu_red * 0.5f);
        }
    }

    float selected_prior = parent->edge(best_idx).prior();
    int selected_N = selected ? selected->visit_count() : 0;
    float selected_Q = (selected && selected->visit_count() > 0) ? selected->mean_value() : sibling_fpu;
    int parent_N_base = parent->visit_count();

    // Try increasing collapse counts until a sibling would win at the projected state.
    // We already applied 1 virtual loss during the real descent, so m=1 is the baseline.
    // Return the largest m in [1, cap] for which selected still wins.
    int best_m = 1;
    for (int m = 2; m <= cap; m++) {
        int parent_N = parent_N_base + (m - 1); // (m-1) extra virtual losses beyond the one already applied
        float sqrt_parent = std::sqrt(static_cast<float>(std::max(1, parent_N)));
        float c_puct = params_.c_puct_init + params_.c_puct_factor * std::log(
            (parent_N + params_.c_puct_base) / params_.c_puct_base);
        if (params_.variance_scaling && parent_N > 1) {
            float var = parent->value_variance();
            float scale = std::sqrt(var) / 0.5f;
            scale = std::max(0.5f, std::min(2.0f, scale));
            c_puct *= scale;
        }

        // Selected child's projected score at +m total visits
        int sel_N_proj = selected_N + m;
        float sel_u = c_puct * selected_prior * sqrt_parent / (1.0f + sel_N_proj);
        float sel_score = selected_Q + sel_u;
        // Uncertainty bonus only applies for visited children; selected was visited (we just descended through it).
        if (params_.uncertainty_weight > 0.0f && selected && selected->visit_count() > 1) {
            sel_score += params_.uncertainty_weight * std::sqrt(selected->value_variance());
        }

        // Scan siblings — use current siblings' state (unchanged by virtual loss on selected).
        float best_sibling = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < parent->num_edges(); i++) {
            if (i == best_idx) continue;
            Node* child = parent->child_node(i);
            if (child && child->terminal_status() == -1) continue;  // losing, skip
            float edge_prior = parent->edge(i).prior();
            float score;
            if (!child || child->visit_count() == 0) {
                float child_fpu = sibling_fpu;
                if (params_.sibling_blending) {
                    float child_prior = edge_prior;
                    float sim_sum = 0.0f;
                    int sim_count = 0;
                    for (int j = 0; j < parent->num_edges(); j++) {
                        Node* sib = parent->child_node(j);
                        if (sib && sib->visit_count() > 0 &&
                                std::abs(parent->edge(j).prior() - child_prior) < 0.10f) {
                            sim_sum += sib->mean_value();
                            sim_count++;
                        }
                    }
                    if (sim_count > 0) child_fpu = sim_sum / sim_count;
                }
                score = child_fpu + c_puct * edge_prior * sqrt_parent;
            } else {
                float q = child->mean_value();
                float u = c_puct * edge_prior * sqrt_parent / (1.0f + child->visit_count());
                score = q + u;
                if (params_.uncertainty_weight > 0.0f && child->visit_count() > 1) {
                    score += params_.uncertainty_weight * std::sqrt(child->value_variance());
                }
            }
            if (score > best_sibling) best_sibling = score;
        }

        if (sel_score < best_sibling) break;  // sibling would win at m; stop at previous best
        best_m = m;
    }

    return best_m;
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

    // Build priors array
    std::vector<float> priors(num_moves);
    for (int i = 0; i < num_moves; i++) {
        priors[i] = (i < static_cast<int>(eval_result.policy.size())) ? eval_result.policy[i] : 0.0f;
    }

    node->create_edges(moves, priors.data(), num_moves);
    node->sort_edges_by_prior();
    node->set_mlh(eval_result.mlh);
}

int GameManager::gather_leaf_from_game(int game_idx, std::vector<PendingLeaf>& batch) {
    auto& game = games_[game_idx];
    Node* node = game.root;
    std::vector<Move> path_moves;
    std::vector<uint64_t> path_hashes;
    std::vector<Node*> path_nodes;

    // Maintain running position instead of replaying from root each level
    Position current_pos = game.history.current();
    path_hashes.push_back(neural::PositionHistory::compute_hash(current_pos));

    // Track the leaf's parent and which edge we descended through from it,
    // for multivisit collapse computation.
    Node* leaf_parent = nullptr;
    int leaf_parent_best_idx = -1;
    bool leaf_parent_is_root = false;

    while (!node->is_leaf()) {
        bool is_root = (node->parent() == nullptr);
        int best_idx = -1;
        Node* child = select_child_advanced(node, is_root, &best_idx);

        leaf_parent = node;
        leaf_parent_best_idx = best_idx;
        leaf_parent_is_root = is_root;

        path_moves.push_back(child->move());

        child->apply_virtual_loss();
        path_nodes.push_back(child);

        // Advance position incrementally
        UndoInfo undo;
        current_pos.make_move(child->move(), undo);

        if (params_.two_fold_draw) {
            uint64_t child_hash = neural::PositionHistory::compute_hash(current_pos);

            if (is_two_fold_repetition(child_hash, path_hashes, game.history)) {
                revert_virtual_loss_path(path_nodes);
                child->set_terminal_status(2);
                backpropagate(child, 0.0f);
                propagate_terminal(child);
                return 1;
            }

            path_hashes.push_back(child_hash);
        }

        node = child;
    }

    // current_pos is already the leaf position

    // Already-visited terminal
    if (node->visit_count() > 0 && node->terminal_status() != 0) {
        float value = 0.0f;
        if (node->terminal_status() == 1) value = 1.0f;
        else if (node->terminal_status() == -1) value = -1.0f;
        revert_virtual_loss_path(path_nodes);
        backpropagate(node, -value);
        return 1;
    }

    // Syzygy WDL probe — same logic as Search::gather_leaf. Resolves the
    // leaf without an NN slot when the position is in the loaded tablebase.
    if (params_.use_syzygy && syzygy::TableBase::ready()) {
        syzygy::ProbeResult tb = syzygy::TableBase::probe_wdl(current_pos);
        if (tb.hit) {
            int8_t status;
            float value;
            switch (tb.wdl) {
                case syzygy::WDL::WIN:  status =  -1; value =  1.0f; break;
                case syzygy::WDL::LOSS: status =   1; value = -1.0f; break;
                default:                status =   2; value =  0.0f; break;
            }
            node->set_terminal_status(status);
            revert_virtual_loss_path(path_nodes);
            backpropagate(node, value);
            propagate_terminal(node);
            return 1;
        }
    }

    // NN cache check
    uint64_t pos_hash = neural::PositionHistory::compute_hash(current_pos);
    const CacheEntry* cached = cache_.get(pos_hash);
    if (cached) {
        EvalResult eval_result;
        eval_result.policy = cached->policy;
        eval_result.value = cached->value;
        eval_result.mlh = cached->mlh;

        expand_node(node, current_pos, eval_result);

        if (node->is_leaf()) {
            float value = 0.0f;
            if (node->terminal_status() == 1) value = 1.0f;
            else if (node->terminal_status() == -1) value = -1.0f;
            revert_virtual_loss_path(path_nodes);
            backpropagate(node, -value);
            propagate_terminal(node);
            return 1;
        }

        revert_virtual_loss_path(path_nodes);
        backpropagate(node, -eval_result.value);
        return 1;
    }

    // Multivisit collapse: project how many extra visits PUCT would route through
    // the same leaf before switching siblings. Apply (M-1) extra virtual losses
    // on the path, and stamp pl.multivisit so batch-return backprops N at once.
    int M = 1;
    if (leaf_parent != nullptr && params_.max_collapse_visits > 1) {
        M = compute_collapse_visits(leaf_parent, leaf_parent_best_idx, leaf_parent_is_root);
        if (M > 1) {
            int extra = M - 1;
            for (Node* n : path_nodes) {
                n->apply_virtual_loss(extra);
            }
        }
    }

    // Queue for batch evaluation
    PendingLeaf pl;
    pl.game_idx = game_idx;
    pl.leaf = node;
    pl.position = current_pos;
    pl.path_nodes = std::move(path_nodes);
    pl.multivisit = M;
    batch.push_back(std::move(pl));
    return M;
}

bool GameManager::should_prune(const GameState& game) const {
    if (!params_.smart_pruning) return false;
    if (game.root->num_edges() < 2) return false;
    if (game.sims_done < params_.batch_size) return false;

    int remaining = game.target_sims - game.sims_done;
    int best_visits = 0, second_visits = 0;
    for (int i = 0; i < game.root->num_edges(); i++) {
        Node* child = game.root->child_node(i);
        int vc = child ? child->visit_count() : 0;
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

    int num_edges = root->num_edges();
    std::vector<float> noise(num_edges);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(params_.dirichlet_alpha, 1.0f);

    float noise_sum = 0.0f;
    for (int i = 0; i < num_edges; i++) {
        noise[i] = gamma(gen);
        noise_sum += noise[i];
    }
    if (noise_sum > 0.0f) {
        for (int i = 0; i < num_edges; i++) {
            noise[i] /= noise_sum;
        }
    }

    float eps = params_.dirichlet_epsilon;
    for (int i = 0; i < num_edges; i++) {
        float old_prior = root->edge(i).prior();
        float new_prior = (1.0f - eps) * old_prior + eps * noise[i];
        root->edge(i).set_prior(new_prior);
    }
}

void GameManager::add_shaped_dirichlet_noise(Node* root) {
    if (root->is_leaf()) return;

    int num_edges = root->num_edges();

    std::vector<float> priors(num_edges);
    std::vector<float> log_priors(num_edges);
    float max_log = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_edges; i++) {
        priors[i] = root->edge(i).prior();
        log_priors[i] = std::log(priors[i] + 1e-8f);
        max_log = std::max(max_log, log_priors[i]);
    }

    float threshold = max_log - 2.0f;

    std::vector<float> weights(num_edges, 0.5f);
    float weight_sum = 0.0f;
    for (int i = 0; i < num_edges; i++) {
        if (log_priors[i] > threshold) {
            weights[i] += 0.5f * (log_priors[i] - threshold) / 2.0f;
        }
        weight_sum += weights[i];
    }
    if (weight_sum > 0.0f) {
        for (int i = 0; i < num_edges; i++) {
            weights[i] /= weight_sum;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<float> noise(num_edges);
    float noise_sum = 0.0f;
    for (int i = 0; i < num_edges; i++) {
        float scaled_alpha = std::max(0.01f, params_.dirichlet_alpha * weights[i] * num_edges);
        std::gamma_distribution<float> gamma_dist(scaled_alpha, 1.0f);
        noise[i] = gamma_dist(gen);
        noise_sum += noise[i];
    }
    if (noise_sum > 0.0f) {
        for (int i = 0; i < num_edges; i++) {
            noise[i] /= noise_sum;
        }
    }

    float eps = params_.dirichlet_epsilon;
    for (int i = 0; i < num_edges; i++) {
        float old_prior = root->edge(i).prior();
        float new_prior = (1.0f - eps) * old_prior + eps * noise[i];
        root->edge(i).set_prior(new_prior);
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
        int sims_accumulated = 0;
        for (int i = 0; i < leaves_this_game; i++) {
            // Each descent contributes >=1 sim (1 for early-exit, M for multivisit)
            int contrib = gather_leaf_from_game(g, batch);
            sims_accumulated += std::max(1, contrib);
            // Stop early if multivisit collapse has already covered the target
            if (game.sims_done + sims_accumulated >= game.target_sims) break;
        }

        game.sims_done += sims_accumulated;

        // Check completion
        if (game.sims_done >= game.target_sims) {
            game.search_complete = true;
            newly_completed++;
        }
    }

    // Top-up pass: fill the batch if cache hits / terminals / repetitions in the
    // proportional pass left it short. Without this, a step where many games hit
    // the NN cache sends a half-empty batch to the GPU and wastes throughput.
    int safety_iterations = total_batch * 2;
    while (static_cast<int>(batch.size()) < total_batch && safety_iterations-- > 0) {
        bool made_progress = false;
        for (int g = 0; g < static_cast<int>(games_.size()); g++) {
            if (static_cast<int>(batch.size()) >= total_batch) break;
            auto& game = games_[g];
            if (game.search_complete) continue;
            if (game.sims_done >= game.target_sims) {
                game.search_complete = true;
                newly_completed++;
                continue;
            }
            if (game.root->terminal_status() != 0) {
                game.search_complete = true;
                newly_completed++;
                continue;
            }
            int contrib = gather_leaf_from_game(g, batch);
            game.sims_done += std::max(1, contrib);
            made_progress = true;
            if (game.sims_done >= game.target_sims) {
                game.search_complete = true;
                newly_completed++;
            }
        }
        if (!made_progress) break;
    }

    // Batch evaluate all pending leaves
    if (!batch.empty()) {
        int batch_size = static_cast<int>(batch.size());

        // Encode all positions — use pre-allocated buffers (no per-step allocation)
        for (int b = 0; b < batch_size; b++) {
            neural::encode_position(batch[b].position,
                flat_encode_buffer_.data() + b * neural::TENSOR_SIZE);
            Move moves[MAX_MOVES];
            int nm = generate_legal_moves(batch[b].position, moves);
            legal_moves_vec_[b].assign(moves, moves + nm);
            num_legal_moves_vec_[b] = nm;
        }

        // Build batch requests
        std::vector<neural::BatchRequest> requests(batch_size);
        for (int b = 0; b < batch_size; b++) {
            requests[b].encoded_planes = flat_encode_buffer_.data() + b * neural::TENSOR_SIZE;
            requests[b].legal_moves = legal_moves_vec_[b].data();
            requests[b].num_legal_moves = num_legal_moves_vec_[b];
        }

        // Run batch inference
        auto results = evaluator_.evaluate_batch_raw(requests);

        // Scatter results back
        for (int b = 0; b < batch_size; b++) {
            auto& pl = batch[b];

            EvalResult eval_result;
            eval_result.policy = results[b].policy;
            eval_result.value = results[b].value;
            eval_result.mlh = results[b].mlh;

            // Expand the leaf
            expand_node(pl.leaf, pl.position, eval_result);

            if (pl.leaf->is_leaf()) {
                // Terminal after expansion
                float value = 0.0f;
                if (pl.leaf->terminal_status() == 1) value = 1.0f;
                else if (pl.leaf->terminal_status() == -1) value = -1.0f;
                revert_virtual_loss_path(pl.path_nodes, pl.multivisit);
                backpropagate(pl.leaf, -value, pl.multivisit);
                propagate_terminal(pl.leaf);
            } else {
                revert_virtual_loss_path(pl.path_nodes, pl.multivisit);
                backpropagate(pl.leaf, -eval_result.value, pl.multivisit);

                // Cache
                uint64_t pos_hash = neural::PositionHistory::compute_hash(pl.position);
                CacheEntry entry;
                entry.policy = eval_result.policy;
                entry.value = eval_result.value;
                entry.num_moves = num_legal_moves_vec_[b];
                entry.mlh = eval_result.mlh;
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
    Node* root = game.root;
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

    int num_edges = root->num_edges();
    result.moves.resize(num_edges);
    result.visit_counts.resize(num_edges);

    int total_child_visits = 0;
    Color stm = root_pos.side_to_move();

    for (int i = 0; i < num_edges; i++) {
        result.moves[i] = root->edge(i).move();
        Node* child = root->child_node(i);
        result.visit_counts[i] = child ? child->visit_count() : 0;
        total_child_visits += result.visit_counts[i];
    }

    // Build policy_target
    result.policy_target.fill(0.0f);
    if (total_child_visits > 0) {
        for (int i = 0; i < num_edges; i++) {
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
