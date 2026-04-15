#ifdef HAS_LIBTORCH
#include <gtest/gtest.h>

#include "mcts/game_manager.h"
#include "neural/neural_evaluator.h"
#include "core/position.h"
#include "core/attacks.h"
#include "core/types.h"

#include <string>
#include <vector>

using namespace mcts;

namespace {

// Minimal stub RawBatchEvaluator for testing the GameManager new API without
// LibTorch models. Returns uniform policy across legal moves, value=0 (draw),
// and a fixed mlh=20.0f.
class StubBatchEvaluator : public neural::RawBatchEvaluator {
public:
    std::vector<neural::BatchResult> evaluate_batch_raw(
            const std::vector<neural::BatchRequest>& requests) override {
        std::vector<neural::BatchResult> out;
        out.reserve(requests.size());
        for (const auto& req : requests) {
            neural::BatchResult r;
            int nm = std::max(1, req.num_legal_moves);
            float p = 1.0f / static_cast<float>(nm);
            r.policy.assign(req.num_legal_moves, p);
            r.full_policy.assign(neural::POLICY_SIZE, 0.0f);
            r.value = 0.0f;
            r.mlh = 20.0f;
            out.push_back(std::move(r));
        }
        return out;
    }
};

class GameManagerRootStats : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        attacks::init();
    }
};

}  // namespace

TEST_F(GameManagerRootStats, StepStatsPopulatesFields) {
    SearchParams params;
    params.batch_size = 8;
    params.add_noise = false;       // deterministic for the test
    params.use_syzygy = false;
    StubBatchEvaluator eval;
    GameManager gm(eval, params);
    gm.init_games(/*num_games=*/2, /*num_sims=*/8);

    std::vector<int> targets = {8, 8};
    auto stats = gm.step_stats(targets);

    ASSERT_EQ(stats.size(), 2u);
    for (const auto& s : stats) {
        EXPECT_EQ(s.terminal_status, 0);
        EXPECT_GT(s.n_legal, 0);
        EXPECT_EQ(static_cast<int>(s.visits.size()), s.n_legal);
        EXPECT_EQ(static_cast<int>(s.q_per_child.size()), s.n_legal);
        EXPECT_EQ(static_cast<int>(s.legal_moves.size()), s.n_legal);
        EXPECT_GE(s.best_child_idx, 0);
        EXPECT_LT(s.best_child_idx, s.n_legal);
        EXPECT_GT(s.sims_done, 0);
        EXPECT_EQ(static_cast<int>(s.raw_nn_policy.size()), neural::POLICY_SIZE);
        // Raw NN value placeholder: value=0 -> (w=0.5, d=0, l=0.5).
        EXPECT_NEAR(s.raw_nn_value[0], 0.5f, 1e-5f);
        EXPECT_NEAR(s.raw_nn_value[2], 0.5f, 1e-5f);
        // Stub returns mlh=20.
        EXPECT_NEAR(s.raw_nn_mlh, 20.0f, 1e-5f);
    }
}

TEST_F(GameManagerRootStats, ApplyMoveAdvancesGame) {
    SearchParams params;
    params.batch_size = 4;
    params.add_noise = false;
    params.use_syzygy = false;
    StubBatchEvaluator eval;
    GameManager gm(eval, params);
    gm.init_games(1, 4);

    auto stats = gm.step_stats({4});
    ASSERT_EQ(stats.size(), 1u);
    ASSERT_GE(stats[0].best_child_idx, 0);

    int ply_before = gm.get_ply(0);
    std::string fen_before = gm.get_fen(0);

    gm.apply_move(0, stats[0].best_child_idx);

    EXPECT_EQ(gm.get_ply(0), ply_before + 1);
    EXPECT_NE(gm.get_fen(0), fen_before);

    // After apply_move, sims_done resets and the tree is fresh.
    // A subsequent step_stats should run fresh sims without crashing.
    auto stats2 = gm.step_stats({4});
    ASSERT_EQ(stats2.size(), 1u);
    EXPECT_GT(stats2[0].sims_done, 0);
    EXPECT_GT(stats2[0].n_legal, 0);
}

#endif  // HAS_LIBTORCH
