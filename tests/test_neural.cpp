// tests/test_neural.cpp
#include <gtest/gtest.h>
#include "neural/policy_map.h"
#include "neural/encoder.h"
#include "core/types.h"
#include "core/attacks.h"
#include "core/movegen.h"

class NeuralTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { attacks::init(); }
};

// Reference values from Python: training/encoder.py move_to_index()
TEST_F(NeuralTest, PolicyMap_E2E4) {
    // E2=12, E4=28, queen-move encoding
    EXPECT_EQ(neural::move_to_policy_index(E2, E4, NO_PIECE_TYPE), 304);
}

TEST_F(NeuralTest, PolicyMap_KnightG1F3) {
    // G1=6, F3=21
    EXPECT_EQ(neural::move_to_policy_index(G1, F3, NO_PIECE_TYPE), 170);
}

TEST_F(NeuralTest, PolicyMap_QueenPromoE7E8) {
    // Queen promo uses normal queen-move encoding (promo=NO_PIECE_TYPE)
    EXPECT_EQ(neural::move_to_policy_index(E7, E8, NO_PIECE_TYPE), 1522);
}

TEST_F(NeuralTest, PolicyMap_UnderpromoKnightE7E8) {
    EXPECT_EQ(neural::move_to_policy_index(E7, E8, KNIGHT), 1554);
}

TEST_F(NeuralTest, PolicyMap_UnderpromoRookE7D8) {
    EXPECT_EQ(neural::move_to_policy_index(E7, D8, ROOK), 1553);
}

TEST_F(NeuralTest, PolicyMap_A2A3) {
    EXPECT_EQ(neural::move_to_policy_index(A2, A3, NO_PIECE_TYPE), 194);
}

TEST_F(NeuralTest, PolicyMap_B1C3) {
    EXPECT_EQ(neural::move_to_policy_index(B1, C3, NO_PIECE_TYPE), 44);
}

TEST_F(NeuralTest, PolicyMap_D1H5) {
    EXPECT_EQ(neural::move_to_policy_index(D1, H5, NO_PIECE_TYPE), 82);
}

TEST_F(NeuralTest, PolicyMap_TotalSize) {
    EXPECT_EQ(neural::POLICY_SIZE, 1858);
}

TEST_F(NeuralTest, PolicyMap_RuntimeTableSize) {
    EXPECT_EQ(neural::policy_table_size(), 1858);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_White) {
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    EXPECT_EQ(neural::move_to_policy_index(m, WHITE), 304);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_BlackMirrors) {
    // Black E7->E5: mirror to E2->E4
    Move m(E7, E5, FLAG_DOUBLE_PUSH);
    EXPECT_EQ(neural::move_to_policy_index(m, BLACK), 304);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_QueenPromo) {
    // Queen promo flag -> NO_PIECE_TYPE (uses normal encoding)
    Move m(E7, E8, FLAG_PROMO_QUEEN);
    EXPECT_EQ(neural::move_to_policy_index(m, WHITE), 1522);
}

TEST_F(NeuralTest, PolicyMap_MoveOverload_KnightPromo) {
    Move m(E7, E8, FLAG_PROMO_KNIGHT);
    EXPECT_EQ(neural::move_to_policy_index(m, WHITE), 1554);
}

TEST_F(NeuralTest, PolicyMap_InvalidReturnsNeg1) {
    // Same square -> invalid
    EXPECT_EQ(neural::move_to_policy_index(A1, A1, NO_PIECE_TYPE), -1);
}

TEST_F(NeuralTest, Encoder_StartingWhitePawns) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    // Plane 0 = our pawns. White to move: pawns on rank 1 (indices 8-15).
    // Plane layout: buf[plane * 64 + rank * 8 + file]
    float sum = 0;
    for (int i = 0; i < 64; i++) sum += buf[0 * 64 + i];
    EXPECT_FLOAT_EQ(sum, 8.0f);
    for (int f = 0; f < 8; f++)
        EXPECT_FLOAT_EQ(buf[0 * 64 + 1 * 8 + f], 1.0f);
}

TEST_F(NeuralTest, Encoder_StartingOpponentPawns) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    float sum = 0;
    for (int i = 0; i < 64; i++) sum += buf[6 * 64 + i];
    EXPECT_FLOAT_EQ(sum, 8.0f);
    for (int f = 0; f < 8; f++)
        EXPECT_FLOAT_EQ(buf[6 * 64 + 6 * 8 + f], 1.0f);
}

TEST_F(NeuralTest, Encoder_StartingWhiteKing) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    EXPECT_FLOAT_EQ(buf[5 * 64 + 0 * 8 + 4], 1.0f);
    float sum = 0;
    for (int i = 0; i < 64; i++) sum += buf[5 * 64 + i];
    EXPECT_FLOAT_EQ(sum, 1.0f);
}

TEST_F(NeuralTest, Encoder_BlackToMoveFlips) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    float sum = 0;
    for (int i = 0; i < 64; i++) sum += buf[0 * 64 + i];
    EXPECT_FLOAT_EQ(sum, 8.0f);
    for (int f = 0; f < 8; f++)
        EXPECT_FLOAT_EQ(buf[0 * 64 + 1 * 8 + f], 1.0f);
}

TEST_F(NeuralTest, Encoder_BlackToMoveFlipsAsymmetric) {
    Position pos;
    pos.set_fen("8/p7/8/8/8/8/P7/8 b - - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    // Black pawn on A7 (rank 6) flipped to rank 1, file 0 → our pawn plane 0
    EXPECT_FLOAT_EQ(buf[0 * 64 + 1 * 8 + 0], 1.0f);
    // White pawn on A2 (rank 1) flipped to rank 6, file 0 → opponent pawn plane 6
    EXPECT_FLOAT_EQ(buf[6 * 64 + 6 * 8 + 0], 1.0f);
}

TEST_F(NeuralTest, Encoder_PartialCastling) {
    Position pos;
    pos.set_fen("r3k3/8/8/8/8/8/8/4K2R w K - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    // White to move, only WHITE_OO → plane 106 filled, 107-109 zero
    float sum106 = 0, sum107 = 0, sum108 = 0, sum109 = 0;
    for (int i = 0; i < 64; i++) {
        sum106 += buf[106 * 64 + i];
        sum107 += buf[107 * 64 + i];
        sum108 += buf[108 * 64 + i];
        sum109 += buf[109 * 64 + i];
    }
    EXPECT_FLOAT_EQ(sum106, 64.0f);
    EXPECT_FLOAT_EQ(sum107, 0.0f);
    EXPECT_FLOAT_EQ(sum108, 0.0f);
    EXPECT_FLOAT_EQ(sum109, 0.0f);
}

TEST_F(NeuralTest, Encoder_ColorPlaneWhite) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int i = 0; i < 64; i++)
        EXPECT_FLOAT_EQ(buf[104 * 64 + i], 1.0f);
}

TEST_F(NeuralTest, Encoder_ColorPlaneBlack) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int i = 0; i < 64; i++)
        EXPECT_FLOAT_EQ(buf[104 * 64 + i], 0.0f);
}

TEST_F(NeuralTest, Encoder_CastlingWhite) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int plane = 106; plane <= 109; plane++) {
        float sum = 0;
        for (int i = 0; i < 64; i++) sum += buf[plane * 64 + i];
        EXPECT_FLOAT_EQ(sum, 64.0f) << "Plane " << plane;
    }
}

TEST_F(NeuralTest, Encoder_NoCastling) {
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int plane = 106; plane <= 109; plane++) {
        float sum = 0;
        for (int i = 0; i < 64; i++) sum += buf[plane * 64 + i];
        EXPECT_FLOAT_EQ(sum, 0.0f) << "Plane " << plane;
    }
}

TEST_F(NeuralTest, Encoder_BiasPlane) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int i = 0; i < 64; i++)
        EXPECT_FLOAT_EQ(buf[111 * 64 + i], 1.0f);
}

TEST_F(NeuralTest, Encoder_HalfmoveClock) {
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 50 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int i = 0; i < 64; i++)
        EXPECT_FLOAT_EQ(buf[110 * 64 + i], 0.5f);
}

TEST_F(NeuralTest, Encoder_MoveCount) {
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 100");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int i = 0; i < 64; i++)
        EXPECT_FLOAT_EQ(buf[105 * 64 + i], 0.5f);
}

TEST_F(NeuralTest, Encoder_TimeStepsDuplicated) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int t = 1; t < 8; t++) {
        for (int i = 0; i < 13 * 64; i++) {
            EXPECT_FLOAT_EQ(buf[i], buf[t * 13 * 64 + i])
                << "Mismatch at time step " << t << ", offset " << i;
        }
    }
}

TEST_F(NeuralTest, Encoder_RepetitionPlaneZero) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float buf[neural::TENSOR_SIZE] = {};
    neural::encode_position(pos, buf);
    for (int i = 0; i < 64; i++)
        EXPECT_FLOAT_EQ(buf[12 * 64 + i], 0.0f);
}

#ifdef HAS_LIBTORCH
#include "neural/neural_evaluator.h"

// Test model path — generate with:
// python -c "from training.export import export_torchscript; from training.model import ChessNetwork; from training.config import NetworkConfig; import os; os.makedirs('tests/fixtures', exist_ok=True); c=NetworkConfig(num_blocks=1, num_filters=16); m=ChessNetwork(c); export_torchscript(m, 'tests/fixtures/test_model.pt')"
static const char* TEST_MODEL = "tests/fixtures/test_model.pt";

TEST_F(NeuralTest, NeuralEval_LoadModel) {
    ASSERT_NO_THROW(neural::NeuralEvaluator eval(TEST_MODEL, "cpu"));
}

TEST_F(NeuralTest, NeuralEval_StartingPosition) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    ASSERT_EQ(n, 20);

    auto result = eval.evaluate(pos, moves, n);
    EXPECT_EQ(result.policy.size(), 20u);
    EXPECT_GE(result.value, -1.0f);
    EXPECT_LE(result.value, 1.0f);
}

TEST_F(NeuralTest, NeuralEval_PolicySumsToOne) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    auto result = eval.evaluate(pos, moves, n);

    float sum = 0;
    for (float p : result.policy) {
        EXPECT_GT(p, 0.0f);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(NeuralTest, NeuralEval_Checkmate) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");
    Position pos;
    pos.set_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    ASSERT_EQ(n, 0);

    auto result = eval.evaluate(pos, moves, n);
    EXPECT_TRUE(result.policy.empty());
    EXPECT_FLOAT_EQ(result.value, -1.0f);
}

TEST_F(NeuralTest, NeuralEval_DifferentPositions) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");

    Position pos1;
    pos1.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move moves1[MAX_MOVES];
    int n1 = generate_legal_moves(pos1, moves1);
    auto r1 = eval.evaluate(pos1, moves1, n1);

    Position pos2;
    pos2.set_fen("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2");
    Move moves2[MAX_MOVES];
    int n2 = generate_legal_moves(pos2, moves2);
    auto r2 = eval.evaluate(pos2, moves2, n2);

    EXPECT_NE(r1.value, r2.value);
}

TEST_F(NeuralTest, Integration_MCTSWithNeural) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");
    mcts::SearchParams params;
    params.num_iterations = 50;
    params.add_noise = false;

    mcts::Search search(eval, params);
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto result = search.run(pos);

    EXPECT_FALSE(result.best_move.is_none());
    int total_visits = 0;
    for (int v : result.visit_counts) total_visits += v;
    EXPECT_EQ(total_visits, 50);
    EXPECT_GE(result.root_value, -1.0f);
    EXPECT_LE(result.root_value, 1.0f);
}

TEST_F(NeuralTest, BatchEval_StructsCompile) {
    // Verify BatchRequest and BatchResult can be created and used
    float buffer[neural::TENSOR_SIZE] = {0};
    Move moves[] = {Move(E2, E4, FLAG_DOUBLE_PUSH)};
    neural::BatchRequest req;
    req.encoded_planes = buffer;
    req.legal_moves = moves;
    req.num_legal_moves = 1;
    EXPECT_EQ(req.num_legal_moves, 1);

    neural::BatchResult res;
    res.value = 0.5f;
    res.policy = {1.0f};
    res.full_policy.resize(neural::POLICY_SIZE, 0.0f);
    EXPECT_FLOAT_EQ(res.value, 0.5f);
    EXPECT_EQ(res.full_policy.size(), static_cast<size_t>(neural::POLICY_SIZE));
}

TEST_F(NeuralTest, BatchEval_EmptyBatch) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");
    std::vector<neural::BatchRequest> empty;
    auto results = eval.evaluate_batch_raw(empty);
    EXPECT_TRUE(results.empty());
}

TEST_F(NeuralTest, BatchEval_SinglePosition) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    ASSERT_EQ(n, 20);

    // Encode position
    float encoded[neural::TENSOR_SIZE];
    neural::encode_position(pos, encoded);

    neural::BatchRequest req;
    req.encoded_planes = encoded;
    req.legal_moves = moves;
    req.num_legal_moves = n;

    auto results = eval.evaluate_batch_raw({req});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].policy.size(), 20u);
    EXPECT_GE(results[0].value, -1.0f);
    EXPECT_LE(results[0].value, 1.0f);
    EXPECT_EQ(results[0].full_policy.size(), static_cast<size_t>(neural::POLICY_SIZE));

    // Policy should sum to ~1
    float sum = 0;
    for (float p : results[0].policy) {
        EXPECT_GT(p, 0.0f);
        sum += p;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(NeuralTest, BatchEval_MultiplePositions) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");

    // Position 1: starting position
    Position pos1;
    pos1.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move moves1[MAX_MOVES];
    int n1 = generate_legal_moves(pos1, moves1);
    float enc1[neural::TENSOR_SIZE];
    neural::encode_position(pos1, enc1);

    // Position 2: after 1.e4 Nc6
    Position pos2;
    pos2.set_fen("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2");
    Move moves2[MAX_MOVES];
    int n2 = generate_legal_moves(pos2, moves2);
    float enc2[neural::TENSOR_SIZE];
    neural::encode_position(pos2, enc2);

    std::vector<neural::BatchRequest> requests(2);
    requests[0] = {enc1, moves1, n1};
    requests[1] = {enc2, moves2, n2};

    auto results = eval.evaluate_batch_raw(requests);
    ASSERT_EQ(results.size(), 2u);

    // Both should have valid policies
    EXPECT_EQ(results[0].policy.size(), static_cast<size_t>(n1));
    EXPECT_EQ(results[1].policy.size(), static_cast<size_t>(n2));

    // Both should have full policy vectors
    EXPECT_EQ(results[0].full_policy.size(), static_cast<size_t>(neural::POLICY_SIZE));
    EXPECT_EQ(results[1].full_policy.size(), static_cast<size_t>(neural::POLICY_SIZE));

    // Values should differ for different positions
    EXPECT_NE(results[0].value, results[1].value);
}

TEST_F(NeuralTest, BatchEval_MatchesSingleEval) {
    neural::NeuralEvaluator eval(TEST_MODEL, "cpu");
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);

    // Single evaluation
    auto single = eval.evaluate(pos, moves, n);

    // Batch evaluation with same position
    float encoded[neural::TENSOR_SIZE];
    neural::encode_position(pos, encoded);
    neural::BatchRequest req = {encoded, moves, n};
    auto batch = eval.evaluate_batch_raw({req});

    ASSERT_EQ(batch.size(), 1u);
    EXPECT_NEAR(batch[0].value, single.value, 1e-5f);
    ASSERT_EQ(batch[0].policy.size(), single.policy.size());
    for (size_t i = 0; i < single.policy.size(); i++) {
        EXPECT_NEAR(batch[0].policy[i], single.policy[i], 1e-5f)
            << "Policy mismatch at index " << i;
    }
}

#endif // HAS_LIBTORCH
