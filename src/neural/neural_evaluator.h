#pragma once
#ifdef HAS_LIBTORCH

#include "mcts/search.h"
#include <torch/script.h>
#include <string>
#include <vector>

namespace neural {

// Batch evaluation request — one per position
struct BatchRequest {
    const float* encoded_planes;  // Pre-encoded 112*8*8 float buffer
    const Move* legal_moves;      // Array of legal moves
    int num_legal_moves;          // Length of legal_moves array
};

// Batch evaluation result — one per position
struct BatchResult {
    std::vector<float> policy;       // Per legal move priors (normalized)
    float value;                     // Win - Loss scalar
    std::vector<float> full_policy;  // Full 1858-dim policy (for raw_policy tracking)
};

// NOT thread-safe: encode_buffer_ is reused across calls.
// For multi-threaded MCTS, use one evaluator per thread or add synchronization.
class NeuralEvaluator : public mcts::Evaluator {
public:
    explicit NeuralEvaluator(const std::string& model_path, const std::string& device = "cpu",
                             float policy_softmax_temp = 2.2f);
    mcts::EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) override;

    // Batch evaluation via Evaluator interface — single GPU forward pass
    std::vector<mcts::EvalResult> evaluate_batch(const std::vector<mcts::BatchEvalRequest>& requests) override;

    // Batch evaluation with full policy output (for training data)
    // Callers must pre-encode positions into encoded_planes buffers.
    std::vector<BatchResult> evaluate_batch_raw(const std::vector<BatchRequest>& requests);

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    float policy_softmax_temp_;
    std::vector<float> encode_buffer_;
    std::vector<float> batch_buffer_;  // Resizable buffer for batch encoding
};

} // namespace neural

#endif
