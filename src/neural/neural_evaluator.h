#pragma once
#ifdef HAS_LIBTORCH

#include "mcts/search.h"
#include <torch/script.h>
#include <string>
#include <vector>

namespace neural {

// NOT thread-safe: encode_buffer_ is reused across calls.
// For multi-threaded MCTS, use one evaluator per thread or add synchronization.
class NeuralEvaluator : public mcts::Evaluator {
public:
    explicit NeuralEvaluator(const std::string& model_path, const std::string& device = "cpu",
                             float policy_softmax_temp = 2.2f);
    mcts::EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) override;

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    float policy_softmax_temp_;
    std::vector<float> encode_buffer_;
};

} // namespace neural

#endif
