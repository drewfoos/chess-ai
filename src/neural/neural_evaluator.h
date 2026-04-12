#pragma once
#ifdef HAS_LIBTORCH

#include "mcts/search.h"
#include <torch/script.h>
#include <string>
#include <vector>

namespace neural {

class NeuralEvaluator : public mcts::Evaluator {
public:
    explicit NeuralEvaluator(const std::string& model_path, const std::string& device = "cpu");
    mcts::EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) override;

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    std::vector<float> encode_buffer_;
};

} // namespace neural

#endif
