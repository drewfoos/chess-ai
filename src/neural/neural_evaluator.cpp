#ifdef HAS_LIBTORCH

#include "neural/neural_evaluator.h"
#include "neural/encoder.h"
#include "neural/policy_map.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>

namespace neural {

NeuralEvaluator::NeuralEvaluator(const std::string& model_path, const std::string& device,
                                 float policy_softmax_temp)
    : device_(device == "cuda" && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
    , policy_softmax_temp_(policy_softmax_temp)
    , encode_buffer_(TENSOR_SIZE)
{
    try {
        model_ = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "NeuralEvaluator: failed to load model from '" + model_path + "': " + e.what());
    }
    model_.to(device_);
    model_.eval();
}

mcts::EvalResult NeuralEvaluator::evaluate(const Position& pos, const Move* moves, int num_moves) {
    mcts::EvalResult result;

    // Terminal positions
    if (num_moves == 0) {
        if (pos.in_check()) {
            result.value = -1.0f;  // Checkmate
        } else {
            result.value = 0.0f;   // Stalemate
        }
        return result;
    }

    // Encode position
    encode_position(pos, encode_buffer_.data());

    // Create input tensor (clone because from_blob doesn't own the memory)
    auto input = torch::from_blob(
        encode_buffer_.data(),
        {1, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE},
        torch::kFloat32
    ).clone().to(device_);

    // Run inference
    torch::NoGradGuard no_grad;
    auto output = model_.forward({input}).toTuple();
    auto policy_logits = output->elements()[0].toTensor().to(torch::kCPU);  // (1, 1858)
    auto wdl_probs = output->elements()[1].toTensor().to(torch::kCPU);      // (1, 3)

    // Value: win - loss
    auto wdl_acc = wdl_probs.accessor<float, 2>();
    result.value = wdl_acc[0][0] - wdl_acc[0][2];

    // Policy: extract logits for legal moves, then softmax
    auto policy_acc = policy_logits.accessor<float, 2>();
    std::vector<float> logits(num_moves);
    for (int i = 0; i < num_moves; i++) {
        int idx = move_to_policy_index(moves[i], pos.side_to_move());
        logits[i] = (idx >= 0) ? policy_acc[0][idx] : -1000.0f;
    }

    // Apply policy softmax temperature (>1.0 widens the distribution)
    if (policy_softmax_temp_ != 1.0f) {
        for (int i = 0; i < num_moves; i++) {
            logits[i] /= policy_softmax_temp_;
        }
    }

    // Softmax over legal moves
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    result.policy.resize(num_moves);
    for (int i = 0; i < num_moves; i++) {
        result.policy[i] = std::exp(logits[i] - max_logit);
        sum += result.policy[i];
    }
    for (int i = 0; i < num_moves; i++) {
        result.policy[i] /= sum;
    }

    return result;
}

} // namespace neural

#endif // HAS_LIBTORCH
