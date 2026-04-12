#ifdef HAS_LIBTORCH

#include "neural/neural_evaluator.h"
#include "neural/encoder.h"
#include "neural/policy_map.h"
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>

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

    // Create input tensor and transfer to device
    // encode_buffer_ is a class member that outlives this call, so from_blob is safe
    auto input = torch::from_blob(
        encode_buffer_.data(),
        {1, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE},
        torch::kFloat32
    ).to(device_);

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

std::vector<mcts::EvalResult> NeuralEvaluator::evaluate_batch(
    const std::vector<mcts::BatchEvalRequest>& requests) {

    int batch_size = static_cast<int>(requests.size());
    if (batch_size == 0) return {};
    if (batch_size == 1) {
        return {evaluate(requests[0].position, requests[0].legal_moves, requests[0].num_legal_moves)};
    }

    // Encode all positions into one contiguous buffer
    batch_buffer_.resize(batch_size * TENSOR_SIZE);
    for (int b = 0; b < batch_size; b++) {
        encode_position(requests[b].position, batch_buffer_.data() + b * TENSOR_SIZE);
    }

    // Single GPU forward pass
    auto input = torch::from_blob(
        batch_buffer_.data(),
        {batch_size, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE},
        torch::kFloat32
    ).to(device_);

    torch::NoGradGuard no_grad;
    auto output = model_.forward({input}).toTuple();
    auto policy_logits = output->elements()[0].toTensor().to(torch::kCPU);
    auto wdl_probs = output->elements()[1].toTensor().to(torch::kCPU);

    auto policy_acc = policy_logits.accessor<float, 2>();
    auto wdl_acc = wdl_probs.accessor<float, 2>();

    std::vector<mcts::EvalResult> results(batch_size);
    for (int b = 0; b < batch_size; b++) {
        results[b].value = wdl_acc[b][0] - wdl_acc[b][2];

        int num_moves = requests[b].num_legal_moves;
        if (num_moves == 0) {
            if (requests[b].position.in_check()) results[b].value = -1.0f;
            else results[b].value = 0.0f;
            continue;
        }

        Color stm = requests[b].position.side_to_move();
        std::vector<float> logits(num_moves);
        for (int i = 0; i < num_moves; i++) {
            int idx = move_to_policy_index(requests[b].legal_moves[i], stm);
            logits[i] = (idx >= 0) ? policy_acc[b][idx] : -1000.0f;
        }

        if (policy_softmax_temp_ != 1.0f) {
            for (int i = 0; i < num_moves; i++) logits[i] /= policy_softmax_temp_;
        }

        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        results[b].policy.resize(num_moves);
        for (int i = 0; i < num_moves; i++) {
            results[b].policy[i] = std::exp(logits[i] - max_logit);
            sum += results[b].policy[i];
        }
        if (sum > 0.0f) {
            for (int i = 0; i < num_moves; i++) results[b].policy[i] /= sum;
        }
    }
    return results;
}

std::vector<BatchResult> NeuralEvaluator::evaluate_batch_raw(
    const std::vector<BatchRequest>& requests) {

    int batch_size = static_cast<int>(requests.size());
    std::vector<BatchResult> results(batch_size);

    if (batch_size == 0) return results;

    // Stack all encoded planes into one contiguous buffer
    batch_buffer_.resize(batch_size * TENSOR_SIZE);
    for (int b = 0; b < batch_size; b++) {
        std::memcpy(
            batch_buffer_.data() + b * TENSOR_SIZE,
            requests[b].encoded_planes,
            TENSOR_SIZE * sizeof(float)
        );
    }

    // Create batch input tensor and transfer to device
    // batch_buffer_ is a class member that outlives this call, so from_blob is safe
    auto input = torch::from_blob(
        batch_buffer_.data(),
        {batch_size, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE},
        torch::kFloat32
    ).to(device_);

    // Run inference
    torch::NoGradGuard no_grad;
    auto output = model_.forward({input}).toTuple();
    auto policy_logits = output->elements()[0].toTensor().to(torch::kCPU);  // (B, 1858)
    auto wdl_probs = output->elements()[1].toTensor().to(torch::kCPU);      // (B, 3)

    auto policy_acc = policy_logits.accessor<float, 2>();
    auto wdl_acc = wdl_probs.accessor<float, 2>();

    // Process each result
    for (int b = 0; b < batch_size; b++) {
        // Value: win - loss
        results[b].value = wdl_acc[b][0] - wdl_acc[b][2];

        // Full policy (1858-dim, before masking to legal moves)
        results[b].full_policy.resize(POLICY_SIZE);
        for (int i = 0; i < POLICY_SIZE; i++) {
            results[b].full_policy[i] = policy_acc[b][i];
        }

        // Policy: extract logits for legal moves, apply PST, softmax
        int num_moves = requests[b].num_legal_moves;
        const Move* moves = requests[b].legal_moves;

        if (num_moves == 0) {
            // Terminal position — handled by caller
            continue;
        }

        // Determine side to move from encoded planes.
        // Plane 104 is the color plane: 1.0 = white, 0.0 = black.
        Color stm = (requests[b].encoded_planes[104 * 64] > 0.5f) ? WHITE : BLACK;

        std::vector<float> logits(num_moves);
        for (int i = 0; i < num_moves; i++) {
            int idx = move_to_policy_index(moves[i], stm);
            logits[i] = (idx >= 0) ? policy_acc[b][idx] : -1000.0f;
        }

        // Apply policy softmax temperature
        if (policy_softmax_temp_ != 1.0f) {
            for (int i = 0; i < num_moves; i++) {
                logits[i] /= policy_softmax_temp_;
            }
        }

        // Softmax over legal moves
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        results[b].policy.resize(num_moves);
        for (int i = 0; i < num_moves; i++) {
            results[b].policy[i] = std::exp(logits[i] - max_logit);
            sum += results[b].policy[i];
        }
        if (sum > 0.0f) {
            for (int i = 0; i < num_moves; i++) {
                results[b].policy[i] /= sum;
            }
        }
    }

    return results;
}

} // namespace neural

#endif // HAS_LIBTORCH
