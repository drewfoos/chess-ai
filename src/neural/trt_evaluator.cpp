#ifdef HAS_TENSORRT

#include "neural/trt_evaluator.h"
#include "neural/encoder.h"
#include "neural/policy_map.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace neural {

namespace {

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

TRTLogger& trt_logger() {
    static TRTLogger logger;
    return logger;
}

int dims_volume(const nvinfer1::Dims& d, int start = 0) {
    int v = 1;
    for (int i = start; i < d.nbDims; i++) v *= static_cast<int>(d.d[i]);
    return v;
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error (") + what + "): " +
                                  cudaGetErrorString(err));
    }
}

} // namespace

TRTEvaluator::TRTEvaluator(const std::string& engine_path,
                           float policy_softmax_temp,
                           int max_batch_size)
    : max_batch_size_(max_batch_size)
    , policy_softmax_temp_(policy_softmax_temp)
    , encode_buffer_(TENSOR_SIZE)
{
    std::ifstream f(engine_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("TRTEvaluator: cannot open engine '" + engine_path + "'");
    }
    f.seekg(0, std::ios::end);
    const auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    f.read(buffer.data(), size);
    f.close();

    runtime_.reset(nvinfer1::createInferRuntime(trt_logger()));
    if (!runtime_) throw std::runtime_error("TRTEvaluator: createInferRuntime failed");

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine_) throw std::runtime_error("TRTEvaluator: deserializeCudaEngine failed");

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("TRTEvaluator: createExecutionContext failed");

    // Discover I/O tensor names and fixed (non-batch) shape components.
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        const char* name = engine_->getIOTensorName(i);
        const auto mode = engine_->getTensorIOMode(name);
        const auto dims = engine_->getTensorShape(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            // Expect (batch, C, H, W). Planes count = C.
            if (dims.nbDims == 4) input_planes_ = static_cast<int>(dims.d[1]);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            // Match by common name first, fall back to shape heuristic.
            std::string sn(name);
            int vol = dims_volume(dims, 1);
            if (sn == "policy" || vol == POLICY_SIZE) {
                policy_name_ = name;
                policy_size_ = vol;
            } else if (sn == "value" || vol == 3) {
                value_name_ = name;
                value_size_ = vol;
            } else if (sn == "mlh") {
                mlh_name_ = name;
                mlh_size_ = std::max(1, vol);
            }
        }
    }
    if (input_name_.empty() || policy_name_.empty() || value_name_.empty()) {
        throw std::runtime_error("TRTEvaluator: engine is missing expected I/O tensors "
                                 "(input, policy, value)");
    }
    if (input_planes_ == 0) input_planes_ = INPUT_PLANES;
    if (policy_size_ == 0) policy_size_ = POLICY_SIZE;
    if (value_size_ == 0) value_size_ = 3;
    // MLH is optional; mlh_name_/mlh_size_ stay empty/0 for legacy engines.

    // Allocate device + pinned host buffers for the maximum batch size.
    const size_t in_bytes  = sizeof(float) * max_batch_size_ * input_planes_ * BOARD_SIZE * BOARD_SIZE;
    const size_t pol_bytes = sizeof(float) * max_batch_size_ * policy_size_;
    const size_t val_bytes = sizeof(float) * max_batch_size_ * value_size_;

    check_cuda(cudaMalloc(&d_input_,  in_bytes),  "cudaMalloc input");
    check_cuda(cudaMalloc(&d_policy_, pol_bytes), "cudaMalloc policy");
    check_cuda(cudaMalloc(&d_value_,  val_bytes), "cudaMalloc value");

    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&h_input_),  in_bytes,  cudaHostAllocDefault),
               "cudaHostAlloc input");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&h_policy_), pol_bytes, cudaHostAllocDefault),
               "cudaHostAlloc policy");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&h_value_),  val_bytes, cudaHostAllocDefault),
               "cudaHostAlloc value");

    if (!mlh_name_.empty()) {
        const size_t mlh_bytes = sizeof(float) * max_batch_size_ * mlh_size_;
        check_cuda(cudaMalloc(&d_mlh_, mlh_bytes), "cudaMalloc mlh");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&h_mlh_), mlh_bytes, cudaHostAllocDefault),
                   "cudaHostAlloc mlh");
    }

    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    context_->setTensorAddress(input_name_.c_str(),  d_input_);
    context_->setTensorAddress(policy_name_.c_str(), d_policy_);
    context_->setTensorAddress(value_name_.c_str(),  d_value_);
    if (!mlh_name_.empty()) {
        context_->setTensorAddress(mlh_name_.c_str(), d_mlh_);
    }
}

TRTEvaluator::~TRTEvaluator() {
    if (stream_)    cudaStreamDestroy(stream_);
    if (d_input_)   cudaFree(d_input_);
    if (d_policy_)  cudaFree(d_policy_);
    if (d_value_)   cudaFree(d_value_);
    if (d_mlh_)     cudaFree(d_mlh_);
    if (h_input_)   cudaFreeHost(h_input_);
    if (h_policy_)  cudaFreeHost(h_policy_);
    if (h_value_)   cudaFreeHost(h_value_);
    if (h_mlh_)     cudaFreeHost(h_mlh_);
}

void TRTEvaluator::infer_batch(int batch_size) {
    // Bind dynamic batch dim for this call.
    nvinfer1::Dims in_dims;
    in_dims.nbDims = 4;
    in_dims.d[0] = batch_size;
    in_dims.d[1] = input_planes_;
    in_dims.d[2] = BOARD_SIZE;
    in_dims.d[3] = BOARD_SIZE;
    if (!context_->setInputShape(input_name_.c_str(), in_dims)) {
        throw std::runtime_error("TRTEvaluator: setInputShape failed");
    }

    const size_t in_bytes  = sizeof(float) * batch_size * input_planes_ * BOARD_SIZE * BOARD_SIZE;
    const size_t pol_bytes = sizeof(float) * batch_size * policy_size_;
    const size_t val_bytes = sizeof(float) * batch_size * value_size_;

    check_cuda(cudaMemcpyAsync(d_input_, h_input_, in_bytes,
                               cudaMemcpyHostToDevice, stream_), "H2D input");
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TRTEvaluator: enqueueV3 failed");
    }
    check_cuda(cudaMemcpyAsync(h_policy_, d_policy_, pol_bytes,
                               cudaMemcpyDeviceToHost, stream_), "D2H policy");
    check_cuda(cudaMemcpyAsync(h_value_, d_value_, val_bytes,
                               cudaMemcpyDeviceToHost, stream_), "D2H value");
    if (!mlh_name_.empty()) {
        const size_t mlh_bytes = sizeof(float) * batch_size * mlh_size_;
        check_cuda(cudaMemcpyAsync(h_mlh_, d_mlh_, mlh_bytes,
                                   cudaMemcpyDeviceToHost, stream_), "D2H mlh");
    }
    check_cuda(cudaStreamSynchronize(stream_), "stream sync");
}

void TRTEvaluator::decode_results(
    int batch_size,
    const std::vector<const Move*>& moves_per_batch,
    const std::vector<int>& num_moves_per_batch,
    const std::vector<Color>& stm_per_batch,
    std::vector<mcts::EvalResult>& out_results)
{
    out_results.resize(batch_size);
    for (int b = 0; b < batch_size; b++) {
        const float* pol_row = h_policy_ + b * policy_size_;
        const float* val_row = h_value_ + b * value_size_;

        // Value head = WDL (win, draw, loss). value = win - loss.
        out_results[b].value = val_row[0] - val_row[2];
        if (h_mlh_) out_results[b].mlh = h_mlh_[b * mlh_size_];

        int num_moves = num_moves_per_batch[b];
        if (num_moves == 0) {
            // Terminal: caller may override; leave value as model's.
            continue;
        }

        Color stm = stm_per_batch[b];
        const Move* moves = moves_per_batch[b];

        std::vector<float> logits(num_moves);
        for (int i = 0; i < num_moves; i++) {
            int idx = move_to_policy_index(moves[i], stm);
            logits[i] = (idx >= 0) ? pol_row[idx] : -1000.0f;
        }
        if (policy_softmax_temp_ != 1.0f) {
            for (int i = 0; i < num_moves; i++) logits[i] /= policy_softmax_temp_;
        }
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        out_results[b].policy.resize(num_moves);
        for (int i = 0; i < num_moves; i++) {
            out_results[b].policy[i] = std::exp(logits[i] - max_logit);
            sum += out_results[b].policy[i];
        }
        if (sum > 0.0f) {
            for (int i = 0; i < num_moves; i++) out_results[b].policy[i] /= sum;
        }
    }
}

mcts::EvalResult TRTEvaluator::evaluate(const Position& pos, const Move* moves, int num_moves) {
    mcts::EvalResult result;
    if (num_moves == 0) {
        result.value = pos.in_check() ? -1.0f : 0.0f;
        return result;
    }

    encode_position(pos, encode_buffer_.data());
    std::memcpy(h_input_, encode_buffer_.data(), TENSOR_SIZE * sizeof(float));
    infer_batch(1);

    std::vector<const Move*> moves_arr{moves};
    std::vector<int> nums{num_moves};
    std::vector<Color> stms{pos.side_to_move()};
    std::vector<mcts::EvalResult> results;
    decode_results(1, moves_arr, nums, stms, results);
    return results[0];
}

std::vector<mcts::EvalResult> TRTEvaluator::evaluate_batch(
    const std::vector<mcts::BatchEvalRequest>& requests)
{
    int batch_size = static_cast<int>(requests.size());
    if (batch_size == 0) return {};
    if (batch_size > max_batch_size_) {
        throw std::runtime_error("TRTEvaluator: batch_size exceeds max_batch_size");
    }
    if (batch_size == 1) {
        return {evaluate(requests[0].position,
                         requests[0].legal_moves,
                         requests[0].num_legal_moves)};
    }

    std::vector<const Move*> moves_arr(batch_size);
    std::vector<int> nums(batch_size);
    std::vector<Color> stms(batch_size);
    for (int b = 0; b < batch_size; b++) {
        encode_position(requests[b].position, h_input_ + b * TENSOR_SIZE);
        moves_arr[b] = requests[b].legal_moves;
        nums[b] = requests[b].num_legal_moves;
        stms[b] = requests[b].position.side_to_move();
    }
    infer_batch(batch_size);

    std::vector<mcts::EvalResult> results;
    decode_results(batch_size, moves_arr, nums, stms, results);
    // Mirror NeuralEvaluator's terminal-position handling for empty move lists.
    for (int b = 0; b < batch_size; b++) {
        if (nums[b] == 0) {
            results[b].value = requests[b].position.in_check() ? -1.0f : 0.0f;
        }
    }
    return results;
}

std::vector<BatchResult> TRTEvaluator::evaluate_batch_raw(
    const std::vector<BatchRequest>& requests)
{
    int batch_size = static_cast<int>(requests.size());
    std::vector<BatchResult> results(batch_size);
    if (batch_size == 0) return results;
    if (batch_size > max_batch_size_) {
        throw std::runtime_error("TRTEvaluator: batch_size exceeds max_batch_size");
    }

    for (int b = 0; b < batch_size; b++) {
        std::memcpy(h_input_ + b * TENSOR_SIZE,
                    requests[b].encoded_planes,
                    TENSOR_SIZE * sizeof(float));
    }
    infer_batch(batch_size);

    for (int b = 0; b < batch_size; b++) {
        const float* pol_row = h_policy_ + b * policy_size_;
        const float* val_row = h_value_ + b * value_size_;
        results[b].value = val_row[0] - val_row[2];
        if (h_mlh_) results[b].mlh = h_mlh_[b * mlh_size_];

        results[b].full_policy.resize(POLICY_SIZE);
        for (int i = 0; i < POLICY_SIZE; i++) {
            results[b].full_policy[i] = pol_row[i];
        }

        int num_moves = requests[b].num_legal_moves;
        const Move* moves = requests[b].legal_moves;
        if (num_moves == 0) continue;

        // Encoded planes layout: plane 104 is color (1.0 = white).
        Color stm = (requests[b].encoded_planes[104 * 64] > 0.5f) ? WHITE : BLACK;

        std::vector<float> logits(num_moves);
        for (int i = 0; i < num_moves; i++) {
            int idx = move_to_policy_index(moves[i], stm);
            logits[i] = (idx >= 0) ? pol_row[idx] : -1000.0f;
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

} // namespace neural

#endif // HAS_TENSORRT
