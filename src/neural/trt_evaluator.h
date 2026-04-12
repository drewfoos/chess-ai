#pragma once
#ifdef HAS_TENSORRT

#include "mcts/search.h"
#include "neural/neural_evaluator.h"  // reuse BatchRequest / BatchResult

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace neural {

// TensorRT-backed evaluator with the same interface as NeuralEvaluator.
// Selected via --use-trt at CLI / SearchEngine construction time.
class TRTEvaluator : public mcts::Evaluator, public RawBatchEvaluator {
public:
    TRTEvaluator(const std::string& engine_path,
                 float policy_softmax_temp = 2.2f,
                 int max_batch_size = 256);
    ~TRTEvaluator();

    TRTEvaluator(const TRTEvaluator&) = delete;
    TRTEvaluator& operator=(const TRTEvaluator&) = delete;

    mcts::EvalResult evaluate(const Position& pos, const Move* moves, int num_moves) override;
    std::vector<mcts::EvalResult> evaluate_batch(
        const std::vector<mcts::BatchEvalRequest>& requests) override;

    // Shared-shape batch path mirroring NeuralEvaluator for GameManager.
    std::vector<BatchResult> evaluate_batch_raw(const std::vector<BatchRequest>& requests) override;

private:
    void infer_batch(int batch_size);
    void decode_results(int batch_size,
                        const std::vector<const Move*>& moves_per_batch,
                        const std::vector<int>& num_moves_per_batch,
                        const std::vector<Color>& stm_per_batch,
                        std::vector<mcts::EvalResult>& out_results);

    struct TrtDeleter { template<class T> void operator()(T* p) const { if (p) delete p; } };
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

    int max_batch_size_;
    float policy_softmax_temp_;

    // Tensor name strings (bindings) discovered at construction time.
    std::string input_name_;
    std::string policy_name_;
    std::string value_name_;
    std::string mlh_name_;   // Empty when engine lacks the moves-left head output.

    int policy_size_ = 0;  // e.g. 1858
    int value_size_ = 0;   // e.g. 3
    int mlh_size_ = 0;     // Typically 1; 0 when MLH output is absent.
    int input_planes_ = 0;

    // Device-side buffers (allocated once for max_batch_size).
    void* d_input_ = nullptr;
    void* d_policy_ = nullptr;
    void* d_value_ = nullptr;
    void* d_mlh_ = nullptr;   // nullptr when MLH output is absent.

    // Host-side staging (pinned) — async DMA to/from device.
    float* h_input_ = nullptr;
    float* h_policy_ = nullptr;
    float* h_value_ = nullptr;
    float* h_mlh_ = nullptr;  // nullptr when MLH output is absent.

    cudaStream_t stream_ = nullptr;

    // Reusable encode buffer for single-position evaluate().
    std::vector<float> encode_buffer_;
};

} // namespace neural

#endif // HAS_TENSORRT
