#pragma once

#ifdef USE_CUDA
#include <atomic>
#include <memory>
#include <list>
#include <array>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdarg.h>
#include <fstream>
#include <numeric>
#include <map>
#include <cassert>

#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_layers.h"
#include "neural/activation.h"
#include "neural/network_basic.h"
#include "neural/description.h"
#include "utils/threadpool.h"
#include "utils/half.h"

using namespace nvinfer1;

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

static std::string vformat(const char *fmt, va_list ap) {
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.  Be prepared to allocate dynamically if it doesn't fit.
    size_t size = 4096;
    char stackbuf[4096];
    std::vector<char> dynamicbuf;
    char *buf = &stackbuf[0];

    int needed;
    while (true) {
        // Try to vsnprintf into our buffer.
        needed = vsnprintf(buf, size, fmt, ap);
        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.

        if (needed <= (int)size && needed >= 0)
            break;

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        size = (needed > 0) ? (needed+1) : (size*2);
        dynamicbuf.resize(size+1);
        buf = &dynamicbuf[0];
    }
    return std::string(buf, (size_t)needed);
}

inline std::string strprintf(const char* fmt, ...) {
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat(fmt, ap);
    va_end (ap);
    return buf;
}

inline std::string readFileBinary(
    const std::string& filename) {
    std::ifstream ifs;
    ifs.open(filename, std::ios::binary);
    std::string str((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    return str;
}

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

class CudaForwardPipe : public NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &input);

    virtual bool Valid() const;

    virtual void Construct(ForwardPipeOption option,
                           std::shared_ptr<DNNWeights> weights);

    virtual void Release();

    virtual void Destroy();

private:

    class BackendContext {
    public:
        std::unique_ptr<nvinfer1::IExecutionContext> execution_context_{nullptr};
        std::map<std::string, void*> buffers_;
    };

    class NNGraph {
        struct Block {
            // TODO: Use list to store all conv.
            cuda::DepthwiseConvolution dw_conv;
            cuda::Convolution pre_btl_conv;
            cuda::Convolution conv1;
            cuda::Convolution conv2;
            cuda::Convolution conv3;
            cuda::Convolution conv4;
            cuda::Convolution post_btl_conv;
            cuda::SEUnit se_module;
        };

        struct Graph {
            // intput
            cuda::Convolution input_conv;

            // block tower
            std::vector<NNGraph::Block> tower;

            // policy head
            cuda::Convolution p_hd_conv;
            cuda::DepthwiseConvolution p_dw_conv;
            cuda::Convolution p_pt_conv;
            cuda::GlobalPooling p_pool;
            cuda::FullyConnect p_inter;

            cuda::Convolution p_prob;
            cuda::FullyConnect p_prob_pass;

            // value head
            cuda::Convolution v_hd_conv;
            cuda::GlobalPooling v_pool;
            cuda::FullyConnect v_inter;

            cuda::Convolution v_ownership;
            cuda::FullyConnect v_misc;
        };

    public:
        NNGraph() {}
        ~NNGraph();
        void ConstructGraph(bool dump_gpu_info,
                            const int gpu,
                            const int max_batch_size,
                            const int board_size,
                            std::shared_ptr<DNNWeights> weights);

        bool build(bool dump_gpu_info,
                            const int gpu,
                            const int max_batch_size,
                            const int board_size,
                            std::shared_ptr<DNNWeights> weights);

        std::vector<OutputResult> BatchForward(const std::vector<InputData> &input);

        void DestroyGraph();

    private:
        void SetComputationMode(cuda::CudaHandles *handles);

        bool ApplyMask(const std::vector<InputData> &input);

        void FillOutputs(const std::vector<float> &batch_prob,
                         const std::vector<float> &batch_prob_pass,
                         const std::vector<float> &batch_value_misc,
                         const std::vector<float> &batch_ownership,
                         const std::vector<InputData> &batch_input,
                         std::vector<OutputResult> &batch_output_result);

        // Create full model using the TensorRT network definition API and build the engine.
        bool constructNetwork(
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network
        );

        ILayer* buildResidualBlock(
            ITensor* input,
            BlockBasic* tower_ptr,
            NNGraph::Block* block_ptr,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

        ILayer* buildBottleneckBlock(
            ITensor* input,
            BlockBasic* tower_ptr,
            NNGraph::Block* block_ptr,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

        ILayer* buildNestedBottleneckBlock(
            ITensor* input,
            BlockBasic* tower_ptr,
            NNGraph::Block* block_ptr,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

        ILayer* buildMixerBlock(
            ITensor* input,
            BlockBasic* tower_ptr,
            NNGraph::Block* block_ptr,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

        ILayer* buildSqueezeExcitationLayer(
            ITensor* residual,
            ITensor* input,
            BlockBasic* tower_ptr,
            NNGraph::Block* block_ptr,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

        void buildPolicyHead(
            ITensor* input,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

        void buildValueHead(
            ITensor* input,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network);

        nvinfer1::ILayer* buildConvLayer(
            nvinfer1::ITensor* input,
            unsigned int filter_size,
            int64_t weights_size,
            void* weights,
            int64_t biases_size,
            void* biases,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            unsigned int outputs,
            const bool depth_wise = false
        );

        nvinfer1::ILayer* buildActivationLayer(
            nvinfer1::ITensor* input,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            const bool needMask = true
        );

        nvinfer1::ILayer* applyGPoolLayer(
            nvinfer1::ITensor* input,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            const bool isValueHead = false
        );

        nvinfer1::ILayer* applyMaskLayer(
            nvinfer1::ITensor* input,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network
        );

        cuda::CudaHandles handles_;

        int board_size_{0};
        int max_batch_;

        std::unique_ptr<Graph> graph_{nullptr};

        void *host_input_planes_;
        void *host_output_prob_;
        void *host_output_prob_pass_;
        void *host_output_val_;
        void *host_output_ownership_;

        void *cuda_input_planes_;
        void *cuda_output_prob_;
        void *cuda_output_prob_pass_;
        void *cuda_output_val_;
        void *cuda_output_ownership_;

        std::shared_ptr<DNNWeights> weights_{nullptr};
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<BackendContext> context_;

        nvinfer1::ITensor* inputMask_{nullptr};
        nvinfer1::ILayer* maskSumLayer_{nullptr};
        nvinfer1::ILayer* maskScaleLayer_{nullptr};
        nvinfer1::ILayer* maskQuadLayer_{nullptr};
        nvinfer1::ICastLayer* shapeLayer_{nullptr};

        std::vector<std::unique_ptr<float[]>> extraWeights_;
        std::vector<std::unique_ptr<half_float_t[]>> extrahalfWeights_;
    };

    struct ForwawrdEntry {
	    const InputData &input;
        OutputResult &output;

        std::condition_variable cv;
        std::mutex mutex;

        ForwawrdEntry(const InputData &in,
                      OutputResult &out) :
                      input(in), output(out) {}
    };

    std::list<std::shared_ptr<ForwawrdEntry>> entry_queue_;
    std::mutex worker_mutex_;
    std::mutex queue_mutex_;
    std::mutex io_mutex_;

    std::condition_variable cv_;

    std::atomic<int> waittime_{0};
    std::atomic<bool> worker_running_;

    std::vector<std::unique_ptr<NNGraph>> nngraphs_;
    std::unique_ptr<ThreadGroup<void>> group_;

    bool dump_gpu_info_;
    int max_batch_per_nn_{0};
    int forwarding_batch_per_nn_{0};
    int board_size_{0};

    void AssignWorkers();
    void Worker(int gpu);
    void QuitWorkers();
};
#endif
