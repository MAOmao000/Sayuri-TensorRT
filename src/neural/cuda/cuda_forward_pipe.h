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

#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_layers.h"
#include "neural/activation.h"
#include "neural/network_basic.h"
#include "neural/description.h"
#include "utils/threadpool.h"
#include "utils/half.h"

constexpr auto SIGMOID = 99;

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
        bool m_buffers_allocated{false};
        std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
        std::map<std::string, void*> mBuffers;
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
        NNGraph(std::mutex &mtx) : io_mutex_(mtx) {}
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

        void FillOutputs(const std::vector<float> &batch_prob,
                         const std::vector<float> &batch_prob_pass,
                         const std::vector<float> &batch_value_misc,
                         const std::vector<float> &batch_ownership,
                         const std::vector<InputData> &batch_input,
                         std::vector<OutputResult> &batch_output_result);

        // Create full model using the TensorRT network definition API and build the engine.
        bool constructNetwork(
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            std::string& tune_desc,
            const int board_size
        );

        nvinfer1::ITensor* initInputs(
            char const *inputName,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            const int channels,
            const int rows,
            const int cols
        );

        nvinfer1::ILayer* buildConvLayer(
            nvinfer1::ITensor* input,
            unsigned int filter_size,
            int64_t weights_size,
            void* weights,
            int64_t biases_size,
            void* biases,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            std::string& tune_desc,
            std::string op_name,
            unsigned int outputs,
            const int groups = 1
        );

        nvinfer1::ILayer* buildActivationLayer(
            nvinfer1::ITensor* input,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            std::string& tune_desc,
            std::string op_name,
            const int act
        );

        nvinfer1::ILayer* applyGPoolLayer(
            nvinfer1::ITensor* input,
            nvinfer1::IConstantLayer* maskScaleLayer,
            nvinfer1::IConstantLayer* maskQuadLayer,
            TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
            const bool isValueHead = false
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

        std::array<void*, 2> host_mask_op_;

        std::array<void*, 2> cuda_scratch_op_;
        std::array<void*, 4> cuda_conv_op_;
        std::array<void*, 3> cuda_pol_op_;
        std::array<void*, 3> cuda_val_op_;
        std::array<void*, 2> cuda_mask_op_;

        std::mutex &io_mutex_;

        size_t scratch_size_;
        std::shared_ptr<DNNWeights> weights_{nullptr};
        std::unique_ptr<nvinfer1::IRuntime> mRuntime;
        std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
        std::unique_ptr<BackendContext> m_context;

        std::vector<std::unique_ptr<float[]>> extraWeights;
        std::vector<std::unique_ptr<half_float_t[]>> extrahalfWeights;
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
