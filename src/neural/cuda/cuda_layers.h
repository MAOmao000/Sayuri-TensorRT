#pragma once

#ifdef USE_CUDA
#include "neural/cuda/cuda_common.h"
#include "neural/activation.h"

#include <vector>
#include <array>

namespace cuda {

class LayerBasic {
protected:
    CudaHandles *handles_{nullptr};
    bool loaded_{false};
    bool fp16_{false};
    Activation act_{Activation::kIdentity};

    int maxbatch_{0};
    int width_{0};
    int height_{0};
    int spatial_size_{0};
};

class Convolution : public LayerBasic {
public:
    Convolution() = default;
    Convolution(CudaHandles *handles, const int batch,
                const int board_size, const int filter,
                const int in_channels, const int out_channels,
                Activation act);
    ~Convolution();

    void LoadWeights(const std::vector<float> &weights);

    void LoadWeights(const std::vector<float> &weights,
                     const std::vector<float> &biases);

    void* GetDevWeights() {
        return cuda_weights_;
    }

    void* GetDevBiases() {
        return cuda_biases_;
    }

private:
    int filter_dim_;
    int filters_;
    int in_channels_;
    int out_channels_;

    void *cuda_weights_;
    void *cuda_biases_{nullptr};
};

class DepthwiseConvolution : public LayerBasic {
public:
    DepthwiseConvolution() = default;
    DepthwiseConvolution(CudaHandles *handles, const int batch,
                         const int board_size, const int filter,
                         const int channels, Activation act);
    ~DepthwiseConvolution();

    void LoadWeights(const std::vector<float> &weights);

    void LoadWeights(const std::vector<float> &weights,
                     const std::vector<float> &biases);

    void* GetDevWeights() {
        return cuda_weights_;
    }

    void* GetDevBiases() {
        return cuda_biases_;
    }

private:
    int filters_;
    int channels_;

    void *cuda_weights_;
    void *cuda_biases_{nullptr};
};

class FullyConnect : public LayerBasic {
public:
    FullyConnect() = default;
    FullyConnect(CudaHandles *handles,
                 const int batch,
                 const int inputs,
                 const int outputs,
                 Activation act);
    ~FullyConnect();

    void LoadWeights(const std::vector<float> &weights,
                     const std::vector<float> &biases);

    void* GetDevWeights() {
        return cuda_weights_;
    }

    void* GetDevBiases() {
        return cuda_biases_;
    }

private:
    int inputs_;
    int outputs_;

    void *cuda_weights_;
    void *cuda_biases_;
};

class GlobalPooling : public LayerBasic {
public:
    GlobalPooling() = default;
    GlobalPooling(CudaHandles *handles,
                  bool is_value_head,
                  const int batch,
                  const int board_size,
                  const int channels);

private:
    int channels_;
    bool is_value_head_;
};

class SEUnit : public LayerBasic {
public:
    SEUnit() = default;
    SEUnit(CudaHandles *handles,
           const int batch,
           const int board_size,
           const int channels,
           const int se_size,
           Activation act);
    ~SEUnit();

    void LoadWeights(const std::vector<float> &weights_w1,
                     const std::vector<float> &weights_b1,
                     const std::vector<float> &weights_w2,
                     const std::vector<float> &weights_b2);

    void* GetDevSqueezeWeights() {
        return cuda_weights_w1_;
    }

    void* GetDevSqueezeBiases() {
        return cuda_weights_b1_;
    }

    void* GetDevExciteWeights() {
        return cuda_weights_w2_;
    }

    void* GetDevExciteBiases() {
        return cuda_weights_b2_;
    }

private:
    int se_size_;
    int channels_;

    void *cuda_weights_w1_;
    void *cuda_weights_b1_;
    void *cuda_weights_w2_;
    void *cuda_weights_b2_;
};

} // namespace cuda

#endif
