#ifdef USE_CUDA

#include "neural/cuda/cuda_layers.h"
#include "utils/half.h"

#include <cassert>
#include <iostream>
#include <algorithm>

namespace cuda {

Convolution::Convolution(CudaHandles *handles,
                         const int max_batch,
                         const int board_size,
                         const int filter_size,
                         const int input_channels,
                         const int output_channels,
                         Activation act) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;

    in_channels_ = input_channels;
    out_channels_ = output_channels;
    filters_ = filter_size;
    filter_dim_ = filters_ * filters_ * in_channels_;
    maxbatch_ = max_batch;

    handles_ = handles;

    fp16_ = handles->fp16;
    loaded_ = false;
    act_ = act;

}

Convolution::~Convolution() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_));

        if (cuda_biases_) {
            ReportCUDAErrors(cudaFree(cuda_biases_));
        }
    }
}

void Convolution::LoadWeights(const std::vector<float> &weights) {

    if (loaded_) {
        return;
    }
    assert((int)weights.size() == filter_dim_ * out_channels_);

    std::vector<float> weights_copy = weights;

    MallocAndCopy(fp16_, &cuda_weights_, weights_copy);

    loaded_ = true;
}

void Convolution::LoadWeights(const std::vector<float> &weights,
                              const std::vector<float> &biases) {
    if (loaded_) {
        return;
    }

    MallocAndCopy(fp16_, &cuda_biases_, biases);
    LoadWeights(weights);
}

DepthwiseConvolution::DepthwiseConvolution(CudaHandles *handles,
                                           const int max_batch,
                                           const int board_size,
                                           const int filter_size,
                                           const int channels,
                                           Activation act) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;

    channels_ = channels;
    filters_ = filter_size;
    maxbatch_ = max_batch;

    handles_ = handles;

    fp16_ = handles->fp16;
    loaded_ = false;
    act_ = act;
}

DepthwiseConvolution::~DepthwiseConvolution() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_));
        if (cuda_biases_) {
            ReportCUDAErrors(cudaFree(cuda_biases_));
        }
    }
}

void DepthwiseConvolution::LoadWeights(const std::vector<float> &weights,
                                       const std::vector<float> &biases) {
    if (loaded_) {
        return;
    }
    MallocAndCopy(fp16_, &cuda_biases_, biases);
    LoadWeights(weights);
}

void DepthwiseConvolution::LoadWeights(const std::vector<float> &weights) {
    if (loaded_) {
        return;
    }
    MallocAndCopy(fp16_, &cuda_weights_, weights);
    loaded_ = true;
}

FullyConnect::FullyConnect(CudaHandles *handles,
                           const int max_batch,
                           const int inputs,
                           const int outputs,
                           Activation act) {
    maxbatch_ = max_batch;
    inputs_ = inputs;
    outputs_ = outputs;
    fp16_ = handles->fp16;
    loaded_ = false;
    act_ = act;
    handles_ = handles;
}

FullyConnect::~FullyConnect() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_));
        ReportCUDAErrors(cudaFree(cuda_biases_));
    }
}

void FullyConnect::LoadWeights(const std::vector<float> &weights,
                               const std::vector<float> &biases) {
    if (loaded_) {
        return;
    }
    MallocAndCopy(fp16_, &cuda_weights_, weights);
    MallocAndCopy(fp16_, &cuda_biases_, biases);
    loaded_ = true;
}

GlobalPooling::GlobalPooling(CudaHandles *handles,
                             bool is_value_head,
                             const int max_batch,
                             const int board_size,
                             const int channels) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;
    is_value_head_ = is_value_head;

    fp16_ = handles->fp16;
    maxbatch_ = max_batch;
    channels_ = channels;
    handles_ = handles;
}

SEUnit::SEUnit(CudaHandles *handles,
               const int max_batch,
               const int board_size,
               const int channels,
               const int se_size,
               Activation act) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;
    act_ = act;

    fp16_ = handles->fp16;
    se_size_ = se_size;
    maxbatch_ = max_batch;
    channels_ = channels;
    loaded_ = false;
    handles_ = handles;
}

void SEUnit::LoadWeights(const std::vector<float> &weights_w1,
                         const std::vector<float> &weights_b1,
                         const std::vector<float> &weights_w2,
                         const std::vector<float> &weights_b2) {
    if (loaded_) {
        return;
    }
    MallocAndCopy(fp16_, &cuda_weights_w1_, weights_w1);
    MallocAndCopy(fp16_, &cuda_weights_b1_, weights_b1);
    MallocAndCopy(fp16_, &cuda_weights_w2_, weights_w2);
    MallocAndCopy(fp16_, &cuda_weights_b2_, weights_b2);

    const size_t fc1_scratch_size  = maxbatch_ * se_size_;
    const size_t fc2_scratch_size  = maxbatch_ * 2 * channels_;
    const size_t pool_scratch_size = maxbatch_ * 3 * channels_;

    loaded_ = true;
}

SEUnit::~SEUnit() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_w1_));
        ReportCUDAErrors(cudaFree(cuda_weights_b1_));
        ReportCUDAErrors(cudaFree(cuda_weights_w2_));
        ReportCUDAErrors(cudaFree(cuda_weights_b2_));
    }
}

} // namespace cuda

#endif
