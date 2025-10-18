#pragma once

#ifdef USE_CUDA
#include <cassert>
#include <type_traits>

#include "neural/activation.h"
#include "neural/cuda/cuda_common.h"

namespace cuda {

void depthwise_conv(float *output, const float *input, const float *weights,
                    const float *biases, const float *residual, const float *mask,
                    int filter_size, int batch, int channels, int height, int width,
                    Activation act, cudaStream_t stream);

} // namespace cuda

#endif
