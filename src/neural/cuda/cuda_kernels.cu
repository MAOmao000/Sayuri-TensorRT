#ifdef USE_PLUGIN
#include "neural/cuda/cuda_kernels.h"
//#include "game/types.h"

#ifndef NDEBUG
#include <stdio.h>
#include <stdlib.h>

void hexdump(const void* data, size_t size) 
{
    char ascii[17];
    size_t i, j;
    ascii[16] = '\0';
    for (i = 0; i < size; ++i) {
        printf("%02X ", ((unsigned char*)data)[i]);
        if (((unsigned char*)data)[i] >= ' ' && ((unsigned char*)data)[i] <= '~') {
            ascii[i % 16] = ((unsigned char*)data)[i];
        } else {
            ascii[i % 16] = '.';
        }
        if ((i+1) % 8 == 0 || i+1 == size) {
            printf(" ");
            if ((i+1) % 16 == 0) {
                printf("|  %s \n", ascii);
            } else if (i+1 == size) {
                ascii[(i+1) % 16] = '\0';
                if ((i+1) % 16 <= 8) {
                    printf(" ");
                }
                for (j = (i+1) % 16; j < 16; ++j) {
                    printf("   ");
                }
                printf("|  %s \n", ascii);
            }
        }
    }
}
#endif

#ifdef USE_CUDA

namespace cuda {

__global__ void depthwise_conv_kernel(float *output, const float *input, const float *weights,
                                      const float *biases, const float *residual, const float *mask,
                                      int filter_size, int N, int C, int H, int W, Activation act) {
    int total_elements = N * C * H * W;
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < total_elements) {
        int filter_dim = filter_size * filter_size;
        int spatial = H * W;
        int nc_index = index / spatial;
        int n_index = nc_index / C;
        int c_index = nc_index % C;
        int hw_index = index % spatial;
        int pad = filter_size / 2;
        int h_in = hw_index / W - pad;
        int w_in = hw_index % W - pad;

        float val = 0.f;
        #pragma unroll
        for (int i = 0; i < filter_size; ++i) {
            #pragma unroll
            for (int j = 0; j < filter_size; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                if (h >= 0 && w >= 0 && h < H && w < W) {
                    val += (float)(weights[i * filter_size + j + c_index * filter_dim]) *
                               (float)(input[nc_index * spatial + h * W + w]);
                }
            }
        }
        if (biases) {
            val += (float)(biases[c_index]);
        }

        ACTIVATION_FUNC(val, act);

        if (residual) {
            val += (float)(residual[index]);
        }
        if (mask) {
            val *= (float)(mask[n_index * spatial + hw_index]);
        }
        output[index] = val;
    }
}

void depthwise_conv(float *output, const float *input, const float *weights,
                    const float *biases, const float *residual, const float *mask,
                    int filter_size, int batch, int channels, int height, int width,
                    Activation act, cudaStream_t stream) {
    const int total_elements = batch * channels * height * width;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

/*
#ifndef NDEBUG
    auto tmp1 = std::vector<float>(batch * height * width);
    ReportCUDAErrors(cudaMemcpy(
        &tmp1[0],
        mask,
        batch * height * width * sizeof(float),
        cudaMemcpyDeviceToHost)
    );
    std::cerr << "mask dump" << std::endl;
    hexdump(tmp1.data(), height * width * 2);
    std::cerr << " ..." << std::endl;
    hexdump(tmp1.data() + (batch - 1) * height * width * 2, height * width * 2);
    auto tmp2 = std::vector<float>(batch * channels * height * width);
    ReportCUDAErrors(cudaMemcpy(
        &tmp2[0],
        input,
        batch * height * channels * width * sizeof(float),
        cudaMemcpyDeviceToHost)
    );
    std::cerr << "input dump" << std::endl;
    hexdump(tmp2.data(), height * width * 2);
    std::cerr << " ..." << std::endl;
    hexdump(tmp2.data() + (batch - 1) * (channels- 1) * height * width * 2, height * width * 2);
    auto tmp3 = std::vector<float>(channels * filter_size * filter_size);
    ReportCUDAErrors(cudaMemcpy(
        &tmp3[0],
        weights,
        channels * filter_size * filter_size * sizeof(float),
        cudaMemcpyDeviceToHost)
    );
    std::cerr << "weights dump" << std::endl;
    hexdump(tmp3.data(), filter_size * filter_size * 2);
    std::cerr << " ..." << std::endl;
    hexdump(tmp3.data() + (channels- 1) * filter_size * filter_size * 2, filter_size * filter_size * 2);
#endif
*/

    depthwise_conv_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, weights,
        biases, residual, mask,
        filter_size, batch, channels, height, width,
        act);

    ReportCUDAErrors(cudaGetLastError());
}
} // namespace cuda
#endif
#endif
