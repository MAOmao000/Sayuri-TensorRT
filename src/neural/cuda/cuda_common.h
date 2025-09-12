#pragma once

#ifdef USE_CUDA

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include "NvInfer.h"

#include "utils/format.h"

namespace cuda {

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity severity = Severity::kERROR)
        : mReportableSeverity(severity) {}
    void log(ILogger::Severity severity, const char* msg) noexcept override {
        // suppress information level log
        if (severity <= mReportableSeverity) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    std::cerr << "[F] " << msg << std::endl;
                    break;
                case Severity::kERROR:
                    std::cerr << "[E] " << msg << std::endl;
                    break;
                case Severity::kWARNING:
                    std::cerr << "[W] " << msg << std::endl;
                    break;
                case Severity::kINFO:
                    std::cerr << "[I] " << msg << std::endl;
                    break;
                case Severity::kVERBOSE:
                    std::cerr << "[V] " << msg << std::endl;
                    break;
                default:
                    std::cerr << "[?] " << msg << std::endl;
            }
        }
    }
    nvinfer1::ILogger& getTRTLogger() noexcept {
        return *this;
    }
    void setReportableSeverity(Severity severity) noexcept {
        mReportableSeverity = severity;
    }
private:
    Severity mReportableSeverity;
};

#define KBLOCKSIZE 256

#define ASSERT(condition)                                           \
    {                                                               \
        if (!(condition)) {                                         \
            LOGGING << Format("Assertion failure %s(%d): %s\n",     \
                __FILE__, __LINE__, #condition);                    \
            throw std::runtime_error("TensorRT error");             \
        }                                                           \
    }

void CudaError(cudaError_t status);

#define ReportCUDAErrors(status) CudaError(status)

size_t GetCudaTypeSize(bool fp16);
cudaDeviceProp GetDeviceProp();
int GetDeviceCount();
int GetDevice();
void SetDevice(int n);
void WaitToFinish(cudaStream_t s);

struct CudaHandles {
    cudaStream_t stream;

    bool fp16;
    bool has_tensor_cores;

    int gpu_id;
    bool initialized{false};

    void ApplyOnCurrentDevice();
    void Release();
};

std::string GetBackendInfo();
std::string GetCurrentDeviceInfo(CudaHandles *handles);

void MallocAndCopy(bool fp16, void **cude_op,
                   const std::vector<float> &weights);

} // namespace cuda

#endif
