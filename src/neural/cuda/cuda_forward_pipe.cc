#ifdef USE_CUDA

#include <sstream>
#include <stdexcept>
#include <filesystem>

#include "config.h"
#include "neural/cuda/cuda_forward_pipe.h"
#include "neural/cuda/cuda_sha2.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/option.h"
#include "version.h"

cuda::Logger tensorrt_logger{};

void CudaForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    LOGGING << cuda::GetBackendInfo();

    dump_gpu_info_ = true;

    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get());

    auto option = ForwardPipeOption::Get().
                      SetBoardSize(GetOption<int>("defualt_boardsize")).
                      SetBatchSize(GetOption<int>("batch_size"));
    Construct(option, weights);

    AssignWorkers(); // Run the batch forwarding worker.
}

OutputResult CudaForwardPipe::Forward(const InputData &input) {
    OutputResult output;
    InputData reordered_input = input;

    // Reorder the inputs data.
    const int planes_bsize = input.board_size;
    const bool should_reorder = planes_bsize != board_size_;

    if (should_reorder) {
        // The input data's board size doesn't match the NN's expected
        // input size. We are reordering the original input data to conform
        // to the NN's input dimensions.
        for (int c = 0; c < weights_->input_channels; ++c) {
            int offset_r = c * board_size_ * board_size_; // data's ordering index
            int offset_p = c * planes_bsize * planes_bsize; // NN's ordering index

            for (int idx = 0; idx < board_size_ * board_size_; ++idx) {
                const int x = idx % board_size_;
                const int y = idx / board_size_;
                if (x < planes_bsize && y < planes_bsize) {
                    reordered_input.planes[offset_r++] = input.planes[offset_p++];
                } else {
                    reordered_input.planes[offset_r++] = 0.f;
                }
            }
        }
    }

    auto entry = std::make_shared<ForwawrdEntry>(reordered_input, output);
    std::unique_lock<std::mutex> lock(entry->mutex);
    {
        // Push the entry into queue.
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        entry_queue_.emplace_back(entry);
    }

    if (static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_) {
        cv_.notify_one(); // Wake up one worker if there are enough batch size.
    }
    entry->cv.wait(lock); // Wait for batch forwarding worker.

    // Reorder the outputs data.
    OutputResult reordered_ouput = output;

    if (should_reorder) {
        // Reorder the NN's outputs data to fit the correct data format.
        int offset_r = 0; // data order index
        int offset_p = 0; // NN order index
        for (int idx = 0; idx < board_size_ * board_size_; ++idx) {
            const int x = idx % board_size_;
            const int y = idx / board_size_;
            if (x < planes_bsize && y < planes_bsize) {
                reordered_ouput.probabilities[offset_r] = output.probabilities[offset_p];
                reordered_ouput.ownership[offset_r] = output.ownership[offset_p];
                offset_r++;
                offset_p++;
            } else {
                offset_p++;
            }
        }
    }

    return reordered_ouput;
}

bool CudaForwardPipe::Valid() const {
    return weights_ != nullptr;
}

void CudaForwardPipe::Construct(ForwardPipeOption option,
                                std::shared_ptr<DNNWeights> weights) {
    // Construct the network with parameters (e.g., board_size) and weights.
    // If the current parameters are the same as the new ones, exit the function 
    // immediately.
    if (weights) {
        weights_ = weights;
    }
    if (weights_ == nullptr) {
        // use dummy backend
        return;
    }

    int board_size = option.IsValidBoardSize() ?
                         option.board_size : board_size_;
    int batch_size = option.IsValidBatchSize() ?
                         option.batch_size : max_batch_per_nn_;
    // Select the matched board size.
    board_size = std::max(board_size, GetOption<int>("fixed_nn_boardsize"));

    if (board_size == 0 || batch_size == 0) {
        LOGGING << "NN board size/batch size should be larger than zero.\n";
        return;
    }

    forwarding_batch_per_nn_ = batch_size;
    if (board_size_ == board_size &&
            batch_size <= max_batch_per_nn_) {
        return;
    }
    Release();

    board_size_ = board_size;
    max_batch_per_nn_ = batch_size;

    // Dynamically allocates GPU resources for neural network computation.
    // It prioritizes user-specified GPUs, validates them, and if none are
    // specified or valid, it automatically assigns all available CUDA devices.
    const auto cuda_device_cnt = cuda::GetDeviceCount();
    const auto specific_gpus_cnt = GetOptionCount("gpus");
    auto gpus_list = std::vector<int>{};

    for (int idx = 0; idx < specific_gpus_cnt; ++idx) {
        auto gpu_id = GetOption<int>("gpus", idx);
        if (gpu_id < cuda_device_cnt) {
            gpus_list.emplace_back(gpu_id);
        } else {
            LOGGING << Format("Not found GPU device (%d).\n", gpu_id);
        }
    }

    if (gpus_list.empty()) {
        LOGGING << "Not found any specific GPU device! Now assign the GPU(s) automatically.\n";
        for (int i = 0; i < cuda_device_cnt; ++i) {
            gpus_list.emplace_back(i);
        }
    }
    if (gpus_list.empty()) {
        throw std::runtime_error("No executable GPU device!");
    }

    for (size_t i = 0; i < gpus_list.size(); ++i) {
        nngraphs_.emplace_back(std::make_unique<NNGraph>());
    }

    max_batch_per_nn_ = batch_size;

    // Construct network graph for each valid GPU.
    for (auto i = size_t{0}; i < gpus_list.size(); ++i) {
        nngraphs_[i]->ConstructGraph(
            dump_gpu_info_, gpus_list[i], max_batch_per_nn_, board_size_, weights_);
        if (!nngraphs_[i]->build(
            dump_gpu_info_, gpus_list[i], max_batch_per_nn_, board_size_, weights_)) {
            throw std::runtime_error("TensorRT backend: failed to construct network!");
        }
    }

    dump_gpu_info_ = false; // don't show the GPU info next time.
}

void CudaForwardPipe::Release() {
    for (auto &g : nngraphs_) {
        g->DestroyGraph();
    }
    nngraphs_.clear();
}

void CudaForwardPipe::Destroy() {
    Release();
    QuitWorkers();
}

void CudaForwardPipe::NNGraph::ConstructGraph(bool dump_gpu_info,
                                              const int gpu,
                                              const int max_batch_size,
                                              const int board_size,
                                              std::shared_ptr<DNNWeights> weights) {
    if (graph_ != nullptr) {
        return;
    }

    graph_ = std::make_unique<Graph>();
    weights_ = weights;

    cuda::SetDevice(gpu);
    handles_.ApplyOnCurrentDevice();

    SetComputationMode(&handles_);
    if (dump_gpu_info) {
        LOGGING << cuda::GetCurrentDeviceInfo(&handles_);
    }

    board_size_ = board_size;
    max_batch_ = max_batch_size;

    // Build the graph first.
    const auto input_channels = weights_->input_channels;
    const auto output_channels = weights_->residual_channels;
    const auto default_act = weights_->default_act;

    // input layer
    graph_->input_conv = cuda::Convolution(
        &handles_,
        max_batch_,          // max batch size
        board_size_,         // board size
        3,                   // kernel size
        input_channels,      // input channels
        output_channels,     // output channels
        default_act          // activation
    );

    // block tower
    const auto blocks = weights_->residual_blocks;
    for (int i = 0; i < weights_->residual_blocks; ++i) {
        graph_->tower.emplace_back(NNGraph::Block{});
    }

    for (int i = 0; i < blocks; ++i) {
        const auto tower_ptr = weights_->tower[i].get();
        if (tower_ptr->IsResidualBlock()) {
            const auto channels = weights_->residual_channels;
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                3,            // kernel size
                channels,     // input channels
                channels,     // output channels
                default_act   // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                3,            // kernel size
                channels,     // input channels
                channels,     // output channels
                last_act      // activation
            );
        } else if (tower_ptr->IsBottleneckBlock()) {
            const auto outer_channels = weights_->residual_channels;
            const auto inner_channels = tower_ptr->bottleneck_channels;
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].pre_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                outer_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].post_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                inner_channels, // input channels
                outer_channels, // output channels
                last_act        // activation
            );
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            const auto outer_channels = weights_->residual_channels;
            const auto inner_channels = tower_ptr->bottleneck_channels;
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].pre_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                outer_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv3 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv4 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].post_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                inner_channels, // input channels
                outer_channels, // output channels
                last_act        // activation
            );
        } else if (tower_ptr->IsMixerBlock()) {
            const auto channels = weights_->residual_channels;
            const auto feedforwards = tower_ptr->feedforward_channels;
            const auto filters = tower_ptr->dw_conv.GetFilter();
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].dw_conv = cuda::DepthwiseConvolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                filters,      // kernel size
                channels,     // input channels
                default_act   // activation
            );
            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                1,            // kernel size
                channels,     // input channels
                feedforwards, // output channels
                default_act   // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                1,            // kernel size
                feedforwards, // input channels
                channels,     // output channels
                last_act      // activation
            );
        }
        if (tower_ptr->apply_se) {
            const auto channels = weights_->residual_channels;
            const size_t se_size = tower_ptr->se_size;

            graph_->tower[i].se_module = cuda::SEUnit(
                &handles_,
                max_batch_,  // max batch size
                board_size_, // board size
                channels,    // channels
                se_size,     // SE size
                default_act  // activation
            );
        }
    }

    // policy head
    const auto policy_head_channels = weights_->policy_head_channels;
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    graph_->p_hd_conv = cuda::Convolution(
        &handles_,
        max_batch_,              // max batch size
        board_size_,             // board size
        1,                       // kernel size
        output_channels,         // input channels
        policy_head_channels,    // output channels
        default_act              // activation
    );
    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        const auto filters = weights_->p_dw_conv.GetFilter();
        graph_->p_dw_conv = cuda::DepthwiseConvolution(
            &handles_,
            max_batch_,              // max batch size
            board_size_,             // board size
            filters,                 // kernel size
            policy_head_channels,    // input channels
            default_act              // activation
        );
        graph_->p_pt_conv = cuda::Convolution(
            &handles_,
            max_batch_,              // max batch size
            board_size_,             // board size
            1,                       // kernel size
            policy_head_channels,    // input channels
            policy_head_channels,    // output channels
            default_act              // activation
        );
    }
    graph_->p_pool = cuda::GlobalPooling(
        &handles_,
        false,
        max_batch_,               // max batch size
        board_size_,              // board size
        policy_head_channels      // input channels
    );
    graph_->p_inter = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*policy_head_channels,   // input sizes
        policy_head_channels,     // outpur size
        default_act               // activation
    );
    graph_->p_prob = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        policy_head_channels,     // input channels
        probabilities_channels,   // output channels
        Activation::kIdentity     // activation
    );
    graph_->p_prob_pass = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        policy_head_channels,     // input sizes
        pass_probability_outputs, // outpur size
        Activation::kIdentity     // activation
    );

    // value head
    const auto value_head_channels = weights_->value_head_channels;
    const auto ownership_channels = weights_->ownership_channels;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    graph_->v_hd_conv = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        output_channels,          // input channels
        value_head_channels,      // output channels
        default_act               // activation
    );
    graph_->v_pool = cuda::GlobalPooling(
        &handles_,
        true,
        max_batch_,               // max batch size
        board_size_,              // board size
        value_head_channels       // input channels
    );
    graph_->v_inter = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*value_head_channels,    // input sizes
        3*value_head_channels,    // outpur size
        default_act               // activation
    );
    graph_->v_ownership = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        value_head_channels,      // input channels
        ownership_channels,       // output channels
        Activation::kIdentity     // activation
    );
    graph_->v_misc = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*value_head_channels,    // input size
        value_misc_outputs,       // output size
        Activation::kIdentity     // relu
    );

    // Now push the weights.

    // input layer
    graph_->input_conv.LoadWeights(
        weights_->input_conv.GetWeights(),
        weights_->input_conv.GetBiases());

    // block tower
    for (int i = 0; i < blocks; ++i) {
        const auto tower_ptr = weights_->tower[i].get();
        if (tower_ptr->IsResidualBlock()) {
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases());
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases());
        } else if (tower_ptr->IsBottleneckBlock()) {
            graph_->tower[i].pre_btl_conv.LoadWeights(
                tower_ptr->pre_btl_conv.GetWeights(),
                tower_ptr->pre_btl_conv.GetBiases());
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases());
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases());
            graph_->tower[i].post_btl_conv.LoadWeights(
                tower_ptr->post_btl_conv.GetWeights(),
                tower_ptr->post_btl_conv.GetBiases());
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            graph_->tower[i].pre_btl_conv.LoadWeights(
                tower_ptr->pre_btl_conv.GetWeights(),
                tower_ptr->pre_btl_conv.GetBiases());
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases());
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases());
            graph_->tower[i].conv3.LoadWeights(
                tower_ptr->conv3.GetWeights(),
                tower_ptr->conv3.GetBiases());
            graph_->tower[i].conv4.LoadWeights(
                tower_ptr->conv4.GetWeights(),
                tower_ptr->conv4.GetBiases());
            graph_->tower[i].post_btl_conv.LoadWeights(
                tower_ptr->post_btl_conv.GetWeights(),
                tower_ptr->post_btl_conv.GetBiases());
        } else if (tower_ptr->IsMixerBlock()) {
            graph_->tower[i].dw_conv.LoadWeights(
                tower_ptr->dw_conv.GetWeights(),
                tower_ptr->dw_conv.GetBiases());
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases());
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases());
        }
        if (tower_ptr->apply_se) {
            graph_->tower[i].se_module.LoadWeights(
                tower_ptr->squeeze.GetWeights(),
                tower_ptr->squeeze.GetBiases(),
                tower_ptr->excite.GetWeights(),
                tower_ptr->excite.GetBiases());
        }
    }

    // policy head
    graph_->p_hd_conv.LoadWeights(
        weights->p_hd_conv.GetWeights(),
        weights->p_hd_conv.GetBiases());

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        graph_->p_dw_conv.LoadWeights(
            weights->p_dw_conv.GetWeights(),
            weights->p_dw_conv.GetBiases());
        graph_->p_pt_conv.LoadWeights(
            weights->p_pt_conv.GetWeights(),
            weights->p_pt_conv.GetBiases());
    }

    graph_->p_inter.LoadWeights(
        weights_->p_inter_fc.GetWeights(), weights_->p_inter_fc.GetBiases());

    graph_->p_prob.LoadWeights(
        weights->prob_conv.GetWeights(),
        weights_->prob_conv.GetBiases());

    graph_->p_prob_pass.LoadWeights(
        weights_->pass_fc.GetWeights(), weights_->pass_fc.GetBiases());

    // value head
    graph_->v_hd_conv.LoadWeights(
        weights->v_hd_conv.GetWeights(),
        weights->v_hd_conv.GetBiases());

    graph_->v_inter.LoadWeights(
        weights_->v_inter_fc.GetWeights(), weights_->v_inter_fc.GetBiases());

    graph_->v_ownership.LoadWeights(
        weights->v_ownership.GetWeights(),
        weights_->v_ownership.GetBiases());

    graph_->v_misc.LoadWeights(
        weights_->v_misc.GetWeights(), weights_->v_misc.GetBiases());
}

void CudaForwardPipe::NNGraph::SetComputationMode(cuda::CudaHandles *handles) {
    cudaDeviceProp dev_prop = cuda::GetDeviceProp();

    if (dev_prop.major <= 6 ||
            !GetOption<bool>("fp16")) {
        // The compute capability is too low. The 5 is Maxwell,
        // such as GTX 980 Ti. The 6 is Pascal, such as GTX 1080 Ti.
        // As fair as I know, the FP16 can work on these devices,
        // but the their performance are bad. The FP32 is better
        // choice. So disable the FP16 computation.
        handles->fp16 = false;
    }

    if (!(handles->fp16)) {
        handles->has_tensor_cores = false;
    }
}

std::vector<OutputResult> CudaForwardPipe::NNGraph::BatchForward(const std::vector<InputData> &batch_input) {
    const auto batch_size = (int)batch_input.size();

    assert(max_batch_ >= batch_size);

    const auto input_channels = weights_->input_channels;
    const auto num_intersections = board_size_ * board_size_;
    auto batch_planes = std::vector<float>(batch_size * input_channels * num_intersections);

    for (int b = 0; b < batch_size; ++b) {
        const auto& input = batch_input[b];
        for (int idx = 0; idx < input_channels * num_intersections; ++idx) {
            batch_planes[b * input_channels * num_intersections + idx] = input.planes[idx];
        }
    }

    auto search = context_->buffers_.find("InputFeature");
    assert(search != context_->buffers_.end());
    cuda::ReportCUDAErrors(cudaMemcpyAsync(
        search->second,
        (float*)&batch_planes[0],
        batch_size * sizeof(float) * input_channels * num_intersections,
        cudaMemcpyHostToDevice,
        cudaStreamPerThread)
    );

    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    const auto ownership_channels = weights_->ownership_channels;

    auto batch_prob = std::vector<float>(batch_size * probabilities_channels * num_intersections);
    auto batch_prob_pass = std::vector<float>(batch_size * pass_probability_outputs);
    auto batch_value_misc = std::vector<float>(batch_size * value_misc_outputs);
    auto batch_ownership = std::vector<float>(batch_size * ownership_channels * num_intersections);

    context_->execution_context_->setInputShape(
        "InputFeature",
        Dims4(
            batch_size,
            input_channels,
            board_size_,
            board_size_)
    );
    context_->execution_context_->setInputShape(
        "BatchSize",
        Dims4(
            batch_size,
            weights_->residual_channels,
            1,
            1)
    );
    ASSERT(context_->execution_context_->enqueueV3(cudaStreamPerThread));

    search = context_->buffers_.find("output_prob");
    assert(search != context_->buffers_.end());
    cuda::ReportCUDAErrors(cudaMemcpyAsync(
        &batch_prob[0],
        search->second,
        batch_size * probabilities_channels * num_intersections * sizeof(float),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread)
    );

    search = context_->buffers_.find("output_prob_pass");
    assert(search != context_->buffers_.end());
    cuda::ReportCUDAErrors(cudaMemcpyAsync(
        &batch_prob_pass[0],
        search->second,
        batch_size * pass_probability_outputs * sizeof(float),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread)
    );

    search = context_->buffers_.find("output_val");
    assert(search != context_->buffers_.end());
    cuda::ReportCUDAErrors(cudaMemcpyAsync(
        &batch_value_misc[0],
        search->second,
        batch_size * value_misc_outputs * sizeof(float),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread)
    );

    search = context_->buffers_.find("output_ownership");
    assert(search != context_->buffers_.end());
    cuda::ReportCUDAErrors(cudaMemcpyAsync(
        &batch_ownership[0],
        search->second,
        batch_size * ownership_channels * num_intersections * sizeof(float),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread)
    );

    // Asynchronously enqueue the inference work
    cudaStreamSynchronize(cudaStreamPerThread);
    auto batch_output_result = std::vector<OutputResult>(batch_size);

    FillOutputs(batch_prob,
                batch_prob_pass,
                batch_value_misc,
                batch_ownership,
                batch_input,
                batch_output_result);
    return batch_output_result;
}

void CudaForwardPipe::NNGraph::FillOutputs(const std::vector<float> &batch_prob,
                                           const std::vector<float> &batch_prob_pass,
                                           const std::vector<float> &batch_value_misc,
                                           const std::vector<float> &batch_ownership,
                                           const std::vector<InputData> &batch_input,
                                           std::vector<OutputResult> &batch_output_result) {
    const int batch_size = batch_output_result.size();
    const auto num_intersections = board_size_ * board_size_;
    const auto encoder_version = Encoder::GetEncoderVersion(weights_->version); 
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    const auto ownership_channels = weights_->ownership_channels;

    if (encoder_version == 1) {
        for (int b = 0; b < batch_size; ++b) {
            auto &output_result = batch_output_result[b];
            const auto &input = batch_input[b];
            const int pol_offset = probabilities_channels * num_intersections;
            const int own_offset = ownership_channels * num_intersections;
            for (int idx = 0; idx < num_intersections; ++idx) {
                int pol_index = b * pol_offset + (int)PolicyBufferOffset::kNormal * num_intersections + idx;
                int own_index = b * own_offset + 0 * num_intersections + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs + 0];

            output_result.wdl[0]      = batch_value_misc[b * value_misc_outputs + 0];
            output_result.wdl[1]      = batch_value_misc[b * value_misc_outputs + 1];
            output_result.wdl[2]      = batch_value_misc[b * value_misc_outputs + 2];
            output_result.stm_winrate = batch_value_misc[b * value_misc_outputs + 3];
            output_result.final_score = batch_value_misc[b * value_misc_outputs + 4];
            output_result.q_error     = 0.0f;
            output_result.score_error = 0.0f;

            output_result.offset = PolicyBufferOffset::kNormal;
            output_result.board_size = input.board_size;
            output_result.komi = input.komi;
            output_result.fp16 = handles_.fp16;
        }
    } else if (encoder_version == 2) {
        for (int b = 0; b < batch_size; ++b) {
            auto &output_result = batch_output_result[b];
            const auto &input = batch_input[b];
            const int pol_offset = probabilities_channels * num_intersections;
            const int own_offset = ownership_channels * num_intersections;
            for (int idx = 0; idx < num_intersections; ++idx) {
                int pol_index = b * pol_offset + (int)input.offset * num_intersections + idx;
                int own_index = b * own_offset + 0 * num_intersections + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs + 0];

            output_result.wdl[0]      = batch_value_misc[b * value_misc_outputs + 0];
            output_result.wdl[1]      = batch_value_misc[b * value_misc_outputs + 1];
            output_result.wdl[2]      = batch_value_misc[b * value_misc_outputs + 2];
            output_result.stm_winrate = batch_value_misc[b * value_misc_outputs + 3];
            output_result.final_score = batch_value_misc[b * value_misc_outputs + 8];
            output_result.q_error     = batch_value_misc[b * value_misc_outputs + 13];
            output_result.score_error = batch_value_misc[b * value_misc_outputs + 14];

            output_result.offset = input.offset;
            output_result.board_size = input.board_size;
            output_result.komi = input.komi;
            output_result.fp16 = handles_.fp16;
        }
    }
}

void CudaForwardPipe::NNGraph::DestroyGraph() {
    if (graph_ == nullptr) {
        return;
    }

    handles_.Release();

    graph_.reset();
    graph_ = nullptr;
}

CudaForwardPipe::NNGraph::~NNGraph() {
    DestroyGraph();
}

void CudaForwardPipe::AssignWorkers() {
    worker_running_.store(true);
    waittime_.store(GetOption<int>("gpu_waittime"), std::memory_order_relaxed);

    ThreadPool::Get("cuda-forward-pipe", nngraphs_.size());
    if (group_->FutureEmpty()) {
        for (int gpu = 0; gpu < (int)nngraphs_.size(); ++gpu) {
            group_->AddTask([g=gpu, this](){ Worker(g); });
        }
    }
}

void CudaForwardPipe::Worker(int gpu) {
    const auto GatherBatches = [this](int gpu_waittime) {
        auto entries = std::vector<std::shared_ptr<ForwawrdEntry>>{};

        // Running the loop until there are enough entries in the queue or time out,
        // then breaking the loop.
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            while(true) {
                if (!worker_running_.load(std::memory_order_relaxed)) {
                    return entries;
                }
                if (static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_) {
                    break;
                }

                bool timeout = false;
                if (waittime_.load(std::memory_order_relaxed) != 0) {
                    // Wait for some time to avoid busy waiting.
                    timeout = !cv_.wait_for(
                        lock, std::chrono::milliseconds(waittime_.load(std::memory_order_relaxed)),
                        [this]() {
                            return !worker_running_.load(std::memory_order_relaxed) ||
                                       static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_; });
                }

                if (entry_queue_.empty()) {
                    // No any entry in the queue. In this case, we can not
                    // expect next forwarding time. Keep increasing waiting
                    // time.
                    auto last_waittime = waittime_.fetch_add(1, std::memory_order_relaxed);
                    if (last_waittime >= gpu_waittime) {
                        waittime_.store(gpu_waittime, std::memory_order_relaxed);
                    }
                } else {
                    if (timeout) {
                        // May be CPU-bound time. Boost forwarding.
                        waittime_.store(0, std::memory_order_relaxed);
                    }
                    break;
                }
            }
        }

        // Gather the entries and return.
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            auto count = std::min(static_cast<int>(entry_queue_.size()), forwarding_batch_per_nn_);
            auto end = std::begin(entry_queue_);
            std::advance(end, count);
            std::move(std::begin(entry_queue_), end, std::back_inserter(entries));
            entry_queue_.erase(std::begin(entry_queue_), end);
        }
        return entries;
    };

    const auto gpu_waittime_base = GetOption<int>("gpu_waittime");
    while (true) {
        if (!worker_running_.load(std::memory_order_relaxed)) return;

        auto entries = GatherBatches(gpu_waittime_base);
        const auto batch_size = entries.size();

        if (batch_size == 0) {
            continue;
        }

        // Gather batch data.
        auto inputs = std::vector<InputData>(batch_size);
        for (auto b = size_t{0}; b < batch_size; ++b) {
            inputs[b] = entries[b]->input;
        }

        // Forwarding...
        auto outputs = nngraphs_[gpu]->BatchForward(inputs);

        for (auto b = size_t{0}; b < batch_size; ++b) {
            entries[b]->output = outputs[b];
            {
                // Be sure the condition variable of current entry is ready.
                std::unique_lock<std::mutex> lk(entries[b]->mutex);
            }
            entries[b]->cv.notify_all();
        }
    }
}

void CudaForwardPipe::QuitWorkers() {
    worker_running_.store(false);
    cv_.notify_all();
    group_->WaitToJoin();
}

bool CudaForwardPipe::NNGraph::build(bool dump_gpu_info,
                                     const int gpu,
                                     const int max_batch_size,
                                     const int board_size,
                                     std::shared_ptr<DNNWeights> weights) {

    auto builder
        = TrtUniquePtr<IBuilder>(createInferBuilder(tensorrt_logger.getTRTLogger()));
    if (!builder) {
        LOGGING << "TensorRT backend: failed to create builder.\n";
        return false;
    }
    auto config = TrtUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        LOGGING << "TensorRT backend: failed to create builder config.\n";
        return false;
    }
    if (handles_.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        LOGGING << "TensorRT backend: failed to create optimization profile.\n";
        return false;
    }
    profile->setDimensions("InputFeature", OptProfileSelector::kMIN,
        Dims4(1, weights->input_channels, board_size, board_size));
    profile->setDimensions("InputFeature", OptProfileSelector::kOPT,
        Dims4(max_batch_size, weights->input_channels, board_size, board_size));
    profile->setDimensions("InputFeature", OptProfileSelector::kMAX,
        Dims4(max_batch_size, weights->input_channels, board_size, board_size));
    profile->setDimensions("BatchSize", OptProfileSelector::kMIN,
        Dims4(1, weights->residual_channels, 1, 1));
    profile->setDimensions("BatchSize", OptProfileSelector::kOPT,
        Dims4(max_batch_size,weights->residual_channels, 1, 1));
    profile->setDimensions("BatchSize", OptProfileSelector::kMAX,
        Dims4(max_batch_size, weights->residual_channels, 1, 1));
    config->addOptimizationProfile(profile);

    nvinfer1::NetworkDefinitionCreationFlags flags = 0U;
    auto network = TrtUniquePtr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        LOGGING << "TensorRT backend: failed to create network definition.\n";
        return false;
    }
    auto weights_file = GetOption<std::string>("weights_file");
    network->setName(weights_file.c_str());
    if (!constructNetwork(network)) {
        LOGGING << "TensorRT backend: failed to construct network.\n";
        return false;
    }
    cudaDeviceProp dev_prop = cuda::GetDeviceProp();
    if (dev_prop.major >= 8) {
        // This is to avoid tactics that have shape switching overhead
        config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS));
        config->setBuilderOptimizationLevel(2);
    }
    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamPerThread);
    // Typical runtime allocation is much less than the 2 GiB specified below
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 31);

    std::string plan;
    if (GetOption<std::string>("mode") == "selfplay") {
        auto planBuffer = std::unique_ptr<IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
        if (!planBuffer) {
            LOGGING << "TensorRT backend: failed to create plan.\n";
            return false;
        }
        plan.insert(
            plan.end(),
            static_cast<char*>(planBuffer->data()),
            static_cast<char*>(planBuffer->data()) + planBuffer->size()
        );
    } else {
        std::ifstream in;
        in.open(weights_file, std::ios::in | std::ios::binary | std::ios::ate);
        std::ifstream::pos_type fileSize = in.tellg();
        if (fileSize < 0) {
            LOGGING << "tellg failed to determine size.\n";
            return false;
        }
        std::string str;
        in.seekg(0, std::ios::beg);
        str.resize(fileSize);
        in.read(&str[0], fileSize);
        in.close();
        char hashResultBuf[65];
        SHA2::get256((const uint8_t*)str.data(), str.size(), hashResultBuf);
        std::string model_hash{""};
        model_hash.assign(hashResultBuf);

        {
            static std::mutex tuneMutex;
            tuneMutex.lock();
            uint8_t deviceHash[32];
            SHA2::get256(dev_prop.name, deviceHash);

            // Truncated to 4 bytes
            char deviceIdent[4 * 2 + 1];
            for(int i = 0; i < 4; i++) {
                sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
            }
            deviceIdent[sizeof(deviceIdent) - 1] = 0;

            std::string precision = (handles_.fp16) ? "half" : "single";

            std::filesystem::path filepath = network->getName();
            auto planCacheFile = strprintf(
                "trt-%d_gpu-%s_net-%s_%s_%dx%d_batch%d_%s",
                getInferLibVersion(),
                deviceIdent,
                filepath.filename().string().c_str(),
                GetProgramVersion().c_str(),
                board_size,
                board_size,
                max_batch_size,
                precision.c_str()
            );
            std::string paramStr = strprintf(
                "_%d_%s_%s_%d_%d_%d_%s",
                getInferLibVersion(),
                deviceIdent,
                GetProgramVersion().c_str(),
                board_size,
                board_size,
                max_batch_size,
                precision.c_str()
            );

            try {
                plan = readFileBinary(planCacheFile);
            } catch (std::exception const& e) {
                (void) e;
            };
            if (plan.size() > 0) {
                if (plan.size() < 64 + paramStr.size()) {
                    LOGGING << "Could not parse plan, unexpected size in " + planCacheFile + ".\n";
                    plan.clear();
                } else {
                    std::string cachedParamStr = plan.substr(plan.size() - paramStr.size());
                    std::string modelHash = plan.substr(plan.size() - 64 - paramStr.size(), 64);
                    if (modelHash != model_hash) {
                        LOGGING << "Plan cache is corrupted or is for the wrong model in " + planCacheFile + ".\n";
                        plan.clear();
                    } else if (cachedParamStr != paramStr) {
                        LOGGING << "Plan cache is corrupted or is for the wrong parameters in " + planCacheFile + ".\n";
                        plan.clear();
                    } else {
                        plan.erase(plan.size() - 64 - paramStr.size());
                    }
                }
            }
            if (plan.size() <= 0) {
                LOGGING << "Creating new plan cache.\n";
                auto planBuffer = std::unique_ptr<IHostMemory>(
                    builder->buildSerializedNetwork(*network, *config));
                if (!planBuffer) {
                    tuneMutex.unlock();
                    LOGGING << "TensorRT backend: failed to create plan.\n";
                    return false;
                }
                plan.insert(
                    plan.end(),
                    static_cast<char*>(planBuffer->data()),
                    static_cast<char*>(planBuffer->data()) + planBuffer->size()
                );
                if (model_hash.size() != 64) {
                    tuneMutex.unlock();
                    LOGGING << "Unexpected model hash size.\n";
                    return false;
                }
                plan.insert(
                    plan.end(),
                    model_hash.begin(),
                    model_hash.end()
                );
                plan.insert(
                    plan.end(),
                    paramStr.begin(),
                    paramStr.end()
                );
#ifdef NDEBUG
                std::ofstream ofs;
                ofs.open(planCacheFile, std::ios_base::out | std::ios_base::binary);
                ofs.write(plan.data(), plan.size());
                ofs.close();
#endif
                LOGGING << "Saved new plan cache to " + planCacheFile + ".\n";
                plan.erase(plan.size() - 64 - paramStr.size());
            } else {
                LOGGING << "Using existing plan cache at " + planCacheFile + ".\n";
            }
            tuneMutex.unlock();
        }
    }

    runtime_.reset(createInferRuntime(tensorrt_logger.getTRTLogger()));
    if (!runtime_) {
        LOGGING << "createInferRuntime error.\n";
        return false;
    }
    engine_.reset(
            runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) {
        LOGGING << "deserializeCudaEngine error.\n";
        return false;
    }
    context_ = std::make_unique<BackendContext>();
    context_->execution_context_.reset(engine_->createExecutionContext());
    if (!context_->execution_context_) {
        LOGGING << "failed to create execution context.\n";
        return false;
    }
    for (auto j = 0; j < engine_->getNbIOTensors(); j++) {
        void* buffer = nullptr;
        auto name = engine_->getIOTensorName(j);
        auto dims = engine_->getTensorShape(name);
        std::string_view name_str{name};
        size_t size_byte;
        if (name_str == "BatchSize") {
            size_byte = sizeof(int32_t);
        } else {
            size_byte = sizeof(float);
        }
        size_t bytes = std::accumulate(
            dims.d + 1,
            dims.d + dims.nbDims,
            max_batch_size * size_byte,
            std::multiplies<size_t>());
        cuda::ReportCUDAErrors(cudaMalloc(&buffer, bytes));
        if (name_str == "BatchSize") {
            auto input_batch
                = std::vector<int32_t>(max_batch_size * weights_->residual_channels, 0);
            cuda::ReportCUDAErrors(cudaMemcpy(
                buffer,
                (int32_t*)&input_batch[0],
                bytes,
                cudaMemcpyHostToDevice));
        }
        context_->buffers_.emplace(std::make_pair(name, buffer));
        if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            context_->execution_context_->setInputTensorAddress(name, buffer);
        } else {
            context_->execution_context_->setOutputTensorAddress(name, buffer);
        }
    }
    context_->execution_context_->setOptimizationProfileAsync(0, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    return true;
}

bool CudaForwardPipe::NNGraph::constructNetwork(
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network) {

    ITensor* inputFeature = nullptr;
    ITensor* inputSe = nullptr;
    ITensor* inputPool = nullptr;
    ITensor* outputConv = nullptr;

    if (handles_.fp16) {
        auto maskScaleLayerWeights = std::make_unique<half_float_t[]>(1);
        maskScaleLayerWeights[0] = GetFp16((sqrtf(board_size_ * board_size_) - 14.0f) * 0.1f);
        maskScaleLayer_ = network->addConstant(
            {4, {1, 1, 1, 1}}, {DataType::kHALF, maskScaleLayerWeights.get(), 1});
        auto maskQuadLayerWeights = std::make_unique<half_float_t[]>(1);
        maskQuadLayerWeights[0] = GetFp16(
            (sqrtf(board_size_ * board_size_) - 14.0f) *
            (sqrtf(board_size_ * board_size_) - 14.0f) * 0.01f - 0.1f
        );
        maskQuadLayer_ = network->addConstant(
            {4, {1, 1, 1, 1}}, {DataType::kHALF, maskQuadLayerWeights.get(), 1});
        extrahalfWeights_.push_back(move(maskScaleLayerWeights));
        extrahalfWeights_.push_back(move(maskQuadLayerWeights));
    } else {
        auto maskScaleLayerWeights = std::make_unique<float[]>(1);
        maskScaleLayerWeights[0] = (sqrtf(board_size_ * board_size_) - 14.0f) * 0.1f;
        maskScaleLayer_ = network->addConstant(
            {4, {1, 1, 1, 1}}, {DataType::kFLOAT, maskScaleLayerWeights.get(), 1});
        auto maskQuadLayerWeights = std::make_unique<float[]>(1);
        maskQuadLayerWeights[0] =
            (sqrtf(board_size_ * board_size_) - 14.0f) *
            (sqrtf(board_size_ * board_size_) - 14.0f) * 0.01f - 0.1f;
        maskQuadLayer_ = network->addConstant(
            {4, {1, 1, 1, 1}}, {DataType::kFLOAT, maskQuadLayerWeights.get(), 1});
        extraWeights_.push_back(move(maskScaleLayerWeights));
        extraWeights_.push_back(move(maskQuadLayerWeights));
    }

    auto batchSizeTensor = initInputs(
        "BatchSize",
        network,
        weights_->residual_channels,
        1,
        1);
    // See. https://github.com/NVIDIA/TensorRT/issues/2282
    auto inShapeLayer = network->addShape(*batchSizeTensor);
    shapeLayer_ = network->addCast(*inShapeLayer->getOutput(0), DataType::kINT32);
    // input layer
    inputFeature = initInputs(
        "InputFeature",
        network,
        weights_->input_channels,
        board_size_,
        board_size_);
    auto initialConvLayer = buildConvLayer(
        inputFeature,
        weights_->input_conv.GetFilter(),
        weights_->input_conv.GetWeights().size(),
        graph_->input_conv.GetDevWeights(),
        weights_->input_conv.GetBiases().size(),
        graph_->input_conv.GetDevBiases(),
        network,
        weights_->input_conv.GetOutputs());
    auto outputConvLayer = buildActivationLayer(
        initialConvLayer->getOutput(0),
        network);
    outputConv = outputConvLayer->getOutput(0);

    // block tower
    const auto blocks = weights_->residual_blocks;
    for (int i = 0; i < blocks; ++i) {
        const auto tower_ptr = weights_->tower[i].get();
        const auto block_ptr = &graph_->tower[i];
        if (tower_ptr->IsResidualBlock()) {
            // residual
            //  in: batch_size * weights_->residual_channels
            auto blockLayer = buildResidualBlock(
                outputConv,
                tower_ptr,
                block_ptr,
                network);
            outputConv = blockLayer->getOutput(0);
        } else if (tower_ptr->IsBottleneckBlock()) {
            // bottleneck
            //  in: batch_size * weights_->residual_channels
            auto blockLayer = buildBottleneckBlock(
                outputConv,
                tower_ptr,
                block_ptr,
                network);
            outputConv = blockLayer->getOutput(0);
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            // nested-bottleneck
            //  in: batch_size * weights_->residual_channels
            auto blockLayer = buildNestedBottleneckBlock(
                outputConv,
                tower_ptr,
                block_ptr,
                network);
            outputConv = blockLayer->getOutput(0);
        } else if (tower_ptr->IsMixerBlock()) {
            // mixer
            //  in: batch_size * weights_->residual_channels
            auto blockLayer = buildMixerBlock(
                outputConv,
                tower_ptr,
                block_ptr,
                network);
            outputConv = blockLayer->getOutput(0);
        }
    }
    // policy head
    // in: batch_size * weights_->residual_channels * num_intersections
    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        buildPolicyHeadRepLK(outputConv, network);
    } else {
        buildPolicyHead(outputConv, network);
    }
    // value head
    // in: batch_size * weights_->residual_channels * num_intersections
    buildValueHead(outputConv, network);

    LOGGING << "Done constructing network...\n";

    return true;
}

ITensor* CudaForwardPipe::NNGraph::initInputs(
    char const *inputName,
    TrtUniquePtr<INetworkDefinition>& network,
    const int channels,
    const int rows,
    const int cols) {

    ITensor* inputFeature;

    std::string_view name_str{inputName};
    inputFeature = network->addInput(
        inputName,
        DataType::kFLOAT,
        {4, {-1, channels, rows, cols}});
    assert(inputFeature != nullptr);
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    return inputFeature;
}

ILayer* CudaForwardPipe::NNGraph::buildResidualBlock(
    ITensor* input,
    BlockBasic* tower_ptr,
    NNGraph::Block* block_ptr,
    TrtUniquePtr<INetworkDefinition>& network) {

    // 1st conv layer
    auto firstConvLayer = buildConvLayer(
        input,
        tower_ptr->conv1.GetFilter(),
        tower_ptr->conv1.GetWeights().size(),
        block_ptr->conv1.GetDevWeights(),
        tower_ptr->conv1.GetBiases().size(),
        block_ptr->conv1.GetDevBiases(),
        network,
        tower_ptr->conv1.GetOutputs());
    auto firstActivationConvLayer = buildActivationLayer(
        firstConvLayer->getOutput(0), network);
    // 2nd conv layer
    auto secondConvLayer = buildConvLayer(
        firstActivationConvLayer->getOutput(0),
        tower_ptr->conv2.GetFilter(),
        tower_ptr->conv2.GetWeights().size(),
        block_ptr->conv2.GetDevWeights(),
        tower_ptr->conv2.GetBiases().size(),
        block_ptr->conv2.GetDevBiases(),
        network,
        tower_ptr->conv2.GetOutputs());
    if (tower_ptr->apply_se) {
        return buildSqueezeExcitationLayer(
            input,
            secondConvLayer->getOutput(0),
            tower_ptr,
            block_ptr,
            network);
    } else {
        auto mergeLayer = network->addElementWise(
            *input,
            *secondConvLayer->getOutput(0),
            ElementWiseOperation::kSUM);
        auto mergeActivationConvLayer = buildActivationLayer(
            mergeLayer->getOutput(0), network);
        return mergeActivationConvLayer;
    }
}

ILayer* CudaForwardPipe::NNGraph::buildBottleneckBlock(
    ITensor* input,
    BlockBasic* tower_ptr,
    NNGraph::Block* block_ptr,
    TrtUniquePtr<INetworkDefinition>& network) {

    // pre-bottleneck
    auto preConvLayer = buildConvLayer(
        input,
        tower_ptr->pre_btl_conv.GetFilter(),
        tower_ptr->pre_btl_conv.GetWeights().size(),
        block_ptr->pre_btl_conv.GetDevWeights(),
        tower_ptr->pre_btl_conv.GetBiases().size(),
        block_ptr->pre_btl_conv.GetDevBiases(),
        network,
        tower_ptr->pre_btl_conv.GetOutputs());
    auto preActivationConvLayer = buildActivationLayer(
        preConvLayer->getOutput(0), network);
    // 1st conv layer
    auto firstConvLayer = buildConvLayer(
        preActivationConvLayer->getOutput(0),
        tower_ptr->conv1.GetFilter(),
        tower_ptr->conv1.GetWeights().size(),
        block_ptr->conv1.GetDevWeights(),
        tower_ptr->conv1.GetBiases().size(),
        block_ptr->conv1.GetDevBiases(),
        network,
        tower_ptr->conv1.GetOutputs());
    auto firstActivationConvLayer = buildActivationLayer(
        firstConvLayer->getOutput(0), network);
    // 2nd conv layer
    auto secondConvLayer = buildConvLayer(
        firstActivationConvLayer->getOutput(0),
        tower_ptr->conv2.GetFilter(),
        tower_ptr->conv2.GetWeights().size(),
        block_ptr->conv2.GetDevWeights(),
        tower_ptr->conv2.GetBiases().size(),
        block_ptr->conv2.GetDevBiases(),
        network,
        tower_ptr->conv2.GetOutputs());
    auto secondActivationConvLayer = buildActivationLayer(
        secondConvLayer->getOutput(0), network);
    // post-bottleneck
    auto postConvLayer = buildConvLayer(
        secondActivationConvLayer->getOutput(0),
        tower_ptr->post_btl_conv.GetFilter(),
        tower_ptr->post_btl_conv.GetWeights().size(),
        block_ptr->post_btl_conv.GetDevWeights(),
        tower_ptr->post_btl_conv.GetBiases().size(),
        block_ptr->post_btl_conv.GetDevBiases(),
        network,
        tower_ptr->post_btl_conv.GetOutputs());
    if (tower_ptr->apply_se) {
        return buildSqueezeExcitationLayer(
            input,
            postConvLayer->getOutput(0),
            tower_ptr,
            block_ptr,
            network);
    } else {
        auto mergeLayer = network->addElementWise(
            *input,
            *postConvLayer->getOutput(0),
            ElementWiseOperation::kSUM);
        auto mergeActivationConvLayer = buildActivationLayer(
            mergeLayer->getOutput(0), network);
        return mergeActivationConvLayer;
    }
}

ILayer* CudaForwardPipe::NNGraph::buildNestedBottleneckBlock(
    ITensor* input,
    BlockBasic* tower_ptr,
    NNGraph::Block* block_ptr,
    TrtUniquePtr<INetworkDefinition>& network) {

    // nested-bottleneck
    //  in: batch_size * weights_->residual_channels
    //  out: batch_size * weights_->residual_channels / 2
    auto preConvLayer = buildConvLayer(
        input,
        tower_ptr->pre_btl_conv.GetFilter(),
        tower_ptr->pre_btl_conv.GetWeights().size(),
        block_ptr->pre_btl_conv.GetDevWeights(),
        tower_ptr->pre_btl_conv.GetBiases().size(),
        block_ptr->pre_btl_conv.GetDevBiases(),
        network,
        tower_ptr->pre_btl_conv.GetOutputs());
    auto preActivationConvLayer = buildActivationLayer(
        preConvLayer->getOutput(0), network);
    // 1st conv layer (1st block)
    //  in: batch_size * weights_->residual_channels / 2
    //  out: batch_size * weights_->residual_channels / 2
    auto firstConvLayer = buildConvLayer(
        preActivationConvLayer->getOutput(0),
        tower_ptr->conv1.GetFilter(),
        tower_ptr->conv1.GetWeights().size(),
        block_ptr->conv1.GetDevWeights(),
        tower_ptr->conv1.GetBiases().size(),
        block_ptr->conv1.GetDevBiases(),
        network,
        tower_ptr->conv1.GetOutputs());
    auto firstActivationConvLayer = buildActivationLayer(
        firstConvLayer->getOutput(0), network);
    // 2nd conv layer (1st block)
    //  in: batch_size * weights_->residual_channels / 2
    //  out: batch_size * weights_->residual_channels / 2
    auto secondConvLayer = buildConvLayer(
        firstActivationConvLayer->getOutput(0),
        tower_ptr->conv2.GetFilter(),
        tower_ptr->conv2.GetWeights().size(),
        block_ptr->conv2.GetDevWeights(),
        tower_ptr->conv2.GetBiases().size(),
        block_ptr->conv2.GetDevBiases(),
        network,
        tower_ptr->conv2.GetOutputs());
    auto secondMergeLayer = network->addElementWise(
        *preActivationConvLayer->getOutput(0),
        *secondConvLayer->getOutput(0),
        ElementWiseOperation::kSUM);
    auto secondActivationConvLayer = buildActivationLayer(
        secondMergeLayer->getOutput(0), network);
    // 3rd conv layer (2nd block)
    //  in: batch_size * weights_->residual_channels / 2
    //  out: batch_size * weights_->residual_channels / 2
    auto thirdConvLayer = buildConvLayer(
        secondActivationConvLayer->getOutput(0),
        tower_ptr->conv3.GetFilter(),
        tower_ptr->conv3.GetWeights().size(),
        block_ptr->conv3.GetDevWeights(),
        tower_ptr->conv3.GetBiases().size(),
        block_ptr->conv3.GetDevBiases(),
        network,
        tower_ptr->conv3.GetOutputs());
    auto thirdActivationConvLayer = buildActivationLayer(
        thirdConvLayer->getOutput(0), network);
    // 4th conv layer (2nd block)
    //  in: batch_size * weights_->residual_channels / 2
    //  out: batch_size * weights_->residual_channels / 2
    auto fourthConvLayer = buildConvLayer(
        thirdActivationConvLayer->getOutput(0),
        tower_ptr->conv4.GetFilter(),
        tower_ptr->conv4.GetWeights().size(),
        block_ptr->conv4.GetDevWeights(),
        tower_ptr->conv4.GetBiases().size(),
        block_ptr->conv4.GetDevBiases(),
        network,
        tower_ptr->conv4.GetOutputs());
    auto fourthMergeLayer = network->addElementWise(
        *secondActivationConvLayer->getOutput(0),
        *fourthConvLayer->getOutput(0),
        ElementWiseOperation::kSUM);
    auto fourthActivationConvLayer = buildActivationLayer(
        fourthMergeLayer->getOutput(0), network);
    // post-bottleneck
    //  in: batch_size * weights_->residual_channels / 2
    //  out: batch_size * weights_->residual_channels
    auto postConvLayer = buildConvLayer(
        fourthActivationConvLayer->getOutput(0),
        tower_ptr->post_btl_conv.GetFilter(),
        tower_ptr->post_btl_conv.GetWeights().size(),
        block_ptr->post_btl_conv.GetDevWeights(),
        tower_ptr->post_btl_conv.GetBiases().size(),
        block_ptr->post_btl_conv.GetDevBiases(),
        network,
        tower_ptr->post_btl_conv.GetOutputs());
    if (tower_ptr->apply_se) {
        return buildSqueezeExcitationLayer(
            input,
            postConvLayer->getOutput(0),
            tower_ptr,
            block_ptr,
            network);
    } else {
        auto mergeLayer = network->addElementWise(
            *input,
            *postConvLayer->getOutput(0),
            ElementWiseOperation::kSUM);
        auto mergeActivationConvLayer = buildActivationLayer(
            mergeLayer->getOutput(0), network);
        return mergeActivationConvLayer;
    }
}

ILayer* CudaForwardPipe::NNGraph::buildMixerBlock(
    ITensor* input,
    BlockBasic* tower_ptr,
    NNGraph::Block* block_ptr,
    TrtUniquePtr<INetworkDefinition>& network) {

    // dw conv layer
    // Class: DepthwiseConvolution
    auto dwConvLayer = buildConvLayer(
        input,
        tower_ptr->dw_conv.GetFilter(),
        tower_ptr->dw_conv.GetWeights().size(),
        block_ptr->dw_conv.GetDevWeights(),
        tower_ptr->dw_conv.GetBiases().size(),
        block_ptr->dw_conv.GetDevBiases(),
        network,
        tower_ptr->dw_conv.GetOutputs(),
        tower_ptr->dw_conv.GetOutputs());
    auto mergeLayer = network->addElementWise(
        *input,
        *dwConvLayer->getOutput(0),
        ElementWiseOperation::kSUM);
    auto dwActivationConvLayer = buildActivationLayer(
        mergeLayer->getOutput(0), network);
    // 1st ffn conv layer
    // Class: Convolution
    auto firstConvLayer = buildConvLayer(
        dwActivationConvLayer->getOutput(0),
        tower_ptr->conv1.GetFilter(),
        tower_ptr->conv1.GetWeights().size(),
        block_ptr->conv1.GetDevWeights(),
        tower_ptr->conv1.GetBiases().size(),
        block_ptr->conv1.GetDevBiases(),
        network,
        tower_ptr->conv1.GetOutputs());
    auto firstActivationConvLayer = buildActivationLayer(
        firstConvLayer->getOutput(0), network);
    // 2nd ffn conv layer
    // Class: Convolution
    auto secondConvLayer = buildConvLayer(
        firstActivationConvLayer->getOutput(0),
        tower_ptr->conv2.GetFilter(),
        tower_ptr->conv2.GetWeights().size(),
        block_ptr->conv2.GetDevWeights(),
        tower_ptr->conv2.GetBiases().size(),
        block_ptr->conv2.GetDevBiases(),
        network,
        tower_ptr->conv2.GetOutputs());
    if (tower_ptr->apply_se) {
        return buildSqueezeExcitationLayer(
            input,
            secondConvLayer->getOutput(0),
            tower_ptr,
            block_ptr,
            network);
    } else {
        auto mergeLayer = network->addElementWise(
            *input,
            *secondConvLayer->getOutput(0),
            ElementWiseOperation::kSUM);
        auto mergeActivationConvLayer = buildActivationLayer(
            mergeLayer->getOutput(0), network);
        return mergeActivationConvLayer;
    }
}

ILayer* CudaForwardPipe::NNGraph::buildSqueezeExcitationLayer(
    ITensor* residual,
    ITensor* input,
    BlockBasic* tower_ptr,
    NNGraph::Block* block_ptr,
    TrtUniquePtr<INetworkDefinition>& network) {

    // squeeze-and-excitation module
    // in: batch_size * weights_->residual_channels * num_intersections
    // out: batch_size * weights_->residual_channels * 3
    auto gpoolLayer = applyGPoolLayer(input, network);
    //  in: batch_size * weights_->residual_channels * 3
    //  out: batch_size * 24(se_size_)
    //  Class: SEUnit
    auto fc1MatMulLayer = buildConvLayer(
        gpoolLayer->getOutput(0),
        1,
        tower_ptr->squeeze.GetWeights().size(),
        block_ptr->se_module.GetDevSqueezeWeights(),
        tower_ptr->squeeze.GetBiases().size(),
        block_ptr->se_module.GetDevSqueezeBiases(),
        network,
        tower_ptr->squeeze.GetOutputs());
    auto fc1ActivationMatLayer = buildActivationLayer(
        fc1MatMulLayer->getOutput(0), network);
    // in: batch_size * 24(se_size_)
    // out: batch_size * weights_->residual_channels * 2
    // Class: SEUnit
    auto fc2MatMulLayer = buildConvLayer(
        fc1ActivationMatLayer->getOutput(0),
        1,
        tower_ptr->excite.GetWeights().size(),
        block_ptr->se_module.GetDevExciteWeights(),
        tower_ptr->excite.GetBiases().size(),
        block_ptr->se_module.GetDevExciteBiases(),
        network,
        tower_ptr->excite.GetOutputs());
    auto gammaLayer = network->addSlice(
        *fc2MatMulLayer->getOutput(0),
        {4 ,{0, 0, 0, 0}},
        {4 ,{0, weights_->residual_channels, 1, 1}},
        {4 ,{1, 1, 1, 1}}
    );
    gammaLayer->setInput(2, *shapeLayer_->getOutput(0));
    auto biasLayer = network->addSlice(
        *fc2MatMulLayer->getOutput(0),
        {4 ,{0, weights_->residual_channels, 0, 0}},
        {4 ,{0, weights_->residual_channels, 1, 1}},
        {4 ,{1, 1, 1, 1}}
    );
    biasLayer->setInput(2, *shapeLayer_->getOutput(0));
    auto sigLayer = network->addActivation(
        *gammaLayer->getOutput(0), ActivationType::kSIGMOID);
    auto scaleLayer = network->addElementWise(
        *sigLayer->getOutput(0),
        *input,
        ElementWiseOperation::kPROD
    );
    auto seLayer = network->addElementWise(
        *scaleLayer->getOutput(0),
        *biasLayer->getOutput(0),
        ElementWiseOperation::kSUM
    );
    auto mergeLayer = network->addElementWise(
        *seLayer->getOutput(0),
        *residual,
        ElementWiseOperation::kSUM);
    auto outputConvLayer = buildActivationLayer(
        mergeLayer->getOutput(0), network);
    return outputConvLayer;
}

void CudaForwardPipe::NNGraph::buildPolicyHead(
    ITensor* input, TrtUniquePtr<INetworkDefinition>& network) {
    // policy head
    // in: batch_size * weights_->residual_channels * num_intersections
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // Class: Convolution
    auto policyConvLayer = buildConvLayer(
        input,
        weights_->p_hd_conv.GetFilter(),
        weights_->p_hd_conv.GetWeights().size(),
        graph_->p_hd_conv.GetDevWeights(),
        weights_->p_hd_conv.GetBiases().size(),
        graph_->p_hd_conv.GetDevBiases(),
        network,
        weights_->p_hd_conv.GetOutputs());
    auto actPolicyLayer = buildActivationLayer(
        policyConvLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->policy_head_channels)
    // out: batch_size * 64(weights_->policy_head_channels) * 3
    auto p_poolLayer = applyGPoolLayer(
        actPolicyLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->policy_head_channels) * 3
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // Class: FullyConnect
    int32_t const mmInputs_p1 = static_cast<int32_t>(
        p_poolLayer->getOutput(0)->getDimensions().d[1]
        * p_poolLayer->getOutput(0)->getDimensions().d[2]
        * p_poolLayer->getOutput(0)->getDimensions().d[3]);
    auto inputReshape_p1 = network->addShuffle(*p_poolLayer->getOutput(0));
    int32_t const variable_batch_p1 = static_cast<int32_t>(
        p_poolLayer->getOutput(0)->getDimensions().d[0]);
    inputReshape_p1->setReshapeDimensions(Dims{4, {variable_batch_p1, mmInputs_p1, 1, 1}});
    auto pol1MatMulLayer = buildConvLayer(
        inputReshape_p1->getOutput(0),
        1,
        weights_->p_inter_fc.GetWeights().size(),
        graph_->p_inter.GetDevWeights(),
        weights_->p_inter_fc.GetBiases().size(),
        graph_->p_inter.GetDevBiases(),
        network,
        weights_->p_inter_fc.GetOutputs());
    // in1: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // in2: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    auto mergeLayer = network->addElementWise(
        *actPolicyLayer->getOutput(0),
        *pol1MatMulLayer->getOutput(0),
        ElementWiseOperation::kSUM);
    // in: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 5(weights_->probabilities_channels) * num_intersections
    // Class: Convolution
    auto p_probConvLayer = buildConvLayer(
        mergeLayer->getOutput(0),
        weights_->prob_conv.GetFilter(),
        weights_->prob_conv.GetWeights().size(),
        graph_->p_prob.GetDevWeights(),
        weights_->prob_conv.GetBiases().size(),
        graph_->p_prob.GetDevBiases(),
        network,
        weights_->prob_conv.GetOutputs());
    // Mark the outputs for the network
    auto output_prob = p_probConvLayer->getOutput(0);
    network->markOutput(*output_prob);
    output_prob->setName("output_prob");
    output_prob->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    output_prob->setType(DataType::kFLOAT);

    // in: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 5(weights_->policy_head_channels)
    // Class: FullyConnect
    int32_t const mmInputs_p2 = static_cast<int32_t>(
        pol1MatMulLayer->getOutput(0)->getDimensions().d[1]
        * pol1MatMulLayer->getOutput(0)->getDimensions().d[2]
        * pol1MatMulLayer->getOutput(0)->getDimensions().d[3]);
    auto inputpassReshape_p2 = network->addShuffle(*pol1MatMulLayer->getOutput(0));
    int32_t const variable_batch_p2 = static_cast<int32_t>(
        pol1MatMulLayer->getOutput(0)->getDimensions().d[0]);
    inputpassReshape_p2->setReshapeDimensions(Dims{4, {variable_batch_p2, mmInputs_p2, 1, 1}});
    auto pol2MatMulLayer = buildConvLayer(
        inputpassReshape_p2->getOutput(0),
        1,
        weights_->pass_fc.GetWeights().size(),
        graph_->p_prob_pass.GetDevWeights(),
        weights_->pass_fc.GetBiases().size(),
        graph_->p_prob_pass.GetDevBiases(),
        network,
        weights_->pass_fc.GetOutputs());
    // Mark the outputs for the network
    auto output_prob_pass = pol2MatMulLayer->getOutput(0);
    network->markOutput(*output_prob_pass);
    output_prob_pass->setName("output_prob_pass");
    output_prob_pass->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    output_prob_pass->setType(DataType::kFLOAT);
}

void CudaForwardPipe::NNGraph::buildPolicyHeadRepLK(
    ITensor* input, TrtUniquePtr<INetworkDefinition>& network) {

    // policy head
    // in: batch_size * weights_->residual_channels * num_intersections
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // Class: Convolution
    auto policyConvLayer = buildConvLayer(
        input,
        weights_->p_hd_conv.GetFilter(),
        weights_->p_hd_conv.GetWeights().size(),
        graph_->p_hd_conv.GetDevWeights(),
        weights_->p_hd_conv.GetBiases().size(),
        graph_->p_hd_conv.GetDevBiases(),
        network,
        weights_->p_hd_conv.GetOutputs());
    auto actPolicyLayer = buildActivationLayer(
        policyConvLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // Class: DepthwiseConvolution
    auto p_dwConvLayer = buildConvLayer(
        actPolicyLayer->getOutput(0),
        weights_->p_dw_conv.GetFilter(),
        weights_->p_dw_conv.GetWeights().size(),
        graph_->p_dw_conv.GetDevWeights(),
        weights_->p_dw_conv.GetBiases().size(),
        graph_->p_dw_conv.GetDevBiases(),
        network,
        weights_->p_dw_conv.GetOutputs(),
        weights_->p_dw_conv.GetOutputs());
    auto p_dwActPolicyLayer = buildActivationLayer(
        p_dwConvLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // Class: Convolution
    auto p_ptConvLayer = buildConvLayer(
        p_dwActPolicyLayer->getOutput(0),
        weights_->p_pt_conv.GetFilter(),
        weights_->p_pt_conv.GetWeights().size(),
        graph_->p_pt_conv.GetDevWeights(),
        weights_->p_pt_conv.GetBiases().size(),
        graph_->p_pt_conv.GetDevBiases(),
        network,
        weights_->p_pt_conv.GetOutputs());
    auto p_ptActPolicyLayer = buildActivationLayer(
        p_ptConvLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->policy_head_channels)
    // out: batch_size * 64(weights_->policy_head_channels) * 3
    auto p_poolLayer = applyGPoolLayer(
        p_ptActPolicyLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->policy_head_channels) * 3
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // Class: FullyConnect
    int32_t const mmInputs_p1 = static_cast<int32_t>(
        p_poolLayer->getOutput(0)->getDimensions().d[1]
        * p_poolLayer->getOutput(0)->getDimensions().d[2]
        * p_poolLayer->getOutput(0)->getDimensions().d[3]);
    auto inputReshape_p1 = network->addShuffle(*p_poolLayer->getOutput(0));
    int32_t const variable_batch_p1 = static_cast<int32_t>(
        p_poolLayer->getOutput(0)->getDimensions().d[0]);
    inputReshape_p1->setReshapeDimensions(Dims{4, {variable_batch_p1, mmInputs_p1, 1, 1}});
    auto pol1MatMulLayer = buildConvLayer(
        inputReshape_p1->getOutput(0),
        1,
        weights_->p_inter_fc.GetWeights().size(),
        graph_->p_inter.GetDevWeights(),
        weights_->p_inter_fc.GetBiases().size(),
        graph_->p_inter.GetDevBiases(),
        network,
        weights_->p_inter_fc.GetOutputs());
    // in1: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // in2: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 64(weights_->policy_head_channels) * num_intersections
    auto mergeLayer = network->addElementWise(
        *p_ptActPolicyLayer->getOutput(0),
        *pol1MatMulLayer->getOutput(0),
        ElementWiseOperation::kSUM);
    // in: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 5(weights_->probabilities_channels) * num_intersections
    // Class: Convolution
    auto p_probConvLayer = buildConvLayer(
        mergeLayer->getOutput(0),
        weights_->prob_conv.GetFilter(),
        weights_->prob_conv.GetWeights().size(),
        graph_->p_prob.GetDevWeights(),
        weights_->prob_conv.GetBiases().size(),
        graph_->p_prob.GetDevBiases(),
        network,
        weights_->prob_conv.GetOutputs());
    // Mark the outputs for the network
    auto output_prob = p_probConvLayer->getOutput(0);
    network->markOutput(*output_prob);
    output_prob->setName("output_prob");
    output_prob->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    output_prob->setType(DataType::kFLOAT);

    // in: batch_size * 64(weights_->policy_head_channels) * num_intersections
    // out: batch_size * 5(weights_->policy_head_channels)
    // Class: FullyConnect
    int32_t const mmInputs_p2 = static_cast<int32_t>(
        pol1MatMulLayer->getOutput(0)->getDimensions().d[1]
        * pol1MatMulLayer->getOutput(0)->getDimensions().d[2]
        * pol1MatMulLayer->getOutput(0)->getDimensions().d[3]);
    auto inputpassReshape_p2 = network->addShuffle(*pol1MatMulLayer->getOutput(0));
    int32_t const variable_batch_p2 = static_cast<int32_t>(
        pol1MatMulLayer->getOutput(0)->getDimensions().d[0]);
    inputpassReshape_p2->setReshapeDimensions(Dims{4, {variable_batch_p2, mmInputs_p2, 1, 1}});
    auto pol2MatMulLayer = buildConvLayer(
        inputpassReshape_p2->getOutput(0),
        1,
        weights_->pass_fc.GetWeights().size(),
        graph_->p_prob_pass.GetDevWeights(),
        weights_->pass_fc.GetBiases().size(),
        graph_->p_prob_pass.GetDevBiases(),
        network,
        weights_->pass_fc.GetOutputs());
    // Mark the outputs for the network
    auto output_prob_pass = pol2MatMulLayer->getOutput(0);
    network->markOutput(*output_prob_pass);
    output_prob_pass->setName("output_prob_pass");
    output_prob_pass->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    output_prob_pass->setType(DataType::kFLOAT);
}

void CudaForwardPipe::NNGraph::buildValueHead(
    ITensor* input, TrtUniquePtr<INetworkDefinition>& network) {

    // value head
    // in: batch_size * weights_->residual_channels * num_intersections
    // out: batch_size * 64(weights_->value_head_channels)
    // Class: Convolution
    auto valueConvLayer = buildConvLayer(
        input,
        weights_->v_hd_conv.GetFilter(),
        weights_->v_hd_conv.GetWeights().size(),
        graph_->v_hd_conv.GetDevWeights(),
        weights_->v_hd_conv.GetBiases().size(),
        graph_->v_hd_conv.GetDevBiases(),
        network,
        weights_->v_hd_conv.GetOutputs());
    auto actValueLayer = buildActivationLayer(
        valueConvLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->value_head_channels)
    // out: batch_size * 64(weights_->value_head_channels) * 3
    auto v_poolLayer = applyGPoolLayer(
        actValueLayer->getOutput(0), network,
        true);
    // in: batch_size * 64(weights_->value_head_channels) * 3
    // out: batch_size * 64(weights_->value_head_channels) * 3
    // Class: FullyConnect
    int32_t const mmInputs_v1 = static_cast<int32_t>(
        v_poolLayer->getOutput(0)->getDimensions().d[1]
        * v_poolLayer->getOutput(0)->getDimensions().d[2]
        * v_poolLayer->getOutput(0)->getDimensions().d[3]);
    auto inputReshape_v1 = network->addShuffle(*v_poolLayer->getOutput(0));
    int32_t const variable_batch_v1 = static_cast<int32_t>(
        v_poolLayer->getOutput(0)->getDimensions().d[0]);
    inputReshape_v1->setReshapeDimensions(Dims{4, {variable_batch_v1, mmInputs_v1, 1, 1}});
    auto val1MatMulLayer = buildConvLayer(
        inputReshape_v1->getOutput(0),
        1,
        weights_->v_inter_fc.GetWeights().size(),
        graph_->v_inter.GetDevWeights(),
        weights_->v_inter_fc.GetBiases().size(),
        graph_->v_inter.GetDevBiases(),
        network,
        weights_->v_inter_fc.GetOutputs());
    auto val1ActValueLayer = buildActivationLayer(
        val1MatMulLayer->getOutput(0), network);
    // in: batch_size * 64(weights_->value_head_channels) * num_intersection
    // out: batch_size * 1(weights_->ownership_channels) * num_intersection
    // Class: Convolution
    auto v_ownershipConvLayer = buildConvLayer(
        actValueLayer->getOutput(0),
        weights_->v_ownership.GetFilter(),
        weights_->v_ownership.GetWeights().size(),
        graph_->v_ownership.GetDevWeights(),
        weights_->v_ownership.GetBiases().size(),
        graph_->v_ownership.GetDevBiases(),
        network,
        weights_->v_ownership.GetOutputs());
    // Mark the outputs for the network
    auto output_ownership = v_ownershipConvLayer->getOutput(0);
    network->markOutput(*output_ownership);
    output_ownership->setName("output_ownership");
    output_ownership->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    output_ownership->setType(DataType::kFLOAT);

    // in: batch_size * 64(weights_->value_head_channels) * 3
    // out: batch_size * 15(weights_->value_misc_outputs)
    // Class: FullyConnect
    int32_t const mmInputs_v2 = static_cast<int32_t>(
        val1ActValueLayer->getOutput(0)->getDimensions().d[1]
        * val1ActValueLayer->getOutput(0)->getDimensions().d[2]
        * val1ActValueLayer->getOutput(0)->getDimensions().d[3]);
    auto inputReshape_v2 = network->addShuffle(*val1ActValueLayer->getOutput(0));
    int32_t const variable_batch_v2 = static_cast<int32_t>(
        val1ActValueLayer->getOutput(0)->getDimensions().d[0]);
    inputReshape_v2->setReshapeDimensions(Dims{4, {variable_batch_v2, mmInputs_v2, 1, 1}});
    auto val2MatMulLayer = buildConvLayer(
        inputReshape_v2->getOutput(0),
        1,
        weights_->v_misc.GetWeights().size(),
        graph_->v_misc.GetDevWeights(),
        weights_->v_misc.GetBiases().size(),
        graph_->v_misc.GetDevBiases(),
        network,
        weights_->v_misc.GetOutputs());
    // Mark the outputs for the network
    auto output_val = val2MatMulLayer->getOutput(0);
    network->markOutput(*output_val);
    output_val->setName("output_val");
    output_val->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    output_val->setType(DataType::kFLOAT);
}

ILayer* CudaForwardPipe::NNGraph::buildConvLayer(
    ITensor* input,
    unsigned int filter_size,
    int64_t weights_size,
    void* weights,
    int64_t biases_size,
    void* biases,
    TrtUniquePtr<INetworkDefinition>& network,
    unsigned int outputs,
    const int groups) {

    auto data_type = handles_.fp16 ? DataType::kHALF : DataType::kFLOAT;
    // For convenience, both I/O tensors have 3 dimentions (in addition to batch), so that
    // matmul is mathmatically equivalent to a 2D convolution of 1x1 features and 1x1 kernels.
    auto convLayer = network->addConvolutionNd(
        *input,
        outputs,
        {2, {filter_size, filter_size}},
        {
            data_type,
            weights,
            weights_size
        },
        {
            data_type,
            biases,
            biases_size
        }
    );
    convLayer->setDilationNd({2, {1, 1}});
    convLayer->setPaddingMode(PaddingMode::kSAME_UPPER);
    convLayer->setNbGroups(groups);
    return convLayer;
}

ILayer* CudaForwardPipe::NNGraph::buildActivationLayer(
    ITensor* input,
    TrtUniquePtr<INetworkDefinition>& network) {

    ILayer* actLayer = nullptr;

    if (weights_->default_act == Activation::kIdentity) {
        actLayer = network->addIdentity(*input);
    } else if (weights_->default_act == Activation::kReLU) {
        actLayer = network->addActivation(*input, ActivationType::kRELU);
    } else if (weights_->default_act == Activation::kELU) {
        actLayer = network->addActivation(*input, ActivationType::kELU);
    } else if (weights_->default_act == Activation::kSELU) {
        actLayer = network->addActivation(*input, ActivationType::kSELU);
    } else if (weights_->default_act == Activation::kGELU) {
        actLayer = network->addActivation(*input, ActivationType::kGELU_TANH);
    } else if (weights_->default_act == Activation::kMISH) {
        auto softplusLayer = network->addActivation(*input, ActivationType::kSOFTPLUS);
        auto tanhLayer = network->addActivation(*softplusLayer->getOutput(0), ActivationType::kTANH);
        actLayer = network->addElementWise(*input, *tanhLayer->getOutput(0), ElementWiseOperation::kPROD);
    } else if (weights_->default_act == Activation::kSwish) {
        auto sigmoidLayer = network->addActivation(*input, ActivationType::kSIGMOID);
        actLayer = network->addElementWise(*input, *sigmoidLayer->getOutput(0), ElementWiseOperation::kPROD);
    } else if (weights_->default_act == Activation::kHardSwish) {
        auto sigmoidLayer = network->addActivation(*input, ActivationType::kHARD_SIGMOID);
        sigmoidLayer->setAlpha(3.0);
        sigmoidLayer->setBeta(6.0);
        actLayer = network->addElementWise(*input, *sigmoidLayer->getOutput(0), ElementWiseOperation::kPROD);
    }
    return actLayer;
}

ILayer* CudaForwardPipe::NNGraph::applyGPoolLayer(
    ITensor* input,
    TrtUniquePtr<INetworkDefinition>& network,
    const bool isValueHead) {

    ILayer* gpoolMeanLayer =
        network->addReduce(*input, ReduceOperation::kAVG, 1U << 2 | 1U << 3, true);

    auto gpoolMeanScaleLayer = network->addElementWise(
        *gpoolMeanLayer->getOutput(0),
        *maskScaleLayer_->getOutput(0),
        ElementWiseOperation::kPROD);

    ILayer* gpoolMaskAddLayer = nullptr;
    ILayer* gpoolMaskShiftLayer = nullptr;
    ILayer* gpoolConcatInputLayer3 = nullptr;
    if(isValueHead) {
        auto gpoolMeanQuadLayer = network->addElementWise(
            *gpoolMeanLayer->getOutput(0),
            *maskQuadLayer_->getOutput(0),
            ElementWiseOperation::kPROD);
        gpoolConcatInputLayer3 = gpoolMeanQuadLayer;
    } else {
        auto gpoolMaxLayer =
            network->addReduce(*input, ReduceOperation::kMAX, 1U << 2 | 1U << 3, true);
        gpoolConcatInputLayer3 = gpoolMaxLayer;
    }

    ITensor* gpoolConcatInputs[] = {
        gpoolMeanLayer->getOutput(0),
        gpoolMeanScaleLayer->getOutput(0),
        gpoolConcatInputLayer3->getOutput(0)};
    auto gpoolConcatLayer = network->addConcatenation(gpoolConcatInputs, 3);
    gpoolConcatLayer->setAxis(1);

    return gpoolConcatLayer;
}
#endif
