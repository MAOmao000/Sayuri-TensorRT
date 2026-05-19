import json
import os
import torch

class Config:
    def __init__(self, inputs, is_file=True):
        if is_file:
            self.read(inputs)
        else:
            self.parse(inputs)

    def read(self, filename):
        with open(filename, "r") as f:
           json_str = f.read()
        self.parse(json_str)

    def parse(self, json_str):
        self.json_str = json_str
        jdata = json.loads(self.json_str)
        self.parse_training_config(jdata)
        self.parse_nn_config(jdata)

    def parse_training_config(self, json_data):
        train = json_data.get("Train", None)

        self.optimizer = train.get("Optimizer", "SGD")
        self.use_gpu = torch.cuda.is_available() if train.get("UseGPU", None) is None else train.get("UseGPU")
        self.use_fp16 = train.get("UseFp16", False) if self.use_gpu else False
        self.weight_decay = train.get("WeightDecay", 1e-4)
        self.lr_schedule = train.get("LearningRateSchedule", [[0, 0.2]])

        self.train_dir = train.get("TrainDirectory", None)
        self.validation_dir = train.get("ValidationDirectory", None)
        self.store_path = train.get("StorePath", None)
        self.batchsize = train.get("BatchSize", 512)
        self.buffersize = train.get("BufferSize", 16 * 1000)
        self.macrofactor = train.get("MacroFactor", 1)
        self.macrobatchsize = self.batchsize // self.macrofactor

        self.num_workers = train.get("Workers", max(os.cpu_count()-2, 1))
        self.steps_per_epoch = train.get("StepsPerEpoch", 1000)
        self.validation_steps = train.get("ValidationSteps", 100)
        self.verbose_steps = train.get("VerboseSteps", 1000)
        self.max_steps_per_running = train.get("MaxStepsPerRunning", 16384000)
        self.down_sample_rate = train.get("DownSampleRate", 16)
        self.num_chunks = train.get("NumberChunks", None)
        self.chunks_increasing_c = train.get("ChunksIncreasingC", None)
        self.chunks_increasing_scale = train.get("ChunksIncreasingScale", 1.0)
        self.chunks_increasing_alpha = train.get("ChunksIncreasingAlpha", 0.75)
        self.chunks_increasing_beta = train.get("ChunksIncreasingBeta", 0.4)
        self.soft_loss_weight = train.get("SoftLossWeight", 0.1)
        self.swa_max_count = train.get("SwaMaxCount", 16)
        self.swa_steps = train.get("SwaSteps", 100)
        self.warmup_steps = train.get("WarmUpSteps", 0)
        self.renorm_max_r = train.get("RenormMaxR", 1)
        self.renorm_max_d = train.get("RenormMaxD", 0)
        self.policy_surprise_factor = train.get("PolicySurpriseFactor", 0.0)
        self.export_onnx = train.get("ExportONNX", False)
        self.use_compile = train.get("UseCompile", False)

        assert self.train_dir != None, ""
        assert self.store_path != None, ""

    def parse_nn_config(self, json_data):
        network = json_data.get("NeuralNetwork", None)

        self.boardsize = network.get("MaxBoardSize", 19)
        self.nntype = network.get("NNType", None)
        self.activation = network.get("Activation", "relu")
        self.input_channels = network.get("InputChannels", 43)
        self.residual_channels = network.get("ResidualChannels", None)

        self.policy_head_type = network.get("PolicyHeadType", { "Type" : "Normal" })
        self.policy_head_channels = network.get("PolicyExtract", None) # v1 ~ v4 net
        if self.policy_head_channels is None:
            self.policy_head_channels = network.get("PolicyHeadChannels", None) # since v5 net
        self.value_head_channels = network.get("ValueExtract", None) # v1 ~ v4 net
        if self.value_head_channels is None:
            self.value_head_channels = network.get("ValueHeadChannels", None) # since v5 net
        self.se_ratio = network.get("SeRatio", 2)
        self.stack = network.get("Stack", [])
        self.netname_postfix = network.get("NamePostfix", "")
        self.mode = network.get("BatchNormMode", "renorm")
        self.is_pre_act = network.get("IsPreAct", False)
        self.use_rope = network.get("UseRoPE", False)
        self.rope_theta = network.get("RoPETheta", 100.0)
        self.learnable_rope = network.get("LearnableRoPE", False)
        self.attention_qk_norm = network.get("AttentionQKNorm", False)
        self.use_gab = network.get("UseGAB", False)
        self.gab_d1 = network.get("GABD1", 16)
        self.gab_d2 = network.get("GABD2", 16)
        self.use_tab = network.get("UseTAB", False)
        self.inline_registers = network.get("InlineRegisters", False)
        self.attention_num_rw_registers = network.get("AttentionNumRWRegisters", 0)
        self.gab_num_templates = network.get("GABNumTemplates", None)
        self.gab_num_fourier_features = network.get("GABNumFourierFeatures", None)
        self.gab_mlp_hidden = network.get("GABMLPHidden", None)
        self.tab_c_z = network.get("TABCZ", None)
        self.tab_num_templates = network.get("TABNumTemplates", None)
        self.tab_num_freqs = network.get("TABNumFreqs", None)
        self.tab_num_blocks = network.get("TABNumBlocks", None)
        self.tab_dilation = network.get("TABDilation", None)
        self.tab_use_frequency_mixing = network.get("TABUseFrequencyMixing", False)
        self.use_swiglu = network.get("UseSwiGLU", True)
        self.transformer_ffn_depthwise_conv = network.get("TransformerFFNDepthwiseConv", False)

        assert self.input_channels != None, ""
        assert self.residual_channels != None, ""
        assert self.policy_head_channels != None, ""
        assert self.value_head_channels != None, ""
