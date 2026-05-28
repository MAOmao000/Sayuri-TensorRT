import faulthandler
faulthandler.enable()
import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import random, time, math, os, glob, io, gzip, sys
import argparse

from config import Config
from network import Network
from data import Data

from torch.nn import DataParallel
from lazy_loader import LazyLoader, LoaderFlag
from status_dict import StatusDict

def stderr_write(val):
    sys.stderr.write(val)
    sys.stderr.flush()

def stdout_write(val):
    sys.stdout.write(val)
    sys.stdout.flush()

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, adjust_lr_fn="match_rms_adamw"):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    if adjust_lr_fn == "match_rms_adamw":
        update *= 0.2 * max(update.size(-2), update.size(-1))**0.5
    elif adjust_lr_fn == "original":
        update *= max(1, update.size(-2) / update.size(-1))**0.5
    else:
        raise AssertionError(f"Unexpected value {adjust_lr_fn=}")
    return update

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
        adjust_lr_fn: Either "original" or "match_rms_adamw"
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, adjust_lr_fn="match_rms_adamw"):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, adjust_lr_fn=adjust_lr_fn)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss

class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, adjust_lr_fn="match_rms_adamw"):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups, adjust_lr_fn="match_rms_adamw", adam_betas=(0.95, 0.995), adam_eps=1e-6):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["adjust_lr_fn"] = group.get("adjust_lr_fn", adjust_lr_fn)
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", adam_betas)
                group["eps"] = group.get("eps", adam_eps)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups, adjust_lr_fn="match_rms_adamw", adam_betas=(0.95, 0.995), adam_eps=1e-6):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["adjust_lr_fn"] = group.get("adjust_lr_fn", adjust_lr_fn)
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", adam_betas)
                group["eps"] = group.get("eps", adam_eps)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

class ONNXExportWrapper(torch.nn.Module):
    """
    Wrapper class to handle the model's forward pass for ONNX export.
    This handles the complex output structure and makes it ONNX-compatible.
    """
    def __init__(self, model):
        super(ONNXExportWrapper, self).__init__()
        self.model = model
    
    def forward(self, input_feature: torch.Tensor):
        """
        Forward pass that returns a flattened tuple of outputs for ONNX compatibility.
        """
        # Call the original model
        outputs = self.model(input_feature)
        outputs = outputs[0]
        pruned_outputs = tuple([outputs[i] for i in [0, 1, 2, 3]])
        return pruned_outputs

def gather_filenames(root, num_chunks=None, sort_key_fn=None):
    def gather_recursive_files(root):
        l = list()
        for name in glob.glob(os.path.join(root, "*")):
            if os.path.isdir(name):
                l.extend(gather_recursive_files(name))
            else:
                l.append(name)
        return l
    chunks = gather_recursive_files(root)

    if num_chunks:
        if len(chunks) > num_chunks:
            if sort_key_fn is None:
                random.shuffle(chunks)
            else:
                chunks.sort(key=sort_key_fn, reverse=True)
            chunks = chunks[:num_chunks]
    return chunks

class StreamLoader:
    def __init__(self):
        pass

    def func(self, filename):
        stream = None
        if not os.path.isfile(filename):
            return stream

        try:
            if filename.find(".gz") >= 0:
                with gzip.open(filename, "rt") as f:
                    stream = io.StringIO(f.read())
            else:
                with open(filename, "r") as f:
                    stream = io.StringIO(f.read())
        except:
            stdout_write("Could not open the file: {}\n".format(filename))
        return stream

class StreamParser:
    def __init__(self, down_sample_rate, policy_surprise_factor=0.0):
        # Use a random sample input data read. This helps improve the spread of
        # games in the shuffle buffer.
        self.down_sample_rate = down_sample_rate

        # We assume each game has 50 positions on average. 
        self.virtual_buffsize = 8000 * 50
        self.running_kld_mean = 1.0
        self.policy_surprise_factor = policy_surprise_factor
        self.num_samples_per_proc = 0

    def _sample(self, data):
        # quick find out the KLD running average so we increase the factor value in
        # the early samples.
        gamma_factor = math.exp((max(self.virtual_buffsize - self.num_samples_per_proc, 0))/ \
                            (self.virtual_buffsize/2.71828182846))
        gamma = (1.0/self.virtual_buffsize) * gamma_factor
        self.running_kld_mean = (1.0 - gamma) * self.running_kld_mean + \
                                    gamma * data.kld
        self.num_samples_per_proc += 1

        # compute policy surprise sample probability
        surprise_freq = (1.0 - self.policy_surprise_factor) + \
                              self.policy_surprise_factor * (data.kld/self.running_kld_mean)
        sample_prob = surprise_freq * (1.0 / self.down_sample_rate)

        if self.num_samples_per_proc < self.virtual_buffsize:
            # be sure each worker sees enough games
            return False
        return sample_prob > random.uniform(0.0, 1.0)

    def func(self, stream):
        if not stream:
            return None

        while True:
            data = Data()

            if not data.load_from_stream(stream):
                return None # stream is end

            if self._sample(data):
                data.parse()
                break
        data.apply_symmetry(random.randint(0, 7))
        return data

class BatchGenerator:
    def __init__(self, boardsize, input_channels):
        self.nn_board_size = boardsize
        self.nn_num_intersections = self.nn_board_size * self.nn_board_size
        self.input_channels = input_channels

    def _wrap_data(self, data):
        nn_board_size = self.nn_board_size
        nn_num_intersections = self.nn_num_intersections

        board_size = data.board_size
        num_intersections = data.board_size * data.board_size

        # allocate all buffers
        input_planes = np.zeros((self.input_channels, nn_board_size, nn_board_size))
        prob = np.zeros(nn_num_intersections+1)
        aux_prob = np.zeros(nn_num_intersections+1)
        ownership = np.zeros((nn_board_size, nn_board_size))
        wdl = np.zeros(3)
        all_q_vals = np.zeros(5)
        all_scores = np.zeros(5)

        buf = np.zeros(num_intersections)
        sqr_buf = np.zeros((nn_board_size, nn_board_size))

        # input planes
        for p in range(self.input_channels-6):
            plane = data.planes[p]
            input_planes[p, 0:board_size, 0:board_size] = np.reshape(plane, (board_size, board_size))[:, :]

        input_planes[self.input_channels-6, 0:board_size, 0:board_size] = data.rule
        input_planes[self.input_channels-5, 0:board_size, 0:board_size] = data.wave
        if data.to_move == 1:
            input_planes[self.input_channels-4, 0:board_size, 0:board_size] =  data.komi/20
            input_planes[self.input_channels-3, 0:board_size, 0:board_size] = -data.komi/20
        else:
            input_planes[self.input_channels-4, 0:board_size, 0:board_size] = -data.komi/20
            input_planes[self.input_channels-3, 0:board_size, 0:board_size] =  data.komi/20
        input_planes[self.input_channels-2, 0:board_size, 0:board_size] = (data.board_size**2)/361
        input_planes[self.input_channels-1, 0:board_size, 0:board_size] = 1 # fill ones

        # probabilities
        buf[:] = data.prob[0:num_intersections]
        sqr_buf[0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]
        prob[0:nn_num_intersections] = np.reshape(sqr_buf, (nn_num_intersections))[:]
        prob[nn_num_intersections] = data.prob[num_intersections]

        # auxiliary probabilities
        buf[:] = data.aux_prob[0:num_intersections]
        sqr_buf[0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]
        aux_prob[0:nn_num_intersections] = np.reshape(sqr_buf, (nn_num_intersections))[:]
        aux_prob[nn_num_intersections] = data.aux_prob[num_intersections]

        # ownership
        ownership[0:board_size, 0:board_size] = np.reshape(data.ownership, (board_size, board_size))[:, :]
        ownership = np.reshape(ownership, (nn_num_intersections))

        # winrate
        wdl[1 - data.result] = 1

        # all q values
        all_q_vals[0] = data.result
        all_q_vals[1] = data.avg_q
        all_q_vals[2] = data.short_avg_q
        all_q_vals[3] = data.mid_avg_q
        all_q_vals[4] = data.long_avg_q

        # all scores
        all_scores[0] = data.final_score
        all_scores[1] = data.avg_score
        all_scores[2] = data.short_avg_score
        all_scores[3] = data.mid_avg_score
        all_scores[4] = data.long_avg_score

        # weight value
        weight = data.weight

        return (
            input_planes,
            prob,
            aux_prob,
            ownership,
            wdl,
            all_q_vals,
            all_scores,
            weight
        )

    def func(self, data_list):
        batch_planes = list()
        batch_prob = list()
        batch_aux_prob = list()
        batch_ownership = list()
        batch_wdl = list()
        batch_q_vals = list()
        batch_scores = list()
        batch_weight = list()

        for data in data_list:
            planes, prob, aux_prob, ownership, wdl, q_vals, scores, weight = self._wrap_data(data)

            batch_planes.append(planes)
            batch_prob.append(prob)
            batch_aux_prob.append(aux_prob)
            batch_ownership.append(ownership)
            batch_wdl.append(wdl)
            batch_q_vals.append(q_vals)
            batch_scores.append(scores)
            batch_weight.append(weight)

        batch_dict = {
            "planes"    : torch.from_numpy(np.array(batch_planes)).float(),
            "prob"      : torch.from_numpy(np.array(batch_prob)).float(),
            "aux_prob"  : torch.from_numpy(np.array(batch_aux_prob)).float(),
            "ownership" : torch.from_numpy(np.array(batch_ownership)).float(),
            "wdl"       : torch.from_numpy(np.array(batch_wdl)).float(),
            "q_vals"    : torch.from_numpy(np.array(batch_q_vals)).float(),
            "scores"    : torch.from_numpy(np.array(batch_scores)).float(),
            "weight"    : torch.from_numpy(np.array(batch_weight)).float(),
        }
        return batch_dict

class TrainingPipe():
    def __init__(self, cfg):
        self.cfg = cfg

        # The mini-batch size, update the network per batch size
        self.batchsize =  cfg.batchsize

        # The marco batch size and factor, (marco batch size) * factor = batch size
        self.macrobatchsize = cfg.macrobatchsize
        self.macrofactor = cfg.macrofactor

        # Number of cpu for the 'DataLoader'.
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir
        self.validation_dir = cfg.validation_dir

        # How many last chunks do we load?
        self.num_chunks = cfg.num_chunks
        self.chunks_increasing_c = cfg.chunks_increasing_c
        self.chunks_increasing_scale = cfg.chunks_increasing_scale
        self.chunks_increasing_alpha = cfg.chunks_increasing_alpha
        self.chunks_increasing_beta = cfg.chunks_increasing_beta
        self.num_all_chunks = 0

        # The stpes of storing the last model and validating it per epoch.
        self.steps_per_epoch =  cfg.steps_per_epoch

        # Report the information per this steps.
        self.verbose_steps = max(100, cfg.verbose_steps)

        self.validation_steps = cfg.validation_steps

        # Lazy loader options.
        self.train_buffer_size = self.cfg.buffersize
        self.validation_buffer_size = self.train_buffer_size // 4
        self.down_sample_rate = cfg.down_sample_rate

        # Steps information
        self.current_steps = 0
        self.max_steps_per_running = cfg.max_steps_per_running
        self.max_steps = self.max_steps_per_running + self.current_steps
        self.current_samples = 0

        # Which optimizer do we use?
        self.opt_name = cfg.optimizer

        # Optimizer's parameters.
        self.weight_decay = cfg.weight_decay
        self.lr_schedule = cfg.lr_schedule

        # The training device.
        self.use_gpu = cfg.use_gpu
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.use_fp16 = cfg.use_fp16
        self.scaler = torch.amp.GradScaler("cuda") if self.use_fp16 else None
        self.net = Network(cfg)
        self.net.train()
        self.netname_postfix = cfg.netname_postfix

        # The Store root directory.
        self.store_path = cfg.store_path
        self._status_dict = StatusDict()

        # The SWA setting.
        self.swa_count = 0
        self.swa_max_count = self.cfg.swa_max_count
        self.swa_steps = self.cfg.swa_steps
        self.swa_net = Network(cfg)
        self.swa_net.eval()

        # Warm up steps.
        self.warmup_steps = self.cfg.warmup_steps

        # The sample rate factor for policy
        self.policy_surprise_factor = self.cfg.policy_surprise_factor

        self._setup()

    def _setup(self):
        if self.cfg.use_compile:
            self.net = self.net.to(self.device)
            self.module = self.net # linking
            self.net = torch.compile(
                self.net,
                backend="inductor",
                options={
                    "triton.autotune_cublasLt": False,  # Disable GEMM auto-tuning
                    "triton.cudagraphs": True  # Keep CUDA Graphs enabled to maintain performance
                }
            )
            self.swa_net = self.swa_net.to(self.device)
        else:
            self.module = self.net # linking
            if self.use_gpu:
                self.net = self.net.to(self.device)
                self.net = DataParallel(self.net) 
                self.module  = self.net.module
                self.swa_net = self.swa_net.to(self.device)

        # Copy the initial weights.
        self.swa_net.accumulate_swa(self.module, 0)

        init_lr = self._get_lr_schedule(0)

        # We may fail to load the optimizer. So initializing
        # it before loading it.
        self.opt = None
        if self.opt_name == "Adam":
            self.opt = torch.optim.Adam(
                self.net.parameters(),
                lr=init_lr,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "Muon":
            self.opt = SingleDeviceMuonWithAuxAdam(self._get_param_groups())
        elif self.opt_name == "SGD" or not self.opt_name in ["Adam", "SGD"]:
            # Recommanded optimizer, the SGD is better than Adam
            # in this kind of training task.
            self.opt = torch.optim.SGD(
                self.net.parameters(),
                lr=init_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay,
            )

        # create workspace
        if not os.path.isdir(self.store_path):
            os.makedirs(self.store_path)

        self.weights_path = os.path.join(self.store_path, "weights")
        if not os.path.isdir(self.weights_path):
            os.mkdir(self.weights_path)

        self.swa_weights_path = os.path.join(self.store_path, "swa")
        if not os.path.isdir(self.swa_weights_path):
            os.mkdir(self.swa_weights_path)

        self.checkpoint_path = os.path.join(self.store_path, "checkpoint")
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        info_file = os.path.join(self.store_path, "info.txt")
        with open(info_file, 'w') as f:
            f.write(self.module.simple_info())

        self._loss_weight_dict = {
            "soft" : self.cfg.soft_loss_weight
        }

    def _get_is_muon_suitable(self, group_name):
        if group_name == "normal" or group_name == "normal_attn" or group_name == "normal_gab" or group_name == "gab_mlp" or group_name == "tab_module":
            return True
        elif group_name in ["normal_gamma", "noreg", "output", "output_noreg", "input", "input_noreg"]:
            return False
        else:
            assert False

    def _get_weight_decay(self, group_name):
        effective_lr_scale = 1.0
        is_muon_suitable = self._get_is_muon_suitable(group_name=group_name)
        if self.opt_name == "Muon":
            batch_scaling = math.sqrt(self.batchsize / 256.0)
        else:
            batch_scaling = self.batchsize / 256.0
        if self.cfg.mode == "fixup":
            if (
                group_name == "input" or
                group_name == "normal" or
                group_name == "normal_gamma" or
                group_name == "output"
            ):
                if self.opt_name == "Muon":
                    return 0.005000 * batch_scaling
                else:
                    return 0.000001 * batch_scaling
            elif group_name == "normal_attn" or group_name == "normal_gab" or group_name == "gab_mlp" or group_name == "tab_module":
                if self.opt_name == "Muon":
                    return 0.005000 * 0.5 * batch_scaling
                else:
                    return 0.000001 * 0.5 * batch_scaling
            elif group_name == "input_noreg" or group_name == "noreg":
                return 0.00000001 * batch_scaling
            elif group_name == "output_noreg":
                return 0.00000001 * batch_scaling
            else:
                assert False
        elif self.cfg.mode == "renorm" or self.cfg.mode == "norm":
            warmup_scale = self._get_lr_schedule(self.current_steps) / self.lr_schedule[-1][1]
            adaptive_scale = 1.0
            if (group_name == "input" or group_name == "normal" or group_name == "normal_attn" or group_name == "normal_gab" or group_name == "normal_gamma" or group_name == "gab_mlp" or group_name == "tab_module"):
                if self.opt_name == "Muon":
                    wd_with_lr_scale = math.pow(effective_lr_scale * warmup_scale, 0.70) * adaptive_scale
                else:
                    wd_with_lr_scale = math.pow(effective_lr_scale * warmup_scale, 0.75) * adaptive_scale

                if group_name == "input":
                    # Branch here is mostly preserving inconsistent historical behavior, there's not
                    # a great reason these should be different.
                    if self.opt_name == "Muon":
                        wd_group_factor = 2.0 / 3.0
                    else:
                        wd_group_factor = 1.0
                elif group_name == "normal":
                    wd_group_factor = 1.0
                elif group_name == "normal_attn":
                    wd_group_factor = 0.5
                elif group_name == "normal_gab":
                    wd_group_factor = 0.3
                elif group_name == "gab_mlp":
                    wd_group_factor = 0.1
                elif group_name == "tab_module":
                    wd_group_factor = 0.1
                elif group_name == "normal_gamma":
                    # Batch norm gammas can be regularized a bit less,
                    # doing them just as much empirically seemed to be a bit more unstable
                    if self.opt_name == "Muon":
                        wd_group_factor = 0.25
                    else:
                        wd_group_factor = 0.125
                else:
                    assert False

                if self.opt_name == "Muon" and not is_muon_suitable:
                    return 0.00900 * batch_scaling * wd_with_lr_scale * wd_group_factor
                elif self.opt_name == "Muon":
                    return 0.02000 * batch_scaling * wd_with_lr_scale * wd_group_factor
                else:
                    return 0.00125 * batch_scaling * wd_with_lr_scale * wd_group_factor

            elif group_name == "output":
                if self.opt_name == "Muon" and not is_muon_suitable:
                    return 0.00400 * batch_scaling
                elif self.opt_name == "Muon":
                    assert False
                else:
                    return 0.000001 * batch_scaling
            elif group_name == "input_noreg" or group_name == "noreg":
                return 0.000001 * batch_scaling * math.pow(effective_lr_scale * warmup_scale, 0.75)
            elif group_name == "output_noreg":
                if self.opt_name == "Muon" and not is_muon_suitable:
                    return 0.000001 * batch_scaling
                elif self.opt_name == "Muon":
                    assert False
                else:
                    return 0.00000001 * batch_scaling
            else:
                assert False
        else:
            assert False

    def _get_param_groups(self):
        reg_dict : Dict[str,List] = {}
        self.module.add_reg_dict(reg_dict)
        param_groups = []
        num_reg_dict_params = 0
        for group_name in reg_dict:
            if len(reg_dict[group_name]) > 0:
                param_groups.append({
                    "params": reg_dict[group_name],
                    "group_name": group_name,
                })
                num_reg_dict_params += len(reg_dict[group_name])

        for group in param_groups:
            group["weight_decay"] = self._get_weight_decay(group_name=group["group_name"])
            group["use_muon"] = self._get_is_muon_suitable(group_name=group["group_name"])

        num_params = 0
        for param in self.net.parameters():
            if param.requires_grad:
                num_params += 1
        assert num_params == num_reg_dict_params, "Reg dict does not have entries for all params in model"
        return param_groups

    def _get_lr_schedule(self, num_steps):
        # Get the current learning rate from schedule.
        curr_lr = 0.2
        for s, lr in self.lr_schedule:
            if s <= num_steps:
                curr_lr = lr
            else:
                break

        if self.warmup_steps > 0 and num_steps < self.warmup_steps:
            curr_lr = curr_lr * ((num_steps+1)/self.warmup_steps)
        return curr_lr

    def _load_current_status(self):
        sort_fn = os.path.getmtime
        files = gather_filenames(self.checkpoint_path, 1, sort_key_fn=sort_fn)
        if len(files) == 0:
            self._status_dict.clear()
        else:
            checkpoint = files.pop()
            self._status_dict.load(checkpoint, device=torch.device("cpu"))
        self._status_dict.load_module(StatusDict.MODEL_KEY, self.module)
        self._status_dict.load_module(StatusDict.SWA_KEY, self.swa_net)
        self._status_dict.load_module(StatusDict.OPTIM_KEY,self.opt)

        self.current_samples = self._status_dict.fancy_get(StatusDict.SAMPLES_KEY)
        self.current_steps = self._status_dict.fancy_get(StatusDict.STEPS_KEY)
        self.module.update_parameters(self.current_steps)

        self.swa_count = self._status_dict.fancy_get(StatusDict.SWA_COUNT_KEY)

        curr_lr = self._get_lr_schedule(self.current_steps)

        if self.opt_name == "Muon":
            for param in self.opt.param_groups:
                if param["group_name"] == "normal" or param["group_name"] == "normal_attn" or param["group_name"] == "normal_gab" or param["group_name"] == "gab_mlp" or param["group_name"] == "tab_module":
                    param["lr"] = curr_lr * 2.0
                elif param["group_name"] == "output" or param["group_name"] == "output_noreg":
                    param["lr"] = curr_lr * 0.5
                else:
                    param["lr"] = curr_lr
                param["weight_decay"] = self._get_weight_decay(group_name=param["group_name"])
        else:
            for param in self.opt.param_groups:
                param["lr"] = curr_lr
                param["weight_decay"] = self.weight_decay

        self.max_steps = self.max_steps_per_running + self.current_steps
        stdout_write("Current steps is {}. Will stop the training at {}.\n".format(self.current_steps, self.max_steps))

    def _save_current_status(self):
        self._validate_the_last_model()
        netname = "{}{}".format(self.module.get_name(), self.netname_postfix)
        status = "{}-s{}-c{}".format(netname, self.current_steps, self.num_all_chunks)
        if not self.num_chunks is None:
            status += "-w{}".format(self.num_chunks)

        checkpoint = os.path.join(
            self.checkpoint_path, "{}-status.pt".format(status))
        self._status_dict.set_module(StatusDict.MODEL_KEY, self.module)
        self._status_dict.set_module(StatusDict.SWA_KEY, self.swa_net)
        self._status_dict.set_module(StatusDict.OPTIM_KEY, self.opt)
        self._status_dict.fancy_set(StatusDict.STEPS_KEY, self.current_steps)
        self._status_dict.fancy_set(StatusDict.SAMPLES_KEY, self.current_samples)
        self._status_dict.fancy_set(StatusDict.SWA_COUNT_KEY, self.swa_count)
        self._status_dict.fancy_set(StatusDict.JSON_KEY, self.cfg.json_str)
        self._status_dict.save(checkpoint)

        if self.cfg.export_onnx:
            onnx_model_name = os.path.join(
                self.swa_weights_path, "{}-swa.onnx".format(status))
            # Set model to evaluation mode
            self.swa_net.eval()
            # Create wrapper for ONNX export
            wrapper = ONNXExportWrapper(self.swa_net)
            wrapper.eval()
            # Create dummy inputs
            batch_size = 8 
            input_feature = torch.randn(
                batch_size,
                self.cfg.input_channels,
                self.cfg.boardsize,
                self.cfg.boardsize,
                dtype=torch.float32,
                device=self.device
            )
            input_feature[:,0,:,:]=1.0
            # Prepare inputs and input names
            inputs = [input_feature]
            input_names = ['InputFeature']
            output_names = ['output_prob', 'output_prob_pass', 'output_val', 'output_ownership']
            # For dynamo, we need to provide dynamic_shapes as a dict or tuple
            # Now that forward has explicit arguments, we can use a dict matching input names
            dynamic_shapes = {'input_feature': {0: "batch_size"}}
            dynamic_axes = None # dynamo uses dynamic_shapes
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,                   # model to export
                    tuple(inputs),             # inputs of the model
                    onnx_model_name,           # filename of the ONNX model
                    export_params=True,        # When ``f`` is specified: If false, parameters (weights) will not be exported
                    # opset_version=20,          # The version of the default (ai.onnx) opset to target
                    # do_constant_folding=True,  # Deprecated option
                    input_names=input_names,   # Rename inputs for the ONNX model
                    output_names=output_names, # Rename outputs for the ONNX model
                    dynamic_axes=dynamic_axes, # Deprecated: Prefer specifying dynamic_shapes when dynamo=True
                    dynamic_shapes=dynamic_shapes, # A dictionary or a tuple of dynamic shapes for the model inputs
                    verbose=False,             # Whether to enable verbose logging
                    dynamo=True,               # True or False to select the exporter to use
                    external_data=False,       # Whether to save the model weights as an external data file
                    report=False               # Whether to generate a markdown report for the export process
                )
            # Add metadata to the ONNX model
            try:
                import onnx
                from onnx import helper
                onnx_model = onnx.load(onnx_model_name)
                # Add metadata_props
                cpu_swa_net = self.swa_net.to("cpu")
                meta = cpu_swa_net.get_meta_data()
                # Clear existing metadata if any to avoid duplicates
                if hasattr(onnx_model, "metadata_props"):
                    del onnx_model.metadata_props[:]
                for key, value in meta.items():
                    meta_entry = onnx_model.metadata_props.add()
                    meta_entry.key = key
                    meta_entry.value = value
                # Save the model with metadata
                onnx.save(onnx_model, onnx_model_name)
                if self.use_gpu:
                    self.swa_net = self.swa_net.to(self.device)
            except ImportError:
                raise AssertionError(f"onnx package not installed, skipping metadata addition.")
            except Exception as e:
                raise AssertionError(f"Failed to add metadata: {e}")
        else:
            weights_name = os.path.join(
                self.weights_path, "{}.bin.txt".format(status))
            cpu_module = self.module.to("cpu")
            cpu_module.transfer_to_bin(weights_name)

            swa_weights_name = os.path.join(
                self.swa_weights_path, "{}-swa.bin.txt".format(status))
            cpu_swa_net = self.swa_net.to("cpu")
            cpu_swa_net.transfer_to_bin(swa_weights_name)

            if self.use_gpu:
                self.module = self.module.to(self.device)
                self.swa_net = self.swa_net.to(self.device)

    def _init_loader(self):
        def compute_window_size(N, c=5000, scale=1.0, alpha=0.75, beta=0.4):
            return round(scale * c * (1 + beta * (math.pow(N/c, alpha) - 1) / alpha))

        self._stream_loader = StreamLoader()
        self._stream_parser = StreamParser(self.down_sample_rate, self.policy_surprise_factor)
        self._batch_gen = BatchGenerator(self.cfg.boardsize, self.cfg.input_channels)

        self.num_all_chunks = len(gather_filenames(self.train_dir))
        if not self.chunks_increasing_c is None:
            # Compute the best window size for self-play learning. The formula is based on
            # "Accelerating Self-Play Learning in Go". Please see the formula here,
            # https://arxiv.org/abs/1902.10565v5
            num_chunks_for_window = compute_window_size(
                N = self.num_all_chunks,
                c = self.chunks_increasing_c,
                scale = self.chunks_increasing_scale,
                alpha = self.chunks_increasing_alpha,
                beta = self.chunks_increasing_beta
            )
            num_chunks_upper = self.num_all_chunks \
                if self.num_chunks is None else min(self.num_all_chunks, self.num_chunks)
            self.num_chunks = min(num_chunks_upper, num_chunks_for_window)
        sort_fn = os.path.getmtime
        chunks = gather_filenames(self.train_dir, self.num_chunks, sort_fn)

        stdout_write("Load the last {} chunks from all {} chunks...\n".format(len(chunks), self.num_all_chunks))

        self.train_flag = LoaderFlag()
        self.train_lazy_loader = LazyLoader(
            filenames = chunks,
            stream_loader = self._stream_loader,
            stream_parser = self._stream_parser,
            batch_generator = self._batch_gen,
            num_workers = self.num_workers,
            buffer_size = self.train_buffer_size,
            batch_size = self.macrobatchsize,
            flag = self.train_flag
        )
        # Try to get the first batch, be sure that the loader is ready.
        batch = next(self.train_lazy_loader)

        if self.validation_dir and os.path.isdir(self.validation_dir):
            self.validation_flag = LoaderFlag()
            self.validation_lazy_loader = LazyLoader(
                filenames = gather_filenames(self.validation_dir, len(chunks), sort_fn),
                stream_loader = self._stream_loader,
                stream_parser = self._stream_parser,
                batch_generator = self._batch_gen,
                num_workers = max(1, round(0.25 * self.num_workers)),
                buffer_size = self.validation_buffer_size,
                batch_size = self.macrobatchsize,
                flag = self.validation_flag
            )
            # Try to get the first batch, be sure that the loader is ready.
            batch = next(self.validation_lazy_loader)
            self.validation_flag.set_suspend_flag()
        else:
            self.validation_dir = None
            self.validation_lazy_loader = None

    def _break_loader(self):
        self.train_flag.set_stop_flag()
        try:
            _, _ = self._gather_data_from_loader(True)
        except StopIteration:
            pass

        if self.validation_lazy_loader:
            self.validation_flag.set_stop_flag()
            try:
                _, _ = self._gather_data_from_loader(False)
            except StopIteration:
                pass

    def _get_new_running_loss_dict(self, all_loss_dict):
        running_loss_dict = dict()
        running_loss_dict["steps"] = 0
        for k, v in all_loss_dict.items():
            running_loss_dict[k] = 0
        return running_loss_dict

    def _accumulate_loss(self, running_loss_dict, all_loss_dict):
        for k, v in all_loss_dict.items():
            running_loss_dict[k] += v.item()
        running_loss_dict["steps"] += 1./self.macrofactor
        return running_loss_dict

    def _handle_loss(self, all_loss_dict):
        for k, v in all_loss_dict.items():
            v /= self.macrofactor
        loss = all_loss_dict["loss"]
        return loss, all_loss_dict

    def _get_current_info(self, speed, running_loss_dict):
        info = str()
        info += "[steps: {}, samples: {}, speed: {:.2f}, learning rate: {}, batch size: {}] ->".format(
                    self.current_steps,
                    self.current_samples,
                    speed,
                    self.opt.param_groups[0]["lr"],
                    self.batchsize)
        steps = running_loss_dict["steps"]
        info += " all loss: {:.4f},".format(running_loss_dict["loss"]/steps)
        info += " prob loss: {:.4f},".format(running_loss_dict["prob_loss"]/steps)
        info += " aux prob loss: {:.4f},".format(running_loss_dict["aux_prob_loss"]/steps)
        info += " soft prob loss: {:.4f},".format(running_loss_dict["soft_prob_loss"]/steps)
        info += " soft aux prob loss: {:.4f},".format(running_loss_dict["soft_aux_prob_loss"]/steps)
        info += " optimistic loss: {:.4f},".format(running_loss_dict["optimistic_loss"]/steps)
        info += " ownership loss: {:.4f},".format(running_loss_dict["ownership_loss"]/steps)
        info += " wdl loss: {:.4f},".format(running_loss_dict["wdl_loss"]/steps)
        info += " Q values loss: {:.4f},".format(running_loss_dict["q_vals_loss"]/steps)
        info += " scores loss: {:.4f},".format(running_loss_dict["scores_loss"]/steps)
        info += " errors loss: {:.4f}".format(running_loss_dict["errors_loss"]/steps)
        return info

    def _save_current_info(self, speed, running_loss_dict, filename):
        line_log = self._get_current_info(speed, running_loss_dict)
        stdout_write(line_log + "\n")

        log_file = os.path.join(self.store_path, filename)
        with open(log_file, 'a') as f:
            f.write(line_log + "\n")

    def _gather_data_from_loader(self, use_training=True):
        # Fetch the next batch data from disk.
        if use_training:
            batch_dict = next(self.train_lazy_loader)
        else:
            if self.validation_lazy_loader is None:
                return None, None
            batch_dict = next(self.validation_lazy_loader)

        # Move the data to the current device.
        if self.use_gpu:
            for k, v in batch_dict.items():
                v = v.to(self.device)

        # Gather batch data
        planes = batch_dict["planes"]
        target = (
            batch_dict["prob"],
            batch_dict["aux_prob"],
            batch_dict["ownership"],
            batch_dict["wdl"],
            batch_dict["q_vals"],
            batch_dict["scores"],
            batch_dict["weight"]
        )
        return planes, target

    def _validate_the_last_model(self):
        if self.validation_lazy_loader is None:
            return
        stdout_write("Validate the network performance...\n")
        self.net.eval()
        self.validation_flag.reset_flag()
        self.train_flag.set_suspend_flag()

        running_loss_dict = dict()
        total_steps = self.validation_steps * self.macrofactor
        clock_time = time.time()

        with torch.no_grad():
            for _ in range(total_steps):
                planes, target = self._gather_data_from_loader(False)
                if self.cfg.use_compile:
                    planes = planes.to(self.device)
                    target_to = ()
                    for batch_dict in target:
                        batch_dict = batch_dict.to(self.device)
                        target_to += (batch_dict, )
                    _, all_loss_dict = self.net(
                        planes,
                        target=target_to,
                        use_symm=True,
                        loss_weight_dict=self._loss_weight_dict
                    )
                else:
                    _, all_loss_dict = self.net(
                        planes,
                        target=target,
                        use_symm=True,
                        loss_weight_dict=self._loss_weight_dict
                    )
                _, all_loss_dict = self._handle_loss(all_loss_dict)

                if len(running_loss_dict) == 0:
                    running_loss_dict = self._get_new_running_loss_dict(all_loss_dict)
                running_loss_dict = self._accumulate_loss(running_loss_dict, all_loss_dict)

        elapsed = time.time() - clock_time
        self._save_current_info(self.validation_steps/elapsed, running_loss_dict, "validation.log")
        self.net.train()
        self.validation_flag.set_suspend_flag()
        self.train_flag.reset_flag()

    def fit_and_store(self):
        self._load_current_status()
        self._init_loader()

        stdout_write("Start training loop...\n")

        running_loss_dict = dict()
        keep_running = True # true will keep the loop
        macro_steps = 0 # actually network forwarding times
        clock_time = time.time() # current time stamp

        while keep_running:
            for _ in range(self.steps_per_epoch):
                planes, target = self._gather_data_from_loader(True)

                # forward and backward for loss
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    if self.cfg.use_compile:
                        planes = planes.to(self.device)
                        target_to = ()
                        for batch_dict in target:
                            batch_dict = batch_dict.to(self.device)
                            target_to += (batch_dict, )
                        _, all_loss_dict = self.net(
                            planes,
                            target=target_to,
                            use_symm=True,
                            loss_weight_dict=self._loss_weight_dict
                        )
                    else:
                        _, all_loss_dict = self.net(
                            planes,
                            target=target,
                            use_symm=True,
                            loss_weight_dict=self._loss_weight_dict
                        )
                    loss, all_loss_dict = self._handle_loss(all_loss_dict)

                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                else:
                    loss.backward()
                macro_steps += 1

                # accumulate loss
                if len(running_loss_dict) == 0:
                    running_loss_dict = self._get_new_running_loss_dict(all_loss_dict)
                running_loss_dict = self._accumulate_loss(running_loss_dict, all_loss_dict)
 
                if math.isnan(running_loss_dict["loss"]):
                    stdout_write("The gradient is explosion. Stop the training...\n")
                    keep_running = False
                    break

                if macro_steps % self.macrofactor == 0:
                    # clip grad
                    if self.opt_name != "Muon" and (self.cfg.mode == "renorm" or self.cfg.mode == "norm"):
                        gnorm = torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), 10000.0).detach().cpu().item()
                    else:
                        if self.opt_name == "Muon":
                            gnorm_cap = 11000.0
                        elif self.cfg.mode == "fixup":
                            gnorm_cap = 2500.0
                        else:
                            gnorm_cap = 5500.0
                        gnorm_cap *= math.sqrt(self.batchsize / 256.0)
                        # gnorm_cap /= math.sqrt(8.0)
                        gnorm = torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), max_norm=gnorm_cap).detach().cpu().item()

                    # update network parameters
                    if self.use_fp16:
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        self.opt.step()
                    self.opt.zero_grad() 

                    # update current count
                    self.current_steps += 1
                    self.current_samples += self.batchsize
                    self.module.update_parameters(self.current_steps)

                    if self.current_steps % self.verbose_steps == 0:
                        # update time stamp
                        elapsed = time.time() - clock_time
                        clock_time = time.time()

                        # dump the verbose and save the log file
                        self._save_current_info(self.verbose_steps/elapsed, running_loss_dict, "training.log")
                        running_loss_dict = self._get_new_running_loss_dict(all_loss_dict)

                    # update SWA model
                    if self.current_steps % self.swa_steps == 0:
                        self.swa_count = min(self.swa_count+1, self.swa_max_count)
                        self.swa_net.accumulate_swa(self.module, self.swa_count)

                    # update learning rate
                    # for param in self.opt.param_groups:
                    #     param["lr"] = self._get_lr_schedule(self.current_steps)
                    curr_lr = self._get_lr_schedule(self.current_steps)
                    if self.opt_name == "Adam" or self.opt_name == "SGD":
                        for param in self.opt.param_groups:
                            param["lr"] = curr_lr
                    else:
                        for param in self.opt.param_groups:
                            if param["group_name"] == "normal" or param["group_name"] == "normal_attn" or param["group_name"] == "normal_gab" or param["group_name"] == "gab_mlp" or param["group_name"] == "tab_module":
                                param["lr"] = curr_lr * 2.0
                            elif param["group_name"] == "output" or param["group_name"] == "output_noreg":
                                param["lr"] = curr_lr * 0.5
                            else:
                                param["lr"] = curr_lr
                            if self.cfg.mode == "renorm" or self.cfg.mode == "norm":
                                param["weight_decay"] = self._get_weight_decay(group_name=param["group_name"])

                # stop the training if achieving max steps
                if self.current_steps >= self.max_steps:
                    keep_running = False
                    break

            # store the last network
            self._save_current_status()
        self._break_loader()
        stdout_write("Training is over.\n")

def train_process(args):
    cfg = Config(args.json)

    # overwrite some values
    if not args.workspace is None:
        cfg.store_path = args.workspace

    pipe = TrainingPipe(cfg)
    pipe.fit_and_store()

import gc
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The setting json file name.", type=str)
    parser.add_argument("-w", "--workspace", metavar="<string>",
                        help="Overwrite the store path.", type=str)
    args = parser.parse_args()

    if args.json == None:
        stdout_write("Please give the setting json file.\n")
    else:
        train_process(args)
    gc.disable()  # Disable garbage collection at program termination.
