import torch
import torch.nn as nn
import torch.nn.functional as F
import packaging
import packaging.version
import numpy as np
import math
import struct
import sys

from symmetry import torch_symmetry

CRAZY_NEGATIVE_VALUE = -5000.0
DEFAULT_ACTIVATION = "relu"

def activation_func(activation, inplace=False):
    if activation == "identity":
        return nn.Identity()
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    if activation == "elu":
        return nn.ELU(inplace=inplace)
    if activation == "selu":
        return nn.SELU(inplace=inplace)
    if activation == "gelu":
        return nn.GELU(inplace=inplace)
    if activation == "mish":
        return nn.Mish(inplace=inplace)
    if activation == "swish":
        return nn.SiLU(inplace=inplace)
    if activation == "hardswish":
        if packaging.version.parse(torch.__version__) > packaging.version.parse("1.6.0"):
            return nn.Hardswish(inplace=inplace)
        else:
            return nn.Hardswish()
    raise Exception("The {} is invalid activation function.".format(activation))

def compute_gain(activation):
    if activation == "identity":
        gain = 1.0
    elif activation == "relu":
        gain = math.sqrt(2.0)
    elif activation == "elu":
        gain = math.sqrt(1.55052)
    elif activation == "selu":
        gain = 3/4
    elif activation == "gelu":
        gain = math.sqrt(2.351718)
    elif activation == "mish":
        gain = math.sqrt(2.210277)
    elif activation == "swish":
        gain = math.sqrt(2.0) # TODO:
    elif activation == "hardswish":
        gain = math.sqrt(2.0)
    else:
        raise Exception("The {} is invalid activation function for computing gain.".format(activation))
    return gain

def dwconv_to_text(in_channels, out_channels, kernel_size):
    return "DepthwiseConvolution {iC} {oC} {KS}\n".format(
               iC=in_channels,
               oC=out_channels,
               KS=kernel_size)

def conv_to_text(in_channels, out_channels, kernel_size):
    return "Convolution {iC} {oC} {KS}\n".format(
               iC=in_channels,
               oC=out_channels,
               KS=kernel_size)

def fullyconnect_to_text(in_size, out_size):
    return "FullyConnect {iS} {oS}\n".format(iS=in_size, oS=out_size)

def bn_to_text(channels):
    return "BatchNorm {C}\n".format(C=channels)

def float_to_bin(num, big_endian):
    fmt = 'f'
    if big_endian:
        fmt = '!' + fmt
    return struct.pack(fmt, num)

def bin_to_float(bnum, big_endian):
    fmt = 'f'
    if big_endian:
        fmt = '!' + fmt
    return struct.unpack(fmt, bnum)[0]

def str_to_bin(st):
    return bytearray(st, "utf-8")

def ffffffff_nan():
    return b'\xff\xff\xff\xff'

def tensor_to_list(t: torch.Tensor):
    return t.detach().cpu().numpy().ravel()

def tensor_to_bin(t: torch.Tensor):
    return b''.join([float_to_bin(w, False) for w in tensor_to_list(t)]) + ffffffff_nan()

def tensor_to_text(t: torch.Tensor, use_bin):
    if use_bin:
        return tensor_to_bin(t)
    return " ".join([str(w) for w in tensor_to_list(t)]) + "\n"

def init_weights(tensor, activation, scale, fan_tensor=None):
    gain = compute_gain(activation)

    if fan_tensor is not None:
        (fan_in, _) = torch.nn.init._calculate_fan_in_and_fan_out(fan_tensor)
    else:
        (fan_in, _) = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    target_std = scale * gain / math.sqrt(fan_in)
    # Multiply slightly since we use truncated normal
    std = target_std / 0.87962566103423978
    if std < 1e-10:
        tensor.fill_(0.0)
    else:
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0*std, b=2.0*std)

# It is imported from KataGo.
class SoftPlusWithGradientFloorFunction(torch.autograd.Function):
    """
    Same as softplus, except on backward pass, we never let the gradient decrease below grad_floor.
    Equivalent to having a dynamic learning rate depending on stop_grad(x) where x is the input.
    If square, then also squares the result while halving the input, and still also keeping the same gradient.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, grad_floor: float, square: bool):
        ctx.save_for_backward(x)
        ctx.grad_floor = grad_floor # grad_floor is not a tensor
        if square:
            return torch.square(F.softplus(0.5 * x))
        else:
            return F.softplus(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_floor = ctx.grad_floor
        grad_x = None
        grad_grad_floor = None
        grad_square = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (grad_floor + (1.0 - grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, grad_grad_floor, grad_square

class CustomIdentity(nn.Module):
    def __init__(self):
        super(CustomIdentity, self).__init__()

    def forward(self, x, mask_buffers):
        return x

class GlobalPool(nn.Module):
    def __init__(self, is_value_head=False):
        super(GlobalPool, self).__init__()
        self.b_avg = (19 + 9) / 2
        self.b_variance = 0.1

        self.is_value_head = is_value_head

    def forward(self, x, mask_buffers):
        mask, mask_sum_hw, mask_sum_hw_sqrt = mask_buffers
        b, c, h, w = x.size()

        div = torch.reshape(mask_sum_hw, (-1,1))
        div_sqrt = torch.reshape(mask_sum_hw_sqrt, (-1,1))

        layer_raw_mean = torch.sum(x, dim=(2,3), keepdims=False) / div
        b_diff = div_sqrt - self.b_avg

        if self.is_value_head:
            # According to KataGo, we compute three orthogonal values. There
            # are 1, (x-14)/10, and (x-14)^2/100 - 0.1. They may improve the value
            # head performance. That because the win-rate and score lead heads consist
            # of komi and intersections.

            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)
            layer2 = layer_raw_mean * (torch.square(b_diff) / 100.0 - self.b_variance)

            layer_pooled = torch.cat((layer0, layer1, layer2), 1)
        else:
            # Apply CRAZY_NEGATIVE_VALUE to out of board area. I guess that 
            # -5000 is large enough.
            raw_x = x + (1.0-mask) * CRAZY_NEGATIVE_VALUE

            layer_raw_max = torch.max(torch.reshape(raw_x, (b,c,h*w)), dim=2, keepdims=False)[0]
            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)
            layer2 = layer_raw_max

            layer_pooled = torch.cat((layer0, layer1, layer2), 1)

        return layer_pooled

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels,
                       se_size,
                       mode,
                       activation,
                       collector=None):
        super(SqueezeAndExcitation, self).__init__()

        self.activation = activation
        self.global_pool = GlobalPool(is_value_head=False)
        self.channels = channels

        self.squeeze = FullyConnect(
            in_size=self.channels * 3,
            out_size=se_size,
            activation=self.activation,
            collector=collector
        )
        if mode == "fixup":
            self.pre_bias_squeeze = BatchNorm2d(
                num_features=self.channels * 3,
                mode=mode
            )
        else:
            self.pre_bias_squeeze = CustomIdentity()

        self.excite = FullyConnect(
            in_size=se_size,
            out_size=self.channels * 2,
            activation="identity",
            collector=collector
        )
        if mode == "fixup":
            self.pre_bias_excite = BatchNorm2d(
                num_features=se_size,
                mode=mode
            )
        else:
            self.pre_bias_excite = CustomIdentity()

    def initialize(self, scale, xavier_init):
        self.squeeze.initialize(scale=scale, xavier_init=xavier_init)
        self.excite.initialize(scale=scale, xavier_init=xavier_init)

    def add_reg_dict(self, reg_dict):
        self.squeeze.add_reg_dict(reg_dict)
        self.excite.add_reg_dict(reg_dict)
        if not isinstance(self.pre_bias_squeeze, CustomIdentity):
            self.pre_bias_squeeze.add_reg_dict(reg_dict)
        if not isinstance(self.pre_bias_excite, CustomIdentity):
            self.pre_bias_excite.add_reg_dict(reg_dict)

    def forward(self, x, mask_buffers):
        b, c, _, _ = x.size()
        mask, _, _ = mask_buffers

        seprocess = self.global_pool(x, mask_buffers)
        seprocess = self.pre_bias_squeeze(seprocess, mask_buffers)
        seprocess = self.squeeze(seprocess)
        seprocess = self.pre_bias_excite(seprocess, mask_buffers)
        seprocess = self.excite(seprocess)

        gammas, betas = torch.split(seprocess, self.channels, dim=1)
        gammas = torch.reshape(gammas, (b, c, 1, 1))
        betas = torch.reshape(betas, (b, c, 1, 1))

        out = torch.sigmoid(gammas) * x + betas
        return out * mask

class BatchNorm2d(nn.Module):
    def __init__(self, num_features,
                       eps=1e-5,
                       momentum=0.01,
                       use_gamma=False,
                       mode="renorm",
                       renorm_clipping={"rmax" : 1, "dmax" : 0},
                       momentum_basic_batchsize=None):
        super(BatchNorm2d, self).__init__()

        if mode == "renorm" or mode == "norm":
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.float)
            )
            self.register_buffer(
                "running_var", torch.ones(num_features, dtype=torch.float)
            )

        if use_gamma:
            self.gamma = torch.nn.Parameter(
                torch.ones(num_features, dtype=torch.float)
            )
        else:
            self.gamma = torch.nn.Parameter(
                torch.ones(num_features, dtype=torch.float),
                requires_grad=False
            )

        self.beta = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )

        self.use_gamma = use_gamma
        self.num_features = num_features
        self.eps = eps
        self.momentum = self._clamp(momentum)
        self.momentum_basic_batchsize = momentum_basic_batchsize

        self.mode = mode
        assert self.mode in ["norm", "renorm", "fixup"]

        # According to the paper "Batch Renormalization: Towards Reducing Minibatch Dependence
        # in Batch-Normalized Models", Batch-Renormalization is much faster and steady than 
        # traditional Batch-Normalized when batch size is very small, eg bs=4.
        self.use_renorm = mode == "renorm"
        if self.use_renorm:
            self.rmax = renorm_clipping["rmax"]
            self.dmax = renorm_clipping["dmax"]

        # Fixup Batch Normalization layer. According to kataGo, Batch Normalization may cause
        # some wierd reuslts becuse the inference and training computation results are different.
        # Fixup can avoid the weird forwarding result. Fixup also speeds up the performance. The
        # improvement may be around x1.6 ~ x1.8 faster.
        self.fixup = mode == "fixup"

    def add_reg_dict(self, reg_dict, placement="in_block"):
        if placement == "in_block":
            if self.use_gamma:
                reg_dict["normal_gamma"].append(self.gamma)
            reg_dict["noreg"].append(self.beta)
        elif placement == "before_block":
            if self.use_gamma:
                reg_dict["input"].append(self.gamma)
            reg_dict["input_noreg"].append(self.beta)
        else:
            if self.use_gamma:
                reg_dict["output"].append(self.gamma)
            reg_dict["output_noreg"].append(self.beta)

    def get_merged_params(self):
        bn_mean = torch.zeros(self.num_features)
        bn_std = torch.zeros(self.num_features)

        # Merge four tensors (mean, variance, gamma, beta) into two tensors (
        # mean, variance).
        bn_mean[:] = self.running_mean[:]
        bn_std[:] = torch.sqrt(self.eps + self.running_var)[:]

        # Original format: gamma * ((x-mean) / std) + beta
        # Target format: (x-mean) / std
        #
        # Solve the following equation:
        #     gamma * ((x-mean) / std) + beta = (x-tgt_mean) / tgt_std
        #
        # We will get:
        #     tgt_std = std / gamma
        #     tgt_mean = mean - beta * (std / gamma)

        bn_std = bn_std / self.gamma
        bn_mean = bn_mean - self.beta * bn_std
        return bn_mean, bn_std

    def _clamp(self, x, lower=0., upper=1.):
        x = max(lower, x)
        x = min(upper, x)
        return x

    def _get_momentum(self, x):
        if self.momentum_basic_batchsize is None:
            return self.momentum
        b, _, _, _ = x.shape
        return self.momentum * math.sqrt(b / self.momentum_basic_batchsize)

    def _apply_renorm(self, x, mean, var):
        mean = mean.view(1, self.num_features, 1, 1)
        std = torch.sqrt(var+self.eps).view(1, self.num_features, 1, 1)
        running_std = torch.sqrt(self.running_var+self.eps).view(1, self.num_features, 1, 1)
        running_mean = self.running_mean.view(1, self.num_features, 1, 1)

        r = (
            std.detach() / running_std
        ).clamp(1 / self.rmax, self.rmax)

        d = (
            (mean.detach() - running_mean) / running_std
        ).clamp(-self.dmax, self.dmax)

        x = (x-mean)/std * r + d
        return x

    def _apply_norm(self, x, mean, var):
        mean = mean.view(1, self.num_features, 1, 1)
        std = torch.sqrt(var+self.eps).view(1, self.num_features, 1, 1)
        x = (x-mean)/std
        return x

    def forward(self, x, mask):
        if self.training and not self.fixup:
            mask_sum = torch.sum(mask) # global sum

            batch_mean = torch.sum(x, dim=(0,2,3)) / mask_sum
            zmtensor = x - batch_mean.view(1, self.num_features, 1, 1)
            batch_var = torch.sum(torch.square(zmtensor * mask), dim=(0,2,3)) / mask_sum

            if self.use_renorm:
                x = self._apply_renorm(x , batch_mean, batch_var)
            else:
                x = self._apply_norm(x , batch_mean, batch_var)

            # Update moving averages.
            momentum = self._get_momentum(x)
            self.running_mean += momentum * (batch_mean.detach() - self.running_mean)
            self.running_var += momentum * (batch_var.detach() - self.running_var)
        elif not self.fixup:
            # Inference step or fixup, they are equal.
            x = self._apply_norm(x, self.running_mean, self.running_var)

        if x.dim() == 4:
            x = x * (self.gamma.view(1, self.num_features, 1, 1))
            x = x + self.beta.view(1, self.num_features, 1, 1)
            return x * mask
        else:
            x = x * (self.gamma.view(1, self.num_features))
            x = x + self.beta.view(1, self.num_features)
            return x

        return x * mask

class BroadcastDepthwiseConv2d(nn.Module):
    def __init__(self, channels,
                       kernel_size,
                       padding="same",
                       bias=True):
        super(BroadcastDepthwiseConv2d, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = bias

        self.weight = nn.Parameter(
            torch.randn((self.channels, 1, self.kernel_size, self.kernel_size), dtype=torch.float)
        )
        self.gamma = nn.Parameter(
            torch.ones(self.channels) / math.sqrt(self.channels)
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.channels, dtype=torch.float)
            )

    def _compute_equivalent_weight(self):
        return self.weight + torch.sum(self.weight * self.gamma.view(self.channels, 1, 1, 1), dim=0, keepdim=True)

    def add_reg_dict(self, reg_dict, placement="in_block"):
        if placement == "in_block":
            reg_dict["normal"].append(self.weight)
            if self.use_bias:
                reg_dict["noreg"].append(self.bias)
            reg_dict["normal_gamma"].append(self.gamma)
        elif placement == "before_block":
            reg_dict["input"].append(self.weight)
            if self.use_bias:
                reg_dict["input_noreg"].append(self.bias)
            reg_dict["input"].append(self.gamma)
        else:
            reg_dict["output"].append(self.weight)
            if self.use_bias:
                reg_dict["output_noreg"].append(self.bias)
            reg_dict["output"].append(self.gamma)

    def get_merged_params(self):
        weight = torch.zeros_like(self.weight)
        bias = torch.zeros(self.channels)

        weight[:] = self._compute_equivalent_weight().detach()[:]
        if self.use_bias:
            bias[:] = self.bias[:]
        return weight, bias

    def forward(self, x):
        weight = self._compute_equivalent_weight()
        x = F.conv2d(
            x,
            weight,
            padding=self.padding,
            groups=self.channels
        )
        if self.use_bias:
            x = x + self.bias.view(1, self.channels, 1, 1)
        return x

class FullyConnect(nn.Module):
    def __init__(self, in_size,
                       out_size,
                       activation,
                       collector=None):
        super(FullyConnect, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(
            in_size,
            out_size,
            bias=True
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._try_collect(collector)

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def initialize(self, scale, xavier_init, bias_scale=0.2):
        if xavier_init:
            nn.init.xavier_normal_(
                self.linear.weight, gain=compute_gain(self.activation))
            nn.init.zeros_(self.linear.bias)
        else:
            init_weights(self.linear.weight, self.activation, scale=scale)
            init_weights(self.linear.bias, self.activation, scale=bias_scale, fan_tensor=self.linear.weight)

    def add_reg_dict(self, reg_dict, placement="in_block"):
        if placement == "in_block":
            reg_dict["normal"].append(self.linear.weight)
            reg_dict["noreg"].append(self.linear.bias)
        elif placement == "before_block":
            reg_dict["input"].append(self.linear.weight)
            reg_dict["input_noreg"].append(self.linear.bias)
        else:
            reg_dict["output"].append(self.linear.weight)
            reg_dict["output_noreg"].append(self.linear.bias)

    def shape_to_text(self):
        return fullyconnect_to_text(self.in_size, self.out_size)

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.linear.weight, use_bin)
        out += tensor_to_text(self.linear.bias, use_bin)
        return out

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x

class Convolve(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       activation,
                       collector=None):
        super(Convolve, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=True,
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._try_collect(collector)

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def initialize(self, scale, xavier_init, bias_scale=0.2):
        if xavier_init:
            nn.init.xavier_normal_(
                self.conv.weight, gain=compute_gain(self.activation))
            nn.init.zeros_(self.conv.bias)
        else:
            init_weights(self.conv.weight, self.activation, scale=scale)
            init_weights(self.conv.bias, self.activation, scale=bias_scale, fan_tensor=self.conv.weight)

    def add_reg_dict(self, reg_dict, placement="in_block"):
        if placement == "in_block":
            reg_dict["normal"].append(self.conv.weight)
            reg_dict["noreg"].append(self.conv.bias)
        elif placement == "before_block":
            reg_dict["input"].append(self.conv.weight)
            reg_dict["input_noreg"].append(self.conv.bias)
        else:
            reg_dict["output"].append(self.conv.weight)
            reg_dict["output_noreg"].append(self.conv.bias)

    def shape_to_text(self):
        return conv_to_text(self.in_channels, self.out_channels, self.kernel_size)

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.conv.weight, use_bin)
        out += tensor_to_text(self.conv.bias, use_bin)
        return out

    def forward(self, x, mask):
        x = self.conv(x) * mask
        x = self.act(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       use_gamma,
                       mode,
                       placement,
                       renorm_clipping,
                       activation,
                       collector=None):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )

        if mode == "fixup" and placement == "in_block":
            self.pre_bias = BatchNorm2d(
                num_features=in_channels,
                use_gamma=False,
                mode=mode
            )
        else:
            self.pre_bias = CustomIdentity()

        self.bn = BatchNorm2d(
            num_features=out_channels,
            use_gamma=use_gamma,
            mode=mode,
            renorm_clipping=renorm_clipping,
            momentum_basic_batchsize=256
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._try_collect(collector)

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def initialize(self, scale, xavier_init, norm_scale=None):
        if xavier_init:
            nn.init.xavier_normal_(
                self.conv.weight, gain=compute_gain(self.activation))
        else:
            init_weights(self.conv.weight, self.activation, scale=scale)

    def add_reg_dict(self, reg_dict, placement="in_block"):
        if placement == "in_block":
            reg_dict["normal"].append(self.conv.weight)
        elif placement == "before_block":
            reg_dict["input"].append(self.conv.weight)
        else:
            reg_dict["normal"].append(self.conv.weight)
        self.bn.add_reg_dict(reg_dict, placement)
        if not isinstance(self.pre_bias, CustomIdentity):
            self.pre_bias.add_reg_dict(reg_dict, placement)

    def shape_to_text(self):
        out = str()
        out += conv_to_text(self.in_channels, self.out_channels, self.kernel_size)
        out += bn_to_text(self.out_channels)
        return out

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.conv.weight, use_bin)
        out += tensor_to_text(torch.zeros(self.out_channels), use_bin) # fill zero

        bn_mean, bn_std = self.bn.get_merged_params()
        out += tensor_to_text(bn_mean, use_bin)
        out += tensor_to_text(bn_std, use_bin)
        return out

    def forward(self, x, mask):
        x = self.pre_bias(x, mask)
        x = self.conv(x) * mask
        x = self.bn(x, mask)
        x = self.act(x)
        return x

class DepthwiseConvBlock(nn.Module):
    def __init__(self, channels,
                       kernel_size,
                       use_gamma,
                       mode,
                       placement,
                       renorm_clipping,
                       activation,
                       collector=None):
        # Implement it based on "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design
        # in CNNs".

        assert kernel_size >= 5, ""
        assert kernel_size % 2 == 1, ""
        super(DepthwiseConvBlock, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = self.channels
        self.conv = BroadcastDepthwiseConv2d(
            self.channels,
            self.kernel_size,
            padding="same",
            bias=True
        )
        self.rep3x3 = BroadcastDepthwiseConv2d(
            self.channels,
            3,
            padding="same",
            bias=True
        )
        if mode == "fixup" and placement == "in_block":
            self.pre_bias1 = BatchNorm2d(
                num_features=channels,
                use_gamma=False,
                mode=mode
            )
            self.pre_bias2 = BatchNorm2d(
                num_features=channels,
                use_gamma=False,
                mode=mode
            )
        else:
            self.pre_bias1 = CustomIdentity()
            self.pre_bias2 = CustomIdentity()

        self.bn = BatchNorm2d(
            num_features=self.channels,
            use_gamma=use_gamma,
            mode=mode,
            renorm_clipping=renorm_clipping,
            momentum_basic_batchsize=256
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._try_collect(collector)

    def initialize(self, scale, xavier_init, norm_scale=None):
        if xavier_init:
            nn.init.xavier_normal_(
                self.conv.weight, gain=compute_gain(self.activation))
            nn.init.xavier_normal_(
                self.rep3x3.weight, gain=compute_gain(self.activation))
        else:
            init_weights(self.conv.weight, self.activation, scale=scale * 0.8)
            init_weights(self.rep3x3.weight, self.activation, scale=scale * 0.6)

    def add_reg_dict(self, reg_dict, placement="in_block"):
        self.conv.add_reg_dict(reg_dict, placement)
        self.rep3x3.add_reg_dict(reg_dict, placement)
        self.bn.add_reg_dict(reg_dict, placement)
        if not isinstance(self.pre_bias1, CustomIdentity):
            self.pre_bias1.add_reg_dict(reg_dict, placement)
        if not isinstance(self.pre_bias2, CustomIdentity):
            self.pre_bias2.add_reg_dict(reg_dict, placement)

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()

        weights, biases = self.conv.get_merged_params()

        ps = int((self.kernel_size - 3) / 2)
        rep3x3_weights, rep3x3_biases = self.rep3x3.get_merged_params()
        weights += F.pad(rep3x3_weights, (ps, ps, ps, ps), "constant", 0)
        biases += rep3x3_biases

        out += tensor_to_text(weights, use_bin)
        out += tensor_to_text(biases, use_bin)

        bn_mean, bn_std = self.bn.get_merged_params()
        out += tensor_to_text(bn_mean, use_bin)
        out += tensor_to_text(bn_std, use_bin)
        return out

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def shape_to_text(self):
        out = str()
        out += dwconv_to_text(self.channels // self.groups, self.channels, self.kernel_size)
        out += bn_to_text(self.channels)
        return out

    def forward(self, x, mask):
        x1 = self.pre_bias1(x, mask)
        x2 = self.pre_bias1(x, mask)
        out = (self.conv(x1) + self.rep3x3(x2)) * mask
        x = self.bn(x, mask)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(ResidualBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.se_size = kwargs.get("se_size", None)
        self.mode = kwargs.get("mode", "renorm")
        collector = kwargs.get("collector", None)

        self.channels = channels
        self.use_se = self.se_size is not None
        self.conv1 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            use_gamma=False,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            use_gamma=True,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.channels,
                se_size=self.se_size,
                mode=self.mode,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        if xavier_init:
            self.conv1.initialize(scale=1.0, xavier_init=xavier_init)
            self.conv2.initialize(scale=1.0, xavier_init=xavier_init)
            if self.use_se:
                self.se_module.initialize(scale=1.0, xavier_init=xavier_init)
        else:
            if self.use_se:
                self.conv1.initialize(scale=se_fixup_scale, xavier_init=xavier_init)
                self.conv2.initialize(scale=0.0, xavier_init=xavier_init)
                self.se_module.initialize(scale=se_fixup_scale, xavier_init=xavier_init)
            else:
                self.conv1.initialize(scale=fixup_scale, xavier_init=xavier_init)
                self.conv2.initialize(scale=0.0, xavier_init=xavier_init)

    def add_reg_dict(self, reg_dict):
        self.conv1.add_reg_dict(reg_dict)
        self.conv2.add_reg_dict(reg_dict)
        if self.use_se:
            self.se_module.add_reg_dict(reg_dict)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        out = x
        if self.use_se and self.mode == "fixup":
            out = self.se_module(out, mask_buffers)
        out = self.conv1(out, mask)
        out = self.conv2(out, mask)
        if self.use_se and self.mode != "fixup":
            out = self.se_module(out, mask_buffers)
        out = out + x
        out = self.act(out)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(BottleneckBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.se_size = kwargs.get("se_size", None)
        self.mode = kwargs.get("mode", "renorm")
        collector = kwargs.get("collector", None)

        assert self.bottleneck_channels is not None, ""
        self.use_se = self.se_size is not None

        # The inner layers channels.
        self.inner_channels = self.bottleneck_channels

        # The main ResidualBlock channels. We say a 15x192
        # resnet. The 192 is outer_channel.
        self.outer_channels = channels

        self.pre_btl_conv = ConvBlock(
            in_channels=self.outer_channels,
            out_channels=self.inner_channels,
            kernel_size=1,
            use_gamma=False,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.conv1 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.post_btl_conv = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.outer_channels,
            kernel_size=1,
            use_gamma=True,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.outer_channels,
                se_size=self.se_size,
                mode=self.mode,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        if xavier_init:
            self.pre_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
            self.conv1.initialize(scale=1.0, xavier_init=xavier_init)
            self.conv2.initialize(scale=1.0, xavier_init=xavier_init)
            self.post_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
            if self.use_se:
                self.se_module.initialize(scale=1.0, xavier_init=xavier_init)
        else:
            if self.use_se:
                self.pre_btl_conv.initialize(
                    scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.conv1.initialize(
                    scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.conv2.initialize(
                    scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.post_btl_conv.initialize(
                    scale=0.0, xavier_init=xavier_init)
                self.se_module.initialize(scale=se_fixup_scale, xavier_init=xavier_init)
            else:
                self.pre_btl_conv.initialize(
                    scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.conv1.initialize(
                    scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.conv2.initialize(
                    scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.post_btl_conv.initialize(
                    scale=0.0, xavier_init=xavier_init)

    def add_reg_dict(self, reg_dict):
        self.pre_btl_conv.add_reg_dict(reg_dict)
        self.conv1.add_reg_dict(reg_dict)
        self.conv2.add_reg_dict(reg_dict)
        self.post_btl_conv.add_reg_dict(reg_dict)
        if self.use_se:
            self.se_module.add_reg_dict(reg_dict)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        out = x
        if self.use_se and self.mode == "fixup":
            out = self.se_module(out, mask_buffers)
        out = self.pre_btl_conv(out, mask)
        out = self.conv1(out, mask)
        out = self.conv2(out, mask)
        out = self.post_btl_conv(out, mask)
        if self.use_se and self.mode != "fixup":
            out = self.se_module(out, mask_buffers)
        out = out + x
        out = self.act(out)
        return out

class NestedBottleneckBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(NestedBottleneckBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.se_size = kwargs.get("se_size", None)
        self.mode = kwargs.get("mode", "renorm")
        collector = kwargs.get("collector", None)

        assert self.bottleneck_channels is not None, ""
        self.use_se = self.se_size is not None

        # The inner layers channels.
        self.inner_channels = self.bottleneck_channels

        # The main ResidualBlock channels. We say a 15x192
        # resnet. The 192 is outer_channel.
        self.outer_channels = channels

        self.pre_btl_conv = ConvBlock(
            in_channels=self.outer_channels,
            out_channels=self.inner_channels,
            kernel_size=1,
            use_gamma=False,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.block1 = ResidualBlock(
            channels=self.inner_channels,
            mode=self.mode,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.block2 = ResidualBlock(
            channels=self.inner_channels,
            mode=self.mode,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.post_btl_conv = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.outer_channels,
            kernel_size=1,
            use_gamma=True,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.outer_channels,
                se_size=self.se_size,
                mode=self.mode,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        if xavier_init:
            self.pre_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
            self.block1.initialize(scale=1.0, xavier_init=xavier_init)
            self.block2.initialize(scale=1.0, xavier_init=xavier_init)
            self.post_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
            if self.use_se:
                self.se_module.initialize(scale=1.0, xavier_init=xavier_init)
        else:
            if self.use_se:
                self.pre_btl_conv.initialize(
                    scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.block1.initialize(
                    fixup_scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.block2.initialize(
                    fixup_scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.post_btl_conv.initialize(scale=0.0, xavier_init=xavier_init)
                self.se_module.initialize(scale=se_fixup_scale, xavier_init=xavier_init)
            else:
                self.pre_btl_conv.initialize(
                    scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.block1.initialize(
                    fixup_scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.block2.initialize(
                    fixup_scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.post_btl_conv.initialize(scale=0.0, xavier_init=xavier_init)

    def add_reg_dict(self, reg_dict):
        self.pre_btl_conv.add_reg_dict(reg_dict)
        self.block1.add_reg_dict(reg_dict)
        self.block2.add_reg_dict(reg_dict)
        self.post_btl_conv.add_reg_dict(reg_dict)
        if self.use_se:
            self.se_module.add_reg_dict(reg_dict)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        out = x
        if self.use_se and self.mode == "fixup":
            out = self.se_module(out, mask_buffers)
        out = self.pre_btl_conv(out, mask)
        out = self.block1(out, mask_buffers)
        out = self.block2(out, mask_buffers)
        out = self.post_btl_conv(out, mask)
        if self.use_se and self.mode != "fixup":
            out = self.se_module(out, mask_buffers)
        out = out + x
        out = self.act(out)
        return out

class MixerBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(MixerBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.se_size = kwargs.get("se_size", None)
        self.kernel_size = kwargs.get("kernel_size", 7)
        self.ffn_expansion_ratio = kwargs.get("ffn_expansion_ratio", 1.5)
        self.version = kwargs.get("version", 1)
        self.mode = kwargs.get("mode", "renorm")
        collector = kwargs.get("collector", None)

        self.channels = channels
        self.use_se = self.se_size is not None
        assert self.version in [1, 2], ""

        self.depthwise_conv = DepthwiseConvBlock(
            channels=self.channels,
            kernel_size=self.kernel_size,
            use_gamma=True,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )

        self.ffn_channels = int(self.ffn_expansion_ratio * self.channels)
        self.ffn1 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.ffn_channels,
            kernel_size=1,
            use_gamma=False,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.ffn2 = ConvBlock(
            in_channels=self.ffn_channels,
            out_channels=self.channels,
            kernel_size=1,
            use_gamma=True,
            mode=self.mode,
            placement="in_block",
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.channels,
                se_size=self.se_size,
                mode=self.mode,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        if xavier_init:
            self.depthwise_conv.initialize(scale=1.0, xavier_init=xavier_init)
            self.ffn1.initialize(scale=1.0, xavier_init=xavier_init)
            self.ffn2.initialize(scale=1.0, xavier_init=xavier_init)
            if self.use_se:
                self.se_module.initialize(scale=1.0, xavier_init=xavier_init)
        else:
            if self.use_se:
                self.depthwise_conv.initialize(
                    scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.ffn1.initialize(
                    scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.ffn2.initialize(
                    scale=0.0, xavier_init=xavier_init)
                self.se_module.initialize(scale=se_fixup_scale, xavier_init=xavier_init)
            else:
                self.depthwise_conv.initialize(
                    scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.ffn1.initialize(
                    scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.ffn2.initialize(
                    scale=0.0, xavier_init=xavier_init)

    def add_reg_dict(self, reg_dict):
        self.depthwise_conv.add_reg_dict(reg_dict)
        self.ffn1.add_reg_dict(reg_dict)
        self.ffn2.add_reg_dict(reg_dict)
        if self.use_se:
            self.se_module.add_reg_dict(reg_dict)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        if self.version == 1:
            out = x
            if self.use_se and self.mode == "fixup":
                out = self.se_module(out, mask_buffers)
            x = self.depthwise_conv(out, mask) + x
            out = x
            out = self.ffn1(out, mask)
            out = self.ffn2(out, mask)
            if self.use_se and self.mode != "fixup":
                out = self.se_module(out, mask_buffers)
            out = out + x
            out = self.act(out)
        elif self.version == 2:
            out = x
            if self.use_se and self.mode == "fixup":
                out = self.se_module(out, mask_buffers)
            out = self.depthwise_conv(out, mask)
            out = self.ffn1(out, mask)
            out = self.ffn2(out, mask)
            if self.use_se and self.mode != "fixup":
                out = self.se_module(out, mask_buffers)
            out = out + x
            out = self.act(out)
        return out

# Simplified functional replacement for better ONNX export
class CustomRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # Explicit implementation: x * w / sqrt(mean(x^2) + eps)
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps).to(x.dtype)) * self.weight

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        RMSNormの初期化
        Args:
            dim: 特徴量の次元数 (hidden_size)
            eps: ゼロ除算を防ぐための小さな値
        """
        super().__init__()
        self.eps = eps
        # 学習可能なスケーリングパラメータ (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMSの計算と正規化"""
        # x^2 の平均をとって平方根を計算
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass
        x: (batch, seq_len, dim)
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cos_sin_2d(dim, pos_len, theta=100.0):
    """Precompute cos and sin tables of 2D frequencies for RoPE (real-valued, interleaved layout).
    Returns shape: (pos_len * pos_len, dim)
    """
    assert dim % 4 == 0
    dim_half = dim // 2

    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half))

    t = torch.arange(pos_len, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(t, t, indexing='ij')

    emb_h = grid_h.unsqueeze(-1) * freqs
    emb_w = grid_w.unsqueeze(-1) * freqs

    emb = torch.cat([emb_h, emb_w], dim=-1)
    emb = emb.flatten(0, 1)
    emb = emb.repeat_interleave(2, dim=-1)

    return emb.cos(), emb.sin()

def apply_rotary_emb(xq, xk, cos, sin):
    """Apply rotary position embeddings to Q and K tensors.
    xq, xk: (Batch, Seq, Heads, Dim)
    cos, sin: (Seq, Dim)
    """
    def rotate_every_two(x):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x.unbind(dim=-1)
        x_rotated = torch.stack([-x1, x0], dim=-1)
        return x_rotated.flatten(-2)

    cos = cos.view(1, xq.shape[1], 1, xq.shape[-1])
    sin = sin.view(1, xq.shape[1], 1, xq.shape[-1])

    xq_out = xq * cos + rotate_every_two(xq) * sin
    xk_out = xk * cos + rotate_every_two(xk) * sin

    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_learnable_rotary_emb(xq, xk, cos_q, sin_q, cos_k, sin_k):
    """Apply learnable rotary position embeddings to Q and K tensors.
    xq: (Batch, Seq, num_heads, Dim)
    xk: (Batch, Seq, num_kv_heads, Dim)
    cos_q, sin_q: (Seq, num_heads, Dim/2) - per-head, per-pair
    cos_k, sin_k: (Seq, num_kv_heads, Dim/2) - per-kv-head, per-pair
    """
    def _rotate(x, cos, sin):
        B, S, H, D = x.shape
        P = D // 2
        x_pairs = x.view(B, S, H, P, 2)
        x0, x1 = x_pairs.unbind(dim=-1)  # each (B, S, H, P)
        cos = cos.unsqueeze(0)  # (1, S, H, P)
        sin = sin.unsqueeze(0)
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.reshape(B, S, H, D).type_as(x)

    return _rotate(xq, cos_q, sin_q), _rotate(xk, cos_k, sin_k)

class TransformerAttentionBlock(nn.Module):
    """Self-attention half of a transformer block, with its own residual connection.

    Contains: RMSNorm -> Q/K/V projections -> (optional RoPE) -> attention -> output projection.
    Returns residual only; caller is responsible for adding to trunk.
    """
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(TransformerAttentionBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        # self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        # self.se_size = kwargs.get("se_size", None)
        # self.mode = kwargs.get("mode", "renorm")
        collector = kwargs.get("collector", None)
        self.pos_len = kwargs.get("pos_len", 9)
        # self.ffn_dim = kwargs.get("transformer_ffn_channels", channels * 2)
        self.use_rope = kwargs.get("use_rope", True)
        self.learnable_rope = kwargs.get("learnable_rope", False) if self.use_rope else False
        self.rope_theta = kwargs.get("rope_theta", 100.0)
        self.ffn_dim = kwargs.get("transformer_ffn_channels", channels * 2)
        self.use_swiglu = kwargs.get("use_swiglu", True)
        self.num_heads = kwargs.get("transformer_heads", 4)
        self.num_kv_heads = kwargs.get("transformer_kv_heads", self.num_heads)
        self.use_qk_norm = kwargs.get("attention_qk_norm", True)
        # Compute how many query heads each KV head serves (group size)
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = channels // self.num_heads
        self.q_head_dim = kwargs.get("attention_query_head_dim", channels // self.num_heads) # 96 // 6
        self.v_head_dim = kwargs.get("attention_value_head_dim", channels // self.num_heads) # 96 // 6

        if self.use_rope:
            # assert self.q_head_dim * self.num_heads == channels, "Embed dim mismatch" # 16 * 6 == 96
            assert self.q_head_dim % 4 == 0, "Head dim mismatch for 2D RoPE" # 16 % 4 == 0
        # assert self.v_head_dim * self.num_heads == channels, "Embed dim mismatch" # 16 * 6 == 96
        # assert self.v_head_dim % 4 == 0, "Head dim mismatch for 2D RoPE" # 16 % 4 == 0
        assert self.num_heads % self.num_kv_heads == 0, \
            f"Query heads ({self.num_heads}) must be divisible by KV heads ({self.num_kv_heads})"

        # Keep full-sized Q projection
        # self.q_proj = torch.nn.Linear(channels, self.num_heads * self.q_head_dim, bias=False)
        self.q_proj = torch.nn.Linear(channels, channels, bias=False)
        # Reduce K/V projection dimensions (GQA)
        # self.k_proj = torch.nn.Linear(channels, self.num_kv_heads * self.q_head_dim, bias=False)
        # self.v_proj = torch.nn.Linear(channels, self.num_kv_heads * self.v_head_dim, bias=False)
        self.k_proj = torch.nn.Linear(channels, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(channels, self.num_kv_heads * self.head_dim, bias=False)
        # self.out_proj = torch.nn.Linear(self.num_heads * self.v_head_dim, channels, bias=False)
        self.out_proj = torch.nn.Linear(channels, channels, bias=False)

        # Cache cos and sin
        # Assume precompute_freqs_cos_sin_2d_fixed is defined externally
        if self.use_rope:
            # self.rope_theta = kwargs.get("rope_theta", 100.0) # KV Heads (Key/Value)
            assert self.rope_theta > self.pos_len * 2.0, f"theta={self.rope_theta} of RoPE may be too small for pos_len={self.pos_len}"
            cos_cached, sin_cached = precompute_freqs_cos_sin_2d(self.head_dim, self.pos_len, self.rope_theta)
            self.register_buffer("cos_cached", cos_cached, persistent=False)
            self.register_buffer("sin_cached", sin_cached, persistent=False)
        else:
            self.cos_cached = None
            self.sin_cached = None

        self.ffn_linear1 = torch.nn.Linear(channels, self.ffn_dim, bias=False)
        if self.use_swiglu:
            self.ffn_linear_gate = torch.nn.Linear(channels, self.ffn_dim, bias=False)
            self.ffn_act = torch.nn.SiLU(inplace=False)  # Only QAT-int8 training requires inplace=False, but for normal training it will not be harmful after compilation
        else:
            self.ffn_act = act(self.activation, inplace=False) # Only QAT-int8 training requires inplace=False, but for normal training it will not be harmful after compilation
            
        self.ffn_linear2 = torch.nn.Linear(self.ffn_dim, channels, bias=False)
        
        # self.norm1 = torch.nn.RMSNorm(channels, eps=1e-6)
        # self.norm2 = torch.nn.RMSNorm(channels, eps=1e-6)
        self.norm1 = CustomRMSNorm(channels, eps=1e-6)
        self.norm2 = CustomRMSNorm(channels, eps=1e-6)
        """
        # QK-norm: RMSNorm on Q and K per-head before the attention dot product.
        # See ViT-22B, etc.
        if self.use_qk_norm:
            # self.q_norm = torch.nn.RMSNorm(self.q_head_dim, eps=1e-6)
            # self.k_norm = torch.nn.RMSNorm(self.q_head_dim, eps=1e-6)
            self.q_norm = RMSNorm(self.q_head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.q_head_dim, eps=1e-6)

        # Inline registers: registers are part of the trunk tensor (NC1S layout),
        # share trunk channel dim, and participate in all layers identically to
        # board tokens. No separate parameters needed.
        self.inline_registers = kwargs.get("inline_registers", False)
        self.num_rw_registers = kwargs.get("attention_num_rw_registers", 0)
        if self.num_rw_registers > 0 and self.inline_registers:
            assert not (self.use_gab or self.use_tab), \
                "Inline register tokens are not currently supported together with GAB/TAB"
            assert self.use_rope and config.get("learnable_rope", False), \
                "Inline register tokens require learnable RoPE"


        if self.use_rope:
            if self.learnable_rope:
                assert self.q_head_dim % 2 == 0, f"Head dim must be even for learnable RoPE, got {self.q_head_dim}"
                num_pairs = self.q_head_dim // 2
                # Learnable 2D RoPE frequencies.
                # Geometric initialization from 1 rad/square to 1/50 rad/square
                log_lo = math.log(1.0 / 50.0)
                log_hi = math.log(1.0)
                init_freqs = (
                    torch.exp(torch.empty(self.num_kv_heads, num_pairs, 2).uniform_(log_lo, log_hi))
                    * (torch.randint(0, 2, (self.num_kv_heads, num_pairs, 2)) * 2 - 1).float()
                )
                self.rope_freqs = torch.nn.Parameter(init_freqs)  # (num_kv_heads, P, 2)
                # self.pos_len = pos_len
                self.cos_cached = None
                self.sin_cached = None
            else:
                assert self.rope_theta > self.pos_len * 2.0, f"theta={self.rope_theta} of RoPE may be too small for pos_len={self.pos_len}"
                cos_cached, sin_cached = precompute_freqs_cos_sin_2d(self.q_head_dim, self.pos_len, self.rope_theta)
                self.register_buffer("cos_cached", cos_cached, persistent=False)
                self.register_buffer("sin_cached", sin_cached, persistent=False)
        else:
            self.cos_cached = None
            self.sin_cached = None

        # self.norm1 = torch.nn.RMSNorm(channels, eps=1e-6)
        self.norm1 = RMSNorm(channels, eps=1e-6)
        """

    def add_reg_dict(self, reg_dict):
        for name, param in self.named_parameters():
            if "norm" in name or "cached" in name:
                reg_dict["noreg"].append(param)
                continue
            if "weight" in name:
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                    reg_dict["normal_attn"].append(param)
                else:
                    reg_dict["normal"].append(param)
            else:
                reg_dict["noreg"].append(param)

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        # Relies on torch initialization, nothing to do here.
        # Since we have active normalization layers, initial scaling doesn't matter so much.
        pass

    # def forward(self, x, mask, mask_sum_hw, mask_sum:float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
    def forward(self, x, mask_buffers):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        # mask, _, _ = mask_buffers
        # mask_sum_hw = torch.sum(mask,dim=(2, 3), keepdim=True)
        # mask_sum = torch.sum(mask)
        batch_size, channels, height, width = x.shape
        # mask = x[:, 0:1, :, :].contiguous()
        # mask_sum_hw = torch.sum(mask,dim=(2,3), keepdim=True)
        # mask_sum = torch.sum(mask)
        mask = None
        # print(f'batch_size: {batch_size}') # 256 + 8
        # print(f'channels: {channels}') # 96
        # print(f'height: {height}') # 9
        # print(f'width: {width}') # 9
        seq_len = height * width
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1)

        x_norm = self.norm1(x_in)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        # k = k.view(batch_size, seq_len, self.num_kv_heads, self.q_head_dim)
        # v = v.view(batch_size, seq_len, self.num_kv_heads, self.v_head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.use_rope:
            q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)
            """
            if self.learnable_rope:
                # When inline registers are active, use precomputed all_pos_x/all_pos_y
                # which covers both board and register positions. Otherwise compute from arange.
                if self.inline_registers and self.num_rw_registers > 0:
                    reg_state = block_shared_data[REGISTER_STATE]
                    s_x = reg_state.all_pos_x  # (B, S)
                    s_y = reg_state.all_pos_y  # (B, S)
                else:
                    s_idx = torch.arange(seq_len, device=q.device)
                    s_y = (s_idx // self.pos_len).float()  # row
                    s_x = (s_idx % self.pos_len).float()   # col
                cos_k, sin_k = compute_learnable_rope_cos_sin(s_x, s_y, self.rope_freqs)  # ([B,] S, H_kv, P)
                # For Q: expand kv head freqs to match num_heads if using multi-query attention
                if self.n_rep > 1:
                    cos_q = cos_k.unsqueeze(-3).expand(*cos_k.shape[:-2], self.n_rep, cos_k.shape[-1]).reshape(*cos_k.shape[:-2], self.num_heads, -1)
                    sin_q = sin_k.unsqueeze(-3).expand(*sin_k.shape[:-2], self.n_rep, sin_k.shape[-1]).reshape(*sin_k.shape[:-2], self.num_heads, -1)
                else:
                    cos_q = cos_k
                    sin_q = sin_k
                q, k = apply_learnable_rotary_emb(q, k, cos_q, sin_q, cos_k, sin_k)
            else:
                q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)
                # Compute per-head, per-pair angles from learnable 2D frequencies.
                # rope_freqs: (num_kv_heads, P, 2) = (H_kv, P, [omega_x, omega_y])
            #     s_idx = torch.arange(seq_len, device=q.device)
            #     s_y = (s_idx // self.pos_len).float()  # row
            #     s_x = (s_idx % self.pos_len).float()   # col
                # angles: (S, H_kv, P) = omega_x * x + omega_y * y
            #     angles = s_x.view(-1, 1, 1) * self.rope_freqs[:, :, 0] + s_y.view(-1, 1, 1) * self.rope_freqs[:, :, 1]
            #     cos_k = torch.cos(angles)  # (S, H_kv, P)
            #     sin_k = torch.sin(angles)
                # For Q: expand kv head freqs to match num_heads if using multi-query attention
            #     if self.n_rep > 1:
            #         cos_q = cos_k.unsqueeze(2).expand(-1, -1, self.n_rep, -1).reshape(seq_len, self.num_heads, -1)
            #         sin_q = sin_k.unsqueeze(2).expand(-1, -1, self.n_rep, -1).reshape(seq_len, self.num_heads, -1)
            #     else:
            #         cos_q = cos_k
            #         sin_q = sin_k
            #     q, k = apply_learnable_rotary_emb(q, k, cos_q, sin_q, cos_k, sin_k)
            # else:
            #     q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)
            """

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.n_rep > 1:
            # k = k.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.q_head_dim)
            # k = k.reshape(batch_size, self.num_heads, seq_len, self.q_head_dim)
            # v = v.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.v_head_dim)
            # v = v.reshape(batch_size, self.num_heads, seq_len, self.v_head_dim)
            k = k.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # if self.use_qk_norm:
        #     q = self.q_norm(q)
        #     k = self.k_norm(k)

        if mask is not None:
            # For inline registers, mask is N11S and already includes register positions
            # (always 1.0), so seq_len already covers them.
            mask_flat = mask.view(batch_size, 1, 1, seq_len)
            attn_mask = torch.zeros_like(mask_flat, dtype=q.dtype)
            attn_mask.masked_fill_(mask_flat == 0, float('-inf'))
        else:
            attn_mask = None

        # Default scaling for q/k dot product, 1/sqrt(query head dim)
        # scale = 1.0 / math.sqrt(self.q_head_dim)

        # If attention weights are requested, force the manual path so we can capture them.
        # wants_attn_weights = (
        #     extra_outputs is not None
        #     and self.name+".attn_weights" in extra_outputs.requested
        # )

        # if not wants_attn_weights:
        #     attn_output = torch.nn.functional.scaled_dot_product_attention(
        #         q, k, v,
        #         attn_mask=attn_mask,
        #         dropout_p=0.0,
        #         scale=scale,
        #     )
        # else:
            # Manual attention path to capture weights.
            # logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            # scale=scale,
        )

        # if attn_mask is not None:
        #     logits = logits + attn_mask

        # attn_weights = torch.softmax(logits, dim=-1)

        #     if extra_outputs is not None:
        #         extra_outputs.report(self.name+".attn_weights", attn_weights)

        # attn_output = torch.matmul(attn_weights, v)  # (B, H, S, Dv)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        # attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        attn_output = attn_output.view(batch_size, seq_len, channels)
        attn_output = self.out_proj(attn_output)
        x = x_in + attn_output
        xn = self.norm2(x)
        
        if self.use_swiglu:
            # SwiGLU: (Act(Linear1(x)) * LinearGate(x)) -> Linear2
            x1 = self.ffn_linear1(xn)
            x1 = self.ffn_act(x1)
            x_gate = self.ffn_linear_gate(xn)
            x1 = x1 * x_gate
        else:
            # Standard: Act(Linear1(x)) -> Linear2
            x1 = self.ffn_linear1(xn)
            x1 = self.ffn_act(x1)
        x1 = self.ffn_linear2(x1)
        x = x + x1
        
        x = x.permute(0, 2, 1).view(batch_size, channels, height, width)
        return x

        # result = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        # if extra_outputs is not None:
        #     extra_outputs.report(self.name+".out", result)
        # return result

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()

        self.layers_collector = list()

        self.nntype = cfg.nntype

        self.activation = cfg.activation.lower()
        self.input_channels = cfg.input_channels
        self.residual_channels = cfg.residual_channels
        self.xsize = cfg.boardsize
        self.ysize = cfg.boardsize
        self.policy_head_channels = cfg.policy_head_channels
        self.value_head_channels = cfg.value_head_channels
        self.se_ratio = cfg.se_ratio
        self.policy_head_type = cfg.policy_head_type
        if type(self.policy_head_type) == str:
            self.policy_head_type = { "Type" : self.policy_head_type }
        self.renorm_clipping = {"rmax" : cfg.renorm_max_r, "dmax" : cfg.renorm_max_d}
        self.value_misc = 15
        self.policy_outs = 5
        self.stack = cfg.stack
        self.version = 5
        self.mode = cfg.mode.lower()
        if self.mode == "fixup":
            self.xavier_init = False
        else:
            self.xavier_init = True

        self.construct_layers()

        num_total_blocks = len(self.residual_tower)
        with torch.no_grad():
            self.input_conv.initialize(scale=1.0, xavier_init=self.xavier_init)
            if self.mode == "fixup":
                fixup_scale = 1.0 / math.sqrt(num_total_blocks)
                se_fixup_scale = math.pow(num_total_blocks, -1.0 / (2 * 4 - 2))
                for block in self.residual_tower:
                    block.initialize(fixup_scale=fixup_scale,
                        se_fixup_scale=se_fixup_scale, xavier_init=self.xavier_init)
            else:
                fixup_scale = 1.0
                for block in self.residual_tower:
                    block.initialize(fixup_scale=fixup_scale,
                        se_fixup_scale=fixup_scale, xavier_init=self.xavier_init)

            self.policy_conv.initialize(scale=0.8, xavier_init=self.xavier_init)
            if self.policy_head_type["Type"] == "RepLK":
                self.policy_depthwise_conv.initialize(scale=1.0, xavier_init=self.xavier_init)
                self.policy_pointwise_conv.initialize(scale=1.0, xavier_init=self.xavier_init)
            self.policy_intermediate_fc.initialize(scale=0.6, xavier_init=self.xavier_init)
            self.pol_misc.initialize(scale=0.3, xavier_init=self.xavier_init)
            self.pol_misc_pass_fc.initialize(scale=0.3, xavier_init=self.xavier_init)
            self.value_conv.initialize(scale=1.0, xavier_init=self.xavier_init)
            self.value_intermediate_fc.initialize(scale=1.0, xavier_init=self.xavier_init)
            self.ownership_conv.initialize(scale=0.2, xavier_init=self.xavier_init)
            self.value_misc_fc.initialize(scale=0.2, xavier_init=self.xavier_init)

    def create_policy_head(self):
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_head_channels,
            kernel_size=1,
            use_gamma=False,
            mode=self.mode,
            placement="after_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=self.layers_collector
        )
        if self.policy_head_type["Type"] == "Normal":
            pass
        elif self.policy_head_type["Type"] == "RepLK":
            dw_kernel_size = max(self.policy_head_type.get("KernelSize", 7), 7)
            self.policy_depthwise_conv = DepthwiseConvBlock(
                channels=self.policy_head_channels,
                kernel_size=dw_kernel_size,
                use_gamma=False,
                mode=self.mode,
                placement="after_block",
                renorm_clipping=self.renorm_clipping,
                activation=self.activation,
                collector=self.layers_collector
            )
            self.policy_pointwise_conv = ConvBlock(
                in_channels=self.policy_head_channels,
                out_channels=self.policy_head_channels,
                kernel_size=1,
                use_gamma=True,
                mode=self.mode,
                placement="after_block",
                renorm_clipping=self.renorm_clipping,
                activation=self.activation,
                collector=self.layers_collector
            )
        else:
            raise Exception("Invalid policy head type.")

        self.policy_intermediate_fc = FullyConnect(
            in_size=self.policy_head_channels * 3,
            out_size=self.policy_head_channels,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.pol_misc = Convolve(
            in_channels=self.policy_head_channels,
            out_channels=self.policy_outs,
            kernel_size=1,
            activation="identity",
            collector=self.layers_collector
        )
        self.pol_misc_pass_fc = FullyConnect(
            in_size=self.policy_head_channels,
            out_size=self.policy_outs,
            activation="identity",
            collector=self.layers_collector
        )

    def create_value_head(self):
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_head_channels,
            kernel_size=1,
            use_gamma=False,
            mode=self.mode,
            placement="after_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.value_intermediate_fc = FullyConnect(
            in_size=self.value_head_channels * 3,
            out_size=self.value_head_channels * 3,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.ownership_conv = Convolve(
            in_channels=self.value_head_channels,
            out_channels=1,
            kernel_size=1,
            activation="identity",
            collector=self.layers_collector
        )
        self.value_misc_fc = FullyConnect(
            in_size=self.value_head_channels * 3,
            out_size=self.value_misc,
            activation="identity",
            collector=self.layers_collector
        )

    def parse_blocksetting(self, blocksetting, blockargs):
        components = list()
        if type(blocksetting) == str:
            components = blocksetting.strip().split('-')
            setting_args = dict()
        else:
            components = blocksetting["Block"].strip().split('-')
            setting_args = blocksetting["Args"]

        block = None
        blockname = None
        channels = self.residual_channels
        for component in components:
            if component == "ResidualBlock":
                block = ResidualBlock
            elif component == "BottleneckBlock":
                blockargs["bottleneck_channels"] = channels // 2
                assert channels % 2 == 0, ""
                block = BottleneckBlock
            elif component == "NestedBottleneckBlock":
                blockargs["bottleneck_channels"] = channels // 2
                assert channels % 2 == 0, ""
                block = NestedBottleneckBlock
            elif component in ["MixerBlock", "MixerBlockV1"]:
                block = MixerBlock
            elif component == "MixerBlockV2":
                block = MixerBlock
                blockargs["version"] = 2
            elif component == "TransformerAttentionBlock":
                block = TransformerAttentionBlock
            elif component == "SE":
                blockargs["se_size"] = channels // self.se_ratio
                assert channels % self.se_ratio == 0, ""
            else:
                raise Exception("Invalid block structure.")

        if block is None:
            raise Exception("There is no basic block.")

        # overwrite default settings
        for key, value in setting_args.items():
            if key == "BottleneckChannels" :
                blockargs["bottleneck_channels"] = value
            elif key == "SeRatio" :
                blockargs["se_size"] = channels // value
                assert channels % self.se_ratio == 0, ""
            elif key == "KernelSize":
                blockargs["kernel_size"] = value
            elif key == "FfnExpansionRatio":
                blockargs["ffn_expansion_ratio"] = value
            else:
                raise Exception("Invalid block setting.")
        return block, channels, blockargs

    def create_residual_tower(self):
        self.residual_tower = nn.ModuleList()

        for blocksetting in self.stack:
            blockargs = {
                "se_size" : None,
                "bottleneck_channels" : None,
                "version" : 1,
                "activation" : self.activation,
                "renorm_clipping" : self.renorm_clipping,
                "mode" : self.mode,
                "pos_len" : self.xsize,
                "collector" : self.layers_collector
            }
            block, channels, blockargs = self.parse_blocksetting(blocksetting, blockargs)
            self.residual_tower.append(block(channels=channels, **blockargs))

    def construct_layers(self):
        self.global_pool = GlobalPool(is_value_head=False)
        self.global_pool_val = GlobalPool(is_value_head=True)

        self.input_conv = ConvBlock(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=3,
            use_gamma=False if self.mode == "fixup" else True,
            mode=self.mode,
            placement="before_block",
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.create_residual_tower()
        self.create_policy_head()
        self.create_value_head()

    def forward(self, planes, *args, **kwargs):
        target = kwargs.get("target", None)
        use_symm = kwargs.get("use_symm", False)
        loss_weight_dict = kwargs.get("loss_weight_dict", None)

        symm = int(np.random.choice(8, 1)[0])
        if use_symm:
            planes = torch_symmetry(symm, planes, invert=False)

        # mask buffers
        mask = planes[:, (self.input_channels-1):self.input_channels , :, :]
        mask_sum_hw = torch.sum(mask, dim=(1,2,3))
        mask_sum_hw_sqrt = torch.sqrt(mask_sum_hw)
        mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)

        # input layer
        x = self.input_conv(planes, mask)

        # residual tower
        for block in self.residual_tower:
            x = block(x, mask_buffers)

        # policy head
        pol = self.policy_conv(x, mask)
        if self.policy_head_type["Type"] == "RepLK":
            pol = self.policy_depthwise_conv(pol, mask)
            pol = self.policy_pointwise_conv(pol, mask)
        pol_gpool = self.global_pool(pol, mask_buffers)
        pol_inter = self.policy_intermediate_fc(pol_gpool)

        # Add intermediate as biases. It may improve the policy performance.
        b, c = pol_inter.shape
        pol = (pol + pol_inter.view(b, c, 1, 1)) * mask

        # Apply CRAZY_NEGATIVE_VALUE on out of board area. This position
        # policy will be zero after softmax 
        output_prob = self.pol_misc(pol, mask) + (1.0-mask) * CRAZY_NEGATIVE_VALUE

        if use_symm:
            output_prob = torch_symmetry(symm, output_prob, invert=True)
        output_prob = torch.flatten(output_prob, start_dim=2, end_dim=3) # b, c, h*w
        output_prob_pass = self.pol_misc_pass_fc(pol_inter)  # b, c

        # value head
        val = self.value_conv(x, mask)
        val_gpool = self.global_pool_val(val, mask_buffers)
        val_inter = self.value_intermediate_fc(val_gpool)

        output_ownership = self.ownership_conv(val, mask)
        if use_symm:
            output_ownership = torch_symmetry(symm, output_ownership, invert=True)
        output_ownership = torch.flatten(output_ownership, start_dim=1, end_dim=3)
        output_ownership = torch.tanh(output_ownership)

        output_val = self.value_misc_fc(val_inter)
        if target is None:
            predict = (
                output_prob,
                output_prob_pass,
                output_val,
                output_ownership
            )
            return predict, None

        b, c = output_prob_pass.shape
        pol_misc = torch.cat((output_prob, output_prob_pass.view(b, c, 1)), dim=2)

        prob, aux_prob, soft_prob, soft_aux_prob, optimistic_prob = torch.split(pol_misc, [1, 1, 1, 1, 1], dim=1)
        prob            = torch.flatten(prob, start_dim=1, end_dim=2)
        aux_prob        = torch.flatten(aux_prob, start_dim=1, end_dim=2)
        soft_prob       = torch.flatten(soft_prob, start_dim=1, end_dim=2)
        soft_aux_prob   = torch.flatten(soft_aux_prob, start_dim=1, end_dim=2)
        optimistic_prob = torch.flatten(optimistic_prob, start_dim=1, end_dim=2)

        wdl, all_q_vals, all_scores, all_errors = torch.split(output_val, [3, 5, 5, 2], dim=1)
        all_q_vals = torch.tanh(all_q_vals)
        all_errors = SoftPlusWithGradientFloorFunction.apply(all_errors, 0.05, True)

        short_term_q_error, short_term_score_error = torch.split(all_errors, [1, 1], dim=1)
        all_scores = 20 * all_scores
        short_term_q_error = 0.25 * short_term_q_error
        short_term_score_error = 150 * short_term_score_error
        all_errors = torch.cat((short_term_q_error, short_term_score_error), dim=1)

        predict = (
            prob, # logits
            aux_prob, # logits
            soft_prob, # logits
            soft_aux_prob, # logits
            optimistic_prob, # logits
            output_ownership,
            wdl, # logits
            all_q_vals, # {final, current, short, middle, long}
            all_scores, # {final, current, short, middle, long}
            all_errors # {q error, score error}
        )
        if use_symm:
            mask = torch_symmetry(symm, mask, invert=True)
            mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)

        all_loss_dict = dict()
        if target is not None:
            all_loss_dict = self.compute_loss(predict, target, mask_buffers, loss_weight_dict)

        return predict, all_loss_dict

    def compute_loss(self, pred, target, mask_buffers, loss_weight_dict):
        mask, mask_sum_hw, _ = mask_buffers
        policy_mask = torch.flatten(mask, start_dim=1, end_dim=3)
        b, _ = policy_mask.shape
        policy_mask = torch.cat((policy_mask, mask.new_ones((b, 1))), dim=1)

        if loss_weight_dict is None:
            soft_weight = 0.1
        else:
            soft_weight = loss_weight_dict["soft"]

        p_prob, p_aux_prob, p_soft_prob, p_soft_aux_prob, p_optimistic_prob, p_ownership, p_wdl, p_q_vals, p_scores, p_errors = pred
        t_prob, t_aux_prob, t_ownership, t_wdl, t_q_vals, t_scores, global_weight = target

        def make_soft_porb(prob, policy_mask, eps=1e-7, t=4):
            soft_prob = (prob + eps) * policy_mask
            soft_prob = torch.pow(soft_prob, 1/t)
            soft_prob /= torch.sum(soft_prob, dim=1, keepdim=True)
            return soft_prob

        def cross_entropy(pred, target, weight=1.):
            loss_sum = -torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1)
            return torch.mean(weight * loss_sum, dim=0)

        def huber_loss(x, y, delta, weight=1.):
            absdiff = torch.abs(x - y)
            loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
            loss_sum = torch.sum(loss, dim=1)
            return torch.mean(weight * loss_sum, dim=0)

        def mse_loss(pred, target, weight=1.):
            loss_sum = torch.mean(torch.square(pred - target), dim=1)
            return torch.mean(weight * loss_sum, dim=0)

        def mse_loss_spat(pred, target, weight=1.):
            loss_sum = torch.sum(torch.square(pred - target), dim=1) / mask_sum_hw
            return torch.mean(weight * loss_sum, dim=0)

        def square_huber_loss(pred, x, y, delta, eps, weight=1.):
            sqerror = torch.square(x - y) + eps
            loss = huber_loss(pred, sqerror, delta=delta, weight=weight)
            return loss

        # will use these values later
        _, short_term_q_pred, _ = torch.split(p_q_vals, [2, 1, 2], dim=1)
        _, short_term_q_target, _ = torch.split(t_q_vals, [2, 1, 2], dim=1)
        _, short_term_score_pred, _ = torch.split(p_scores, [2, 1, 2], dim=1)
        _, short_term_score_target, _ = torch.split(t_scores, [2, 1, 2], dim=1)
        short_term_q_error, short_term_score_error = torch.split(p_errors, [1, 1], dim=1)

        # current player's probabilities loss
        prob_loss = 1. * cross_entropy(p_prob, t_prob, global_weight)

        # opponent's probabilities loss
        aux_prob_loss = 0.15 * cross_entropy(p_aux_prob, t_aux_prob, global_weight)

        # current player's soft probabilities loss
        soft_prob_loss = 1. * soft_weight * cross_entropy(p_soft_prob, make_soft_porb(t_prob, policy_mask), global_weight)

        # opponent's soft probabilities loss
        soft_aux_prob_loss = 0.15 * soft_weight * cross_entropy(p_soft_aux_prob, make_soft_porb(t_aux_prob, policy_mask), global_weight)

        # short-term optimistic probabilities loss
        z_short_term_q = (short_term_q_target - short_term_q_pred.detach()) / torch.sqrt(short_term_q_error.detach() + 0.0001)
        z_short_term_score = (short_term_score_target - short_term_score_pred.detach()) / torch.sqrt(short_term_score_error.detach() + 0.25)

        optimistic_weight = torch.clamp(
            torch.sigmoid((z_short_term_q - 1.5) * 3.0) + torch.sigmoid((z_short_term_score - 1.5) * 3.0),
            min=0.0,
            max=1.0,
        )
        b, _ = optimistic_weight.shape
        optimistic_weight = torch.reshape(optimistic_weight, (b, ))
        optimistic_loss = 1 * cross_entropy(p_optimistic_prob, t_prob, optimistic_weight)

        # ownership loss
        ownership_loss = 1.5 * mse_loss_spat(p_ownership, t_ownership, global_weight)

        # win-draw-lose loss
        wdl_loss = cross_entropy(p_wdl, t_wdl)

        # all Q values loss
        q_vals_loss = mse_loss(p_q_vals, t_q_vals, global_weight)

        # all scores loss
        scores_loss = 0.0012 * huber_loss(p_scores, t_scores, 12., global_weight)

        # all short term square error loss
        q_error_loss = 2 * square_huber_loss(
            short_term_q_error,
            short_term_q_pred.detach(),
            short_term_q_target,
            delta=0.4, eps=1.0e-8,
            weight=global_weight
        )
        score_error_loss = 0.00002 * square_huber_loss(
            short_term_score_error,
            short_term_score_pred.detach(),
            short_term_score_target,
            delta=100.0, eps=1.0e-4,
            weight=global_weight
        )
        errors_loss = q_error_loss + score_error_loss

        # add all loss
        loss = prob_loss + \
                   aux_prob_loss + \
                   soft_prob_loss + \
                   soft_aux_prob_loss + \
                   optimistic_loss + \
                   ownership_loss + \
                   wdl_loss + \
                   q_vals_loss + \
                   scores_loss + \
                   errors_loss

        # make loss dictionary
        all_loss_dict = {
            "loss"               : loss,
            "prob_loss"          : prob_loss,
            "aux_prob_loss"      : aux_prob_loss,
            "soft_prob_loss"     : soft_prob_loss,
            "soft_aux_prob_loss" : soft_aux_prob_loss,
            "optimistic_loss"    : optimistic_loss,
            "ownership_loss"     : ownership_loss,
            "wdl_loss"           : wdl_loss,
            "q_vals_loss"        : q_vals_loss,
            "scores_loss"        : scores_loss,
            "errors_loss"        : errors_loss
        }
        return all_loss_dict

    def update_parameters(self, curr_steps):
        pass

    def accumulate_swa(self, other_network, swa_count):
        def accum_weights(v, w, n):
            # EMA formula
            if n <= 0:
                decay = 0.
            else:
                decay = n / (n + 1.)
            return decay * v.detach() + (1. - decay) * w.detach()

        for a, b in zip(self.parameters(), other_network.parameters()):
            a.data = accum_weights(a.data, b.data, swa_count)

        for a, b in zip(self.buffers(), other_network.buffers()):
            a.data = accum_weights(a.data, b.data, swa_count)

    def get_meta_data(self):
        stack_name = self.get_stack_name(self.stack)
        meta_stack = ",".join(stack_name)
        meta = {
            "Version": str(self.version),
            "xsize": str(self.xsize),
            "ysize": str(self.ysize),
            "ResidualBlocks": str(len(self.stack)),
            "ResidualChannels": str(self.residual_channels),
            "StackName": meta_stack,
            "PolicyHeadType": self.policy_head_type["Type"],
            "PolicyHeadChannels": str(self.policy_head_channels),
            "ValueHeadChannels": str(self.value_head_channels),
            "ActivationFunction": self.activation,
            "BatchNormMode": self.mode
        }
        return meta

    def simple_info(self):
        info = str()
        info += "NN Type: {type}\n".format(type=self.nntype)
        info += "NN size [x,y]: [{xsize}, {ysize}]\n".format(xsize=self.xsize, ysize=self.ysize)
        info += "Input channels: {channels}\n".format(channels=self.input_channels)
        info += "Residual channels: {channels}\n".format(channels=self.residual_channels)
        info += "Residual tower: size -> {s} [\n".format(s=len(self.stack))
        for s in self.get_stack_name(self.stack):
            info += "  {}\n".format(s)
        info += "]\n"
        info += "Policy head channels: {polhead}\n".format(polhead=self.policy_head_channels)
        info += "Value head channels: {valhead}\n".format(valhead=self.value_head_channels)
        info += "Value misc size: {valuemisc}\n".format(valuemisc=self.value_misc)
        info += "Policy Head Type: {polheadtype}\n".format(polheadtype=self.policy_head_type["Type"])
        info += "Default activation: {act}\n".format(act=self.activation)
        info += "Batchnorm Mode: {mode}\n".format(mode=self.mode)
        return info

    def get_name(self):
        blocks = len(self.stack)
        channels = self.residual_channels
        return "sayuri-b{}xc{}".format(blocks, channels)

    def get_stack_name(self, stack):
        stackname = list()
        for blocksetting in self.stack:
            if type(blocksetting) == str:
                blockname = blocksetting
            else:
                blockname = blocksetting["Block"]
            stackname.append(blockname)
        return stackname                

    def transfer_to_bin(self, filename):
        def write_stack(f, stack):
            f.write(str_to_bin("get stack\n"))
            for s in stack:
                f.write(str_to_bin("{}\n".format(s)))
            f.write(str_to_bin("end stack\n"))

        def write_struct(f, layers_collector):
            f.write(str_to_bin("get struct\n"))
            for layer in layers_collector:
                f.write(str_to_bin(layer.shape_to_text()))
            f.write(str_to_bin("end struct\n"))

        def write_params(f, layers_collector):
            f.write(str_to_bin("get parameters\n"))
            for layer in layers_collector:
                f.write(layer.tensors_to_text(True))
            f.write(str_to_bin("end parameters\n"))

        with open(filename, "wb") as f:
            f.write(str_to_bin("get main\n"))

            f.write(str_to_bin("get info\n"))
            f.write(str_to_bin("NNType {}\n".format(self.nntype)))
            f.write(str_to_bin("Version {}\n".format(self.version)))
            f.write(str_to_bin("FloatType {}\n".format("float32bin")))
            f.write(str_to_bin("InputChannels {}\n".format(self.input_channels)))
            f.write(str_to_bin("ResidualChannels {}\n".format(self.residual_channels)))
            f.write(str_to_bin("ResidualBlocks {}\n".format(len(self.stack))))
            f.write(str_to_bin("PolicyHeadChannels {}\n".format(self.policy_head_channels)))
            f.write(str_to_bin("ValueHeadChannels {}\n".format(self.value_head_channels)))
            f.write(str_to_bin("ValueMisc {}\n".format(self.value_misc)))
            f.write(str_to_bin("PolicyHeadType {}\n".format(self.policy_head_type["Type"])))
            f.write(str_to_bin("ActivationFunction {}\n".format(self.activation)))
            f.write(str_to_bin("end info\n"))

            write_stack(f, self.get_stack_name(self.stack))
            write_struct(f, self.layers_collector)
            write_params(f, self.layers_collector)

            f.write(str_to_bin("end main"))

    def transfer_to_text(self, filename):
        def write_stack(f, stack):
            f.write("get stack\n")
            for s in stack:
                f.write("{}\n".format(s))
            f.write("end stack\n")

        def write_struct(f, layers_collector):
            f.write("get struct\n")
            for layer in layers_collector:
                f.write(layer.shape_to_text())
            f.write("end struct\n")

        def write_params(f, layers_collector):
            f.write("get parameters\n")
            for layer in layers_collector:
                f.write(layer.tensors_to_text(False))
            f.write("end parameters\n")

        with open(filename, "w") as f:
            f.write("get main\n")

            f.write("get info\n")
            f.write("NNType {}\n".format(self.nntype))
            f.write("Version {}\n".format(self.version))
            f.write("FloatType {}\n".format("float32"))
            f.write("InputChannels {}\n".format(self.input_channels))
            f.write("ResidualChannels {}\n".format(self.residual_channels))
            f.write("ResidualBlocks {}\n".format(len(self.stack)))
            f.write("PolicyHeadChannels {}\n".format(self.policy_head_channels))
            f.write("ValueHeadChannels {}\n".format(self.value_head_channels))
            f.write("ValueMisc {}\n".format(self.value_misc))
            f.write("PolicyHeadType {}\n".format(self.policy_head_type["Type"]))
            f.write("ActivationFunction {}\n".format(self.activation))
            f.write("end info\n")

            write_stack(f, self.get_stack_name(self.stack))
            write_struct(f, self.layers_collector)
            write_params(f, self.layers_collector)

            f.write("end main")

    def add_reg_dict(self, reg_dict):
        reg_dict["input"] = []
        reg_dict["input_noreg"] = []
        reg_dict["normal"] = []
        reg_dict["normal_gamma"] = []
        reg_dict["normal_attn"] = []
        reg_dict["output"] = []
        reg_dict["noreg"] = []
        reg_dict["output_noreg"] = []

        self.input_conv.add_reg_dict(reg_dict, placement="before_block")
        for block in self.residual_tower:
            block.add_reg_dict(reg_dict)
        self.policy_conv.add_reg_dict(reg_dict, placement="after_block")
        if self.policy_head_type["Type"] == "RepLK":
            self.policy_depthwise_conv.add_reg_dict(reg_dict, placement="after_block")
            self.policy_pointwise_conv.add_reg_dict(reg_dict, placement="after_block")
        self.policy_intermediate_fc.add_reg_dict(reg_dict, placement="after_block")
        self.pol_misc.add_reg_dict(reg_dict, placement="after_block")
        self.pol_misc_pass_fc.add_reg_dict(reg_dict, placement="after_block")
        self.value_conv.add_reg_dict(reg_dict, placement="after_block")
        self.value_intermediate_fc.add_reg_dict(reg_dict, placement="after_block")
        self.ownership_conv.add_reg_dict(reg_dict, placement="after_block")
        self.value_misc_fc.add_reg_dict(reg_dict, placement="after_block")
