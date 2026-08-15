import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import packaging
import packaging.version
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Set
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
    if activation == "silu":
        return nn.SiLU(inplace=inplace)
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
    elif activation == "silu":
        gain = math.sqrt(2.0)  # Theoretically should be sqrt(2.8108), kept sqrt(2.0) for compat reasons.
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

    def add_reg_dict(self, reg_dict, placement):
        pass

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
        self.excite = FullyConnect(
            in_size=se_size,
            out_size=self.channels * 2,
            activation="identity",
            collector=collector
        )

    def initialize(self, scale, xavier_init):
        self.squeeze.initialize(scale=scale, xavier_init=xavier_init)
        self.excite.initialize(scale=scale, xavier_init=xavier_init)

    def add_reg_dict(self, reg_dict):
        self.squeeze.add_reg_dict(reg_dict)
        self.excite.add_reg_dict(reg_dict)

    def forward(self, x, mask_buffers):
        b, c, _, _ = x.size()
        mask, _, _ = mask_buffers

        seprocess = self.global_pool(x, mask_buffers)
        seprocess = self.squeeze(seprocess)
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
                       momentum_basic_batchsize=256):
        super(BatchNorm2d, self).__init__()

        if mode == "renorm" or mode == "norm":
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.float)
            )
            self.register_buffer(
                "running_var", torch.ones(num_features, dtype=torch.float)
            )
            if mode == "renorm":
                self.register_buffer(
                    "num_batches_tracked", torch.tensor(0, dtype=torch.long)
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

        # Fixup Batch Normalization layer. According to kataGo, Batch Normalization may cause
        # some wierd reuslts becuse the inference and training computation results are different.
        # Fixup can avoid the weird forwarding result. Fixup also speeds up the performance. The
        # improvement may be around x1.6 ~ x1.8 faster.
        self.fixup = mode == "fixup"

    @property
    def rmax(self) -> torch.Tensor:
        # 6k: 1.0, 40k: 3.0
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        # 25k: 5.0
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

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
        ).clamp_(1 / self.rmax, self.rmax)

        d = (
            (mean.detach() - running_mean) / running_std
        ).clamp_(-self.dmax, self.dmax)

        x = (x-mean)/std * r + d
        with torch.no_grad():
            self.num_batches_tracked += 1
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
            with torch.no_grad():
                self.running_mean += momentum * (batch_mean.detach() - self.running_mean)
                self.running_var += momentum * (batch_var.detach() - self.running_var)
        elif not self.fixup:
            # Inference step, they are equal.
            x = self._apply_norm(x, self.running_mean, self.running_var)

        x = x * (self.gamma.view(1, self.num_features, 1, 1))
        x = x + self.beta.view(1, self.num_features, 1, 1)

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
                       bias=True,
                       collector=None):
        super(Convolve, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=bias
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
            if self.bias:
                nn.init.zeros_(self.conv.bias)
        else:
            init_weights(self.conv.weight, self.activation, scale=scale)
            if self.bias:
                init_weights(self.conv.bias, self.activation, scale=bias_scale, fan_tensor=self.conv.weight)

    def add_reg_dict(self, reg_dict, placement="in_block"):
        if placement == "in_block":
            reg_dict["normal"].append(self.conv.weight)
            if self.bias:
                reg_dict["noreg"].append(self.conv.bias)
        elif placement == "before_block":
            reg_dict["input"].append(self.conv.weight)
            if self.bias:
                reg_dict["input_noreg"].append(self.conv.bias)
        else:
            reg_dict["output"].append(self.conv.weight)
            if self.bias:
                reg_dict["output_noreg"].append(self.conv.bias)

    def shape_to_text(self):
        return conv_to_text(self.in_channels, self.out_channels, self.kernel_size)

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.conv.weight, use_bin)
        if self.bias:
            out += tensor_to_text(self.conv.bias, use_bin)
        else:
            out += tensor_to_text(torch.zeros(self.out_channels), use_bin) # fill zero
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
                       activation,
                       is_pre_act=False,
                       collector=None):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_pre_act = is_pre_act
        self.use_gamma = use_gamma
        self.mode = mode
        self.placement = placement
        self.activation = activation
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        if self.is_pre_act and self.placement == "in_block":
            self.pre_bn = BatchNorm2d(
                num_features=in_channels,
                use_gamma=self.use_gamma,
                mode=mode
            )
            self.pre_act = activation_func(self.activation, inplace=True)
            self.bn = CustomIdentity()
            self.act = nn.Identity()
        else:
            self.pre_bn = CustomIdentity()
            self.pre_act = nn.Identity()
            self.bn = BatchNorm2d(
                num_features=out_channels,
                use_gamma=self.use_gamma,
                mode=mode
            )
            self.act = activation_func(self.activation, inplace=True)

        self._try_collect(collector)

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def initialize(self, scale, xavier_init):
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
            reg_dict["output"].append(self.conv.weight)
        if not isinstance(self.pre_bn, CustomIdentity):
            self.pre_bn.add_reg_dict(reg_dict, placement)
        if not isinstance(self.bn, CustomIdentity):
            self.bn.add_reg_dict(reg_dict, placement)

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
        x = self.pre_bn(x, mask)
        x = self.pre_act(x)
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
                       activation,
                       is_pre_act=False,
                       collector=None):
        # Implement it based on "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design
        # in CNNs".

        assert kernel_size >= 5, ""
        assert kernel_size % 2 == 1, ""
        super(DepthwiseConvBlock, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = self.channels
        self.is_pre_act = is_pre_act
        self.use_gamma = use_gamma
        self.mode = mode
        self.placement = placement
        self.activation = activation
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
        if self.is_pre_act and self.placement == "in_block":
            self.pre_bn = BatchNorm2d(
                num_features=self.channels,
                use_gamma=self.use_gamma,
                mode=mode
            )
            self.pre_act = activation_func(self.activation, inplace=True)
            self.bn = CustomIdentity()
            self.act = nn.Identity()
        else:
            self.pre_bn = CustomIdentity()
            self.pre_act = nn.Identity()
            self.bn = BatchNorm2d(
                num_features=channels,
                use_gamma=self.use_gamma,
                mode=mode
            )
            self.act = activation_func(self.activation, inplace=True)

        self._try_collect(collector)

    def initialize(self, scale, xavier_init):
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
        if not isinstance(self.pre_bn, CustomIdentity):
            self.pre_bn.add_reg_dict(reg_dict, placement)
        if not isinstance(self.bn, CustomIdentity):
            self.bn.add_reg_dict(reg_dict, placement)

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
        x = self.pre_bn(x, mask)
        x = self.pre_act(x)
        x = (self.conv(x) + self.rep3x3(x)) * mask
        x = self.bn(x, mask)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(ResidualBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.se_size = kwargs.get("se_size", None)
        self.mode = kwargs.get("mode", "renorm")
        self.is_pre_act = kwargs.get("is_pre_act", False)
        collector = kwargs.get("collector", None)

        self.channels = channels
        self.use_se = self.se_size is not None
        self.conv1 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            use_gamma=False,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            use_gamma=True,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation if self.is_pre_act else "identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )

        if self.is_pre_act:
            self.act = nn.Identity()
        else:
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
        if self.use_se and self.is_pre_act:
            out = self.se_module(out, mask_buffers)
        out = self.conv1(out, mask)
        out = self.conv2(out, mask)
        if self.use_se and not self.is_pre_act:
            out = self.se_module(out, mask_buffers)
        out = out + x
        if not self.is_pre_act:
            out = self.act(out)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(BottleneckBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.se_size = kwargs.get("se_size", None)
        self.mode = kwargs.get("mode", "renorm")
        self.is_pre_act = kwargs.get("is_pre_act", False)
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
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=collector
        )
        self.conv1 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=collector
        )
        self.post_btl_conv = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.outer_channels,
            kernel_size=1,
            use_gamma=True,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation if self.is_pre_act else "identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.outer_channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )

        if self.is_pre_act:
            self.act = nn.Identity()
        else:
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
        if self.use_se and self.is_pre_act:
            out = self.se_module(out, mask_buffers)
        out = self.pre_btl_conv(out, mask)
        out = self.conv1(out, mask)
        out = self.conv2(out, mask)
        out = self.post_btl_conv(out, mask)
        if self.use_se and not self.is_pre_act:
            out = self.se_module(out, mask_buffers)
        out = out + x
        if not self.is_pre_act:
            out = self.act(out)
        return out

class NestedBottleneckBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(NestedBottleneckBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.se_size = kwargs.get("se_size", None)
        self.mode = kwargs.get("mode", "renorm")
        self.is_pre_act = kwargs.get("is_pre_act", False)
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
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=collector
        )
        self.block1 = ResidualBlock(
            channels=self.inner_channels,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            activation=self.activation,
            collector=collector
        )
        self.block2 = ResidualBlock(
            channels=self.inner_channels,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            activation=self.activation,
            collector=collector
        )
        self.post_btl_conv = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.outer_channels,
            kernel_size=1,
            use_gamma=True,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation if self.is_pre_act else "identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.outer_channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )

        if self.is_pre_act:
            self.act = nn.Identity()
        else:
            self.act = activation_func(self.activation, inplace=True)

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        if xavier_init:
            self.pre_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
            self.block1.initialize(fixup_scale=1.0, se_fixup_scale=1.0, xavier_init=xavier_init)
            self.block2.initialize(fixup_scale=1.0, se_fixup_scale=1.0, xavier_init=xavier_init)
            self.post_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
            if self.use_se:
                self.se_module.initialize(scale=1.0, xavier_init=xavier_init)
        else:
            if self.use_se:
                self.pre_btl_conv.initialize(
                    scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.block1.initialize(
                    fixup_scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)),
                    se_fixup_scale=1.0,
                    xavier_init=xavier_init)
                self.block2.initialize(
                    fixup_scale=math.pow(se_fixup_scale, 1.0 / (1.0 + 2.0)),
                    se_fixup_scale=1.0,
                    xavier_init=xavier_init)
                self.post_btl_conv.initialize(scale=0.0, xavier_init=xavier_init)
                self.se_module.initialize(scale=se_fixup_scale, xavier_init=xavier_init)
            else:
                self.pre_btl_conv.initialize(
                    scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)), xavier_init=xavier_init)
                self.block1.initialize(
                    fixup_scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)),
                    se_fixup_scale=1.0,
                    xavier_init=xavier_init)
                self.block2.initialize(
                    fixup_scale=math.pow(fixup_scale, 1.0 / (1.0 + 2.0)),
                    se_fixup_scale=1.0,
                    xavier_init=xavier_init)
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
        if self.use_se and self.is_pre_act:
            out = self.se_module(out, mask_buffers)
        out = self.pre_btl_conv(out, mask)
        out = self.block1(out, mask_buffers)
        out = self.block2(out, mask_buffers)
        out = self.post_btl_conv(out, mask)
        if self.use_se and not self.is_pre_act:
            out = self.se_module(out, mask_buffers)
        out = out + x
        if not self.is_pre_act:
            out = self.act(out)
        return out

class MixerBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(MixerBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.se_size = kwargs.get("se_size", None)
        self.kernel_size = kwargs.get("kernel_size", 7)
        self.ffn_expansion_ratio = kwargs.get("ffn_expansion_ratio", 1.5)
        self.version = kwargs.get("version", 1)
        self.mode = kwargs.get("mode", "renorm")
        self.is_pre_act = kwargs.get("is_pre_act", False)
        collector = kwargs.get("collector", None)

        self.channels = channels
        self.use_se = self.se_size is not None
        assert self.version in [1, 2], ""

        self.depthwise_conv = DepthwiseConvBlock(
            channels=self.channels,
            kernel_size=self.kernel_size,
            use_gamma=True,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
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
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=collector
        )
        self.ffn2 = ConvBlock(
            in_channels=self.ffn_channels,
            out_channels=self.channels,
            kernel_size=1,
            use_gamma=True,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation if self.is_pre_act else "identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )

        if self.is_pre_act:
            self.act = nn.Identity()
        else:
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
            if self.use_se and self.is_pre_act:
                out = self.se_module(out, mask_buffers)
            x = self.depthwise_conv(out, mask) + x
            out = x
            out = self.ffn1(out, mask)
            out = self.ffn2(out, mask)
            if self.use_se and not self.is_pre_act:
                out = self.se_module(out, mask_buffers)
            if not self.is_pre_act:
                out = out + x
                out = self.act(out)
        elif self.version == 2:
            out = x
            if self.use_se and self.is_pre_act:
                out = self.se_module(out, mask_buffers)
            out = self.depthwise_conv(out, mask)
            out = self.ffn1(out, mask)
            out = self.ffn2(out, mask)
            if self.use_se and not self.is_pre_act:
                out = self.se_module(out, mask_buffers)
            out = out + x
            if not self.is_pre_act:
                out = self.act(out)
        return out

# Simplified functional replacement for better ONNX export
class CustomRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RMSNormMask(torch.nn.Module):
    """RMSNorm applied per spatial position across channels, with masking for off-board positions.
    Computes RMS across both channels and spatial positions (masked), producing
    one scalar RMS per sample instead of per position.
    If cgroup_size is not None and greater than 0, breaks channels into groups of the given size
    and normalizes within each group across channels_in_group x H x W (like group norm but RMS only,
    no mean centering).
    """
    def __init__(self, c_in, cgroup_size):
        super(RMSNormMask, self).__init__()
        self.c_in = c_in
        self.cgroup_size = cgroup_size
        self.eps = 1e-6
        if cgroup_size is not None and cgroup_size > 0:
            assert c_in % cgroup_size == 0, f"c_in ({c_in}) must be divisible by cgroup_size ({cgroup_size})"
            self.num_groups = c_in // cgroup_size
        self.gamma = torch.nn.Parameter(torch.ones(c_in))
        self.beta = torch.nn.Parameter(torch.zeros(c_in))

    def add_reg_dict(self, reg_dict, placement="after_block"):
        reg_dict["output"].append(self.gamma)
        reg_dict["output"].append(self.beta)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        if self.cgroup_size is not None and self.cgroup_size > 0:
            # Group-wise spatial RMS: normalize within each group of channels across group_channels x H x W
            N, C, H, W = x.shape
            x_grouped = x.view(N, self.num_groups, self.cgroup_size, H, W)
            mask_grouped = mask.view(N, 1, 1, H, W)
            # mean of x^2 over group channels and masked spatial positions
            mean_sq = torch.sum(
                x_grouped * x_grouped * mask_grouped,
                dim=(2, 3, 4),
                keepdim=True) / (self.cgroup_size * mask_sum_hw.unsqueeze(2) + self.eps)
            rms = torch.sqrt(mean_sq + self.eps)
            out = x_grouped / rms
            out = out.view(N, C, H, W)
        else:
            # RMS across C,H,W for masked positions only, one scalar per sample
            # mean of x^2 over C and masked spatial positions
            mean_sq = torch.sum(x * x * mask, dim=(1, 2, 3), keepdim=True) / (self.c_in * mask_sum_hw + self.eps)
            rms = torch.sqrt(mean_sq + self.eps)
            out = x / rms
        return (out * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)) * mask

def compute_learnable_rope_cos_sin(s_x, s_y, freqs):
    """Compute cos/sin rotation tables from spatial positions and learnable 2D frequencies.
    s_x: (...,) float tensor of column positions
    s_y: (...,) float tensor of row positions
    freqs: (H_kv, P, 2) learnable frequencies (omega_x, omega_y) per head per pair
    Returns: (cos, sin) each of shape (..., H_kv, P)
    """
    # angles: (..., H_kv, P) = omega_x * x + omega_y * y
    angles = s_x.unsqueeze(-1).unsqueeze(-1) * freqs[:, :, 0] + s_y.unsqueeze(-1).unsqueeze(-1) * freqs[:, :, 1]
    return torch.cos(angles), torch.sin(angles)

def apply_learnable_rotary_emb(xq, xk, cos_q, sin_q, cos_k, sin_k, learned_rope_cast=False):
    """Apply learnable rotary position embeddings to Q and K tensors.
    xq: (Batch, Seq, num_heads, Dim)
    xk: (Batch, Seq, num_kv_heads, Dim)
    cos_q, sin_q: (Seq, num_heads, Dim/2) or (Batch, Seq, num_heads, Dim/2)
    cos_k, sin_k: (Seq, num_kv_heads, Dim/2) or (Batch, Seq, num_kv_heads, Dim/2)
    """
    def _rotate(x, cos, sin):
        B, S, H, D = x.shape
        P = D // 2
        x_pairs = x.view(B, S, H, P, 2)
        x0, x1 = x_pairs.unbind(dim=-1)  # each (B, S, H, P)
        if cos.dim() == 3:
            cos = cos.unsqueeze(0)  # (1, S, H, P)
            sin = sin.unsqueeze(0)
        if learned_rope_cast:
            cos = cos.to(dtype=x.dtype)
            sin = sin.to(dtype=x.dtype)
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.reshape(B, S, H, D).type_as(x)

    return _rotate(xq, cos_q, sin_q), _rotate(xk, cos_k, sin_k)

TAB_KQ = "tab_kq"
# When present in block_shared_data (training-time attention logit penalty), each attention layer
# appends its per-(batch,head) differentiable upper bound on pre-mask attention logit magnitude.
ATTN_LOGIT_UB = "attn_logit_ub"

@dataclass
class TABKeyQueryData:
    """Precomputed factored TAB keys and queries, shared across all blocks in a forward pass.
    Instead of materializing (N, T, S, S) templates, stores the factored keys/queries
    so they can be concatenated onto the main attention K/Q.
    By convention, keys and/or queries are pre-scaled by 1/sqrt of the appropriate quantity
    so that a weighted combination does not need further scaling.
    """
    keys: torch.Tensor    # (N, 2*F, 1, S) - single complex key shared across templates
    queries: torch.Tensor # (N, 2*F, T, S) - complex query vectors per template

def tab_rotate(z, cos_a, sin_a):
    """Apply complex rotation to z.
    z: (*, 2, c_z, H, W) where dim -4 is [real, imag]
    cos_a, sin_a: broadcastable to (*, 1, c_z, H, W)
    Returns: same shape as z
    """
    r = z[:, 0:1, :, :, :]  # (*, 1, c_z, H, W)
    i = z[:, 1:2, :, :, :]
    new_r = r * cos_a - i * sin_a
    new_i = r * sin_a + i * cos_a
    return torch.cat([new_r, new_i], dim=-4)

class ComplexConv2d(torch.nn.Module):
    """A 2D convolution that enforces complex multiplication structure.

    Stores real_kernel and imag_kernel of shape (c_out, c_in, K, K).
    Builds the (2*c_out, 2*c_in, K, K) block-structured kernel:
        [[real_kernel, -imag_kernel],
         [imag_kernel,  real_kernel]]
    and applies F.conv2d.

    Input: (*, 2*c_in, H, W), Output: (*, 2*c_out, H, W).
    """
    def __init__(self, c_in, c_out=None, kernel_size=1, dilation=1):
        super().__init__()
        if c_out is None:
            c_out = c_in
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.real_kernel = torch.nn.Parameter(torch.empty(c_out, c_in, kernel_size, kernel_size))
        self.imag_kernel = torch.nn.Parameter(torch.empty(c_out, c_in, kernel_size, kernel_size))

    def forward(self, x):
        # We encode c_in x c_in complex convolution as a 2*c_in x 2*c_in real convolution
        # where the kernel is constrained to have the appropriate structure.
        top = torch.cat([self.real_kernel, -self.imag_kernel], dim=1)  # (c_out, 2*c_in, K, K)
        bot = torch.cat([self.imag_kernel, self.real_kernel], dim=1)   # (c_out, 2*c_in, K, K)
        kernel = torch.cat([top, bot], dim=0)  # (2*c_out, 2*c_in, K, K)
        padding = self.dilation * (self.kernel_size // 2)
        return torch.nn.functional.conv2d(x, kernel, padding=padding, dilation=self.dilation)

    def initialize(self, activation, scale=1.0):
        init_weights(self.real_kernel, activation, scale=scale / math.sqrt(2.0))
        init_weights(self.imag_kernel, activation, scale=scale / math.sqrt(2.0))

class TABEquivariantBlock(torch.nn.Module):
    """One equivariant residual block for TAB.

    Contains two complex convolutions (first with dilation, second without)
    with activations and RoPE-style rotations for equivariance.
    """
    def __init__(self, c_z, activation, dilation):
        super().__init__()
        self.act1 = activation_func(activation)
        self.conv1 = ComplexConv2d(c_z, kernel_size=3, dilation=dilation)
        self.act2 = activation_func(activation)
        self.conv2 = ComplexConv2d(c_z, kernel_size=3, dilation=1)
        self.c_z = c_z

    def forward(self, z, cos_a, sin_a, block_idx):
        """
        z: (NF, 2, c_z, H, W)
        cos_a, sin_a: (NF, 1, 1, H, W)
        block_idx: int, for variance normalization
        """
        zskip = z
        # Normalize - variance after block_idx prior blocks is proportional to block_idx + 1
        # (if we model the input as variance 1 and each block as contributing variance 1)
        z = z * (1.0 / math.sqrt(block_idx + 1))
        z = self.act1(z)
        z = tab_rotate(z, cos_a, sin_a)
        z = z.reshape(z.shape[0], 2 * self.c_z, z.shape[3], z.shape[4])
        z = self.conv1(z)
        z = z.reshape(z.shape[0], 2, self.c_z, z.shape[2], z.shape[3])
        z = tab_rotate(z, cos_a, -sin_a)
        z = self.act2(z)
        z = tab_rotate(z, cos_a, sin_a)
        z = z.reshape(z.shape[0], 2 * self.c_z, z.shape[3], z.shape[4])
        z = self.conv2(z)
        z = z.reshape(z.shape[0], 2, self.c_z, z.shape[2], z.shape[3])
        z = tab_rotate(z, cos_a, -sin_a)
        z = z + zskip
        return z

    def initialize(self, activation):
        self.conv1.initialize(activation, scale=1.0)
        self.conv2.initialize(activation, scale=1.0)

class TABModule(torch.nn.Module):
    """Shared module that generates factored input-dependent attention bias.

    Uses a stack of rotationally-equivariant complex convolutional blocks
    with learnable 2D RoPE-style frequencies. Produces factored keys and queries
    via complex key-query projections.

    Uses a single shared key projection and T query projections,
    returning factored (keys, queries) that are concatenated onto
    the main attention K/Q in each transformer block.

    Computed once and shared across all transformer blocks.
    """
    def __init__(
        self,
        trunk_channels,
        tab_c_z,
        tab_num_templates,
        tab_num_freqs,
        tab_num_blocks,
        tab_dilation,
        activation,
        pos_len
    ):
        super().__init__()
        self.tab_c_z = tab_c_z
        self.tab_num_freqs = tab_num_freqs
        self.tab_num_templates = tab_num_templates
        self.tab_num_blocks = tab_num_blocks
        self.activation = activation

        # 1x1 conv to project trunk channels -> 2*F*c_z (interpreted as F*c_z complex values)
        self.input_proj = torch.nn.Conv2d(trunk_channels, 2 * tab_num_freqs * tab_c_z, kernel_size=1, bias=False)

        # Learnable 2D RoPE frequencies: (F, 2) for (omega_X, omega_Y)
        # Geometric initialization from 1 rad/square to 1/50 rad/square
        log_lo = math.log(1.0 / 50.0)
        log_hi = math.log(1.0)
        init_freqs = torch.exp(torch.empty(tab_num_freqs, 2).uniform_(log_lo, log_hi))
        init_freqs = init_freqs * (torch.randint(0, 2, (tab_num_freqs, 2)) * 2 - 1).float()
        self.rope_freqs = torch.nn.Parameter(init_freqs)

        self.blocks = torch.nn.ModuleList()
        for _ in range(tab_num_blocks):
            self.blocks.append(TABEquivariantBlock(tab_c_z, activation, tab_dilation))

        self.final_act = activation_func(activation)
        self.key_proj = ComplexConv2d(tab_c_z, 1, kernel_size=1)
        self.query_proj = ComplexConv2d(tab_c_z, tab_num_templates, kernel_size=1)

    def forward(self, x, mask):
        """
        x: (N, C, H, W) trunk output
        mask: (N, 1, H, W) or None
        Returns: (keys, queries) with keys (N, 2*F, 1, S) and queries (N, 2*F, T, S), pre-scaled
        """
        N, C, H, W = x.shape
        S = H * W
        F = self.tab_num_freqs
        T = self.tab_num_templates
        c_z = self.tab_c_z

        z = self.input_proj(x)  # (N, 2*F*c_z, H, W)
        z = z.view(N, F, 2, c_z, H, W)

        # Precompute angles from learnable frequencies and grid coordinates
        gy = torch.arange(H, device=x.device, dtype=x.dtype)
        gx = torch.arange(W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)
        # angles[f, y, x] = omega_f_X * x + omega_f_Y * y
        angles = self.rope_freqs[:, 0:1].unsqueeze(-1) * grid_x.unsqueeze(0) + \
                 self.rope_freqs[:, 1:2].unsqueeze(-1) * grid_y.unsqueeze(0)  # (F, H, W)
        cos_a = torch.cos(angles).view(1, F, 1, 1, H, W)  # (1, F, 1, 1, H, W)
        sin_a = torch.sin(angles).view(1, F, 1, 1, H, W)

        # Apply mask to zero off-board positions
        if mask is not None:
            z = z * mask.view(N, 1, 1, 1, H, W)

        # Fold N*F into batch dimension for batched processing
        z = z.reshape(N * F, 2, c_z, H, W)
        cos_a_batched = cos_a.expand(N, F, 1, 1, H, W).reshape(N * F, 1, 1, H, W)
        sin_a_batched = sin_a.expand(N, F, 1, 1, H, W).reshape(N * F, 1, 1, H, W)

        # Equivariant blocks
        block_idx = 0
        for block in self.blocks:
            z = block(z, cos_a_batched, sin_a_batched, block_idx)
            block_idx += 1

        # Normalize to variance 1 - variance after block_idx prior blocks is proportional to block_idx + 1
        # (if we model the input as variance 1 and each block as contributing variance 1)
        z = z * (1.0 / math.sqrt(block_idx + 1))

        # Final projection: activate, rotate into RoPE space, project keys/queries
        z = self.final_act(z)
        z = tab_rotate(z, cos_a_batched, sin_a_batched)

        z_flat = z.reshape(N * F, 2 * c_z, H, W)

        keys = self.key_proj(z_flat)      # (N*F, 2, H, W)
        queries = self.query_proj(z_flat)  # (N*F, 2*T, H, W)
        # Reshape: (N*F, 2*(T or 1), H, W) -> (N, 2*F, (T or 1), S)
        keys = keys.view(N, 2 * F, 1, S)
        queries = queries.view(N, 2 * F, T, S)
        return keys / math.sqrt(F), queries / math.sqrt(self.tab_num_templates)

    def initialize(self):
        init_weights(self.input_proj.weight, self.activation, scale=1.0)
        for block in self.blocks:
            block.initialize(self.activation)
        self.key_proj.initialize(self.activation, scale=1.0)
        self.query_proj.initialize(self.activation, scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["tab_module"].append(self.input_proj.weight)
        reg_dict["noreg"].append(self.rope_freqs)
        for block in self.blocks:
            reg_dict["tab_module"].append(block.conv1.real_kernel)
            reg_dict["tab_module"].append(block.conv1.imag_kernel)
            reg_dict["tab_module"].append(block.conv2.real_kernel)
            reg_dict["tab_module"].append(block.conv2.imag_kernel)
        reg_dict["tab_module"].append(self.key_proj.real_kernel)
        reg_dict["tab_module"].append(self.key_proj.imag_kernel)
        reg_dict["tab_module"].append(self.query_proj.real_kernel)
        reg_dict["tab_module"].append(self.query_proj.imag_kernel)

class TransformerAttentionBlock(nn.Module):
    """Self-attention half and Feed-forward half of a transformer block with its own residual connection.
    Contains: RMSNorm -> Q/K/V projections -> (optional RoPE) -> attention -> output projection
    Returns NCHW.
    """
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(TransformerAttentionBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.pos_len = kwargs.get("pos_len", 19)
        self.use_tab = kwargs.get("use_tab", False)
        self.use_qk_norm = kwargs.get("attention_qk_norm", False)
        self.num_heads = kwargs.get("transformer_heads", 3)
        self.num_kv_heads = kwargs.get("transformer_kv_heads", self.num_heads)
        # Compute how many query heads each KV head serves (group size)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_head_dim = kwargs.get("attention_query_head_dim", channels // self.num_heads)
        self.v_head_dim = kwargs.get("attention_value_head_dim", channels // self.num_heads)
        self.ffn_dim = kwargs.get("transformer_ffn_channels", 256)
        self.use_swiglu = kwargs.get("use_swiglu", True)
        self.use_depthwise_conv = kwargs.get("transformer_ffn_depthwise_conv", False)
        # Under AMP, cast the small cos/sin rotation tables to the input dtype before the
        # batch-sized Q/K rotation, instead of promoting the batch-sized rotation
        # intermediates to FP32. The trigonometric functions themselves remain FP32.
        self.learned_rope_cast_to_input_dtype = kwargs.get("learned_rope_cast_to_input_dtype", False)

        assert self.q_head_dim % 4 == 0, f"Query head dim must be divisible by 4 for 2D RoPE"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"Query heads ({self.num_heads}) must be divisible by KV heads ({self.num_kv_heads})"
        self.q_proj = torch.nn.Linear(channels, self.num_heads * self.q_head_dim, bias=False)
        self.k_proj = torch.nn.Linear(channels, self.num_kv_heads * self.q_head_dim, bias=False)
        self.v_proj = torch.nn.Linear(channels, self.num_kv_heads * self.v_head_dim, bias=False)
        self.out_proj = torch.nn.Linear(self.num_heads * self.v_head_dim, channels, bias=False)

        # QK-norm: RMSNorm on Q and K per-head before the attention dot product.
        # See ViT-22B, etc.
        if self.use_qk_norm:
            self.q_norm = CustomRMSNorm(self.q_head_dim, eps=1e-6)
            self.k_norm = CustomRMSNorm(self.q_head_dim, eps=1e-6)

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

        if self.use_tab:
            tab_d1 = kwargs.get("tab_d1", 16)
            tab_d2 = kwargs.get("tab_d2", 16)
            self.tab_num_templates = kwargs.get("tab_num_templates", 32)
            # Per-head weights: one per TAB template.
            # TAB weights are per-template (shared across 2*F real/imag freq channels).
            self.total_num_weights = self.tab_num_templates
            self.tab_proj1 = torch.nn.Linear(channels, tab_d1, bias=False)
            self.tab_proj2 = torch.nn.Linear(tab_d1, tab_d2, bias=False)
            self.tab_norm1 = CustomRMSNorm(tab_d2, eps=1e-6)
            self.tab_proj3 = torch.nn.Linear(tab_d2, self.num_heads * self.total_num_weights, bias=False)
            self.tab_norm2 = CustomRMSNorm(self.num_heads * self.total_num_weights, eps=1e-6)
            self.tab_act1 = activation_func(self.activation, inplace=False)
            self.tab_act2 = activation_func(self.activation, inplace=False)

        self.attn_norm = CustomRMSNorm(channels, eps=1e-6)

        self.ffn_linear1 = torch.nn.Linear(channels, self.ffn_dim, bias=False)
        if self.use_swiglu:
            self.ffn_linear_gate = torch.nn.Linear(channels, self.ffn_dim, bias=False)
            self.ffn_act = torch.nn.SiLU(inplace=False)
        else:
            self.ffn_act = activation_func(self.activation, inplace=False)
        if self.use_depthwise_conv:
            self.ffn_dwconv = torch.nn.Conv2d(
                self.ffn_dim, self.ffn_dim, kernel_size=3, padding=1, groups=self.ffn_dim, bias=False)
        self.ffn_linear2 = torch.nn.Linear(self.ffn_dim, channels, bias=False)
        self.ffn_norm = CustomRMSNorm(channels, eps=1e-6)

    def add_reg_dict(self, reg_dict):
        for name, param in self.named_parameters():
            if "norm" in name:
                reg_dict["noreg"].append(param)
                continue
            if "weight" in name:
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                    reg_dict["normal_attn"].append(param)
                elif "tab_proj" in name:
                    reg_dict["normal_tab"].append(param)
                else:
                    reg_dict["normal"].append(param)
            else:
                reg_dict["noreg"].append(param)

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        pass

    def _compute_tab_bias(self, x_norm, mask, mask_sum_hw, block_shared_data):
        """Compute attention bias from TAB factored keys/queries.
        x_norm: (B, S, C) normalized token representations
        mask: (N, 1, H, W) or None
        mask_sum_hw: (N, 1, 1, 1) or None
        block_shared_data: dict with precomputed template/key-query data
        Returns: (extra_kq) where
            extra_kq: (extra_k, extra_q) to concatenate onto main K/Q, or None
        """
        batch_size, seq_len, _ = x_norm.shape

        # Per-token projection
        y = self.tab_proj1(x_norm) # (B, S, d1)

        # Masked mean pooling over valid positions
        if mask is not None:
            mask_flat = mask.view(batch_size, seq_len, 1)  # (B, S, 1)
            y = y * mask_flat
            pooled = y.sum(dim=1) / mask_sum_hw.view(batch_size, 1)  # (B, d1)
        else:
            pooled = y.mean(dim=1)                       # (B, d1)

        # Compress + activation + norm
        z = self.tab_act1(self.tab_proj2(pooled))         # (B, d2)
        z = self.tab_norm1(z)

        # Generate per-head weights for all bias mechanisms
        z = self.tab_act2(self.tab_proj3(z))              # (B, H*total_num_weights)
        z = self.tab_norm2(z)
        z = z.view(batch_size, self.num_heads, self.total_num_weights)  # (B, H, W_total)

        extra_k_parts = []
        extra_q_parts = []
        idx = 0

        # TAB contribution: mix templates in K/Q space, then append 2*F_tab dims.
        # Instead of keeping T templates separate (which would need 2*F*T extra dims),
        # we contract over templates before the dot product, yielding one mixed
        # key/query per frequency per head - only 2*F_tab extra dims.
        z_tab = z[:, :, idx:idx + self.tab_num_templates]  # (B, H, T)
        idx += self.tab_num_templates
        tab_data = block_shared_data[TAB_KQ]
        tab_keys = tab_data.keys         # (N, 2*F_tab, 1, S)
        tab_queries = tab_data.queries   # (N, 2*F_tab, T, S)
        # Mix queries across templates: einsum "bht, bfts -> bhfs"
        # z_tab: (B, H, T), tab_queries: (B, 2*F_tab, T, S) -> mixed_q: (B, H, 2*F_tab, S)
        mixed_q = torch.einsum("bht,bfts->bhfs", z_tab, tab_queries)  # (B, H, 2*F_tab, S)
        extra_q_parts.append(mixed_q.permute(0, 1, 3, 2))   # (B, H, S, 2*F_tab)

        tab_keys = tab_keys.squeeze(2).permute(0, 2, 1)       # (B, S, 2*F_tab)
        tab_keys = tab_keys.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        extra_k_parts.append(tab_keys)  # (B, H, S, 2*F_tab)

        assert idx == self.total_num_weights, ""

        extra_kq = None
        if extra_k_parts:
            extra_k = torch.cat(extra_k_parts, dim=-1)  # (B, H, S, D_extra)
            extra_q = torch.cat(extra_q_parts, dim=-1)  # (B, H, S, D_extra)
            extra_kq = (extra_k, extra_q)

        return extra_kq

    def forward(self, x, mask, mask_sum_hw, mask_sum, block_shared_data=None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: unused
        mask_sum: unused

        Returns: NCHW
        """
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1) # [B, (H * W), C]

        x_norm = self.attn_norm(x_in)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.q_head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.v_head_dim)

        # compute from arange.
        s_idx = torch.arange(seq_len, device=q.device)
        s_y = (s_idx // self.pos_len).float()  # row
        s_x = (s_idx % self.pos_len).float()   # col
        cos_k, sin_k = compute_learnable_rope_cos_sin(s_x, s_y, self.rope_freqs)  # ([B,] S, H_kv, P)
        # For Q: expand kv head freqs to match num_heads if using grouped-query attention.
        # cos_k/sin_k are ([B,] S, H_kv, P); repeat each kv head n_rep times along a new axis
        # inserted right after the head axis, so query head h maps to kv head h // n_rep --
        # matching the k/v expansion below and the C++ backends' kvh = h * num_kv / num_heads.
        if self.n_rep > 1:
            cos_q = cos_k.unsqueeze(-2).expand(
                *cos_k.shape[:-1],
                self.n_rep,
                cos_k.shape[-1]
                ).reshape(*cos_k.shape[:-2], self.num_heads, -1)
            sin_q = sin_k.unsqueeze(-2).expand(
                *sin_k.shape[:-1],
                self.n_rep,
                sin_k.shape[-1]
                ).reshape(*sin_k.shape[:-2], self.num_heads, -1)
        else:
            cos_q = cos_k
            sin_q = sin_k
        q, k = apply_learnable_rotary_emb(
            q, k, cos_q, sin_q, cos_k, sin_k, self.learned_rope_cast_to_input_dtype)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.q_head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.q_head_dim)
            v = v.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.v_head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.v_head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        extra_kq = None
        if self.use_tab:
            extra_kq = self._compute_tab_bias(x_norm, mask, mask_sum_hw, block_shared_data)

        if mask is not None:
            mask_flat = mask.reshape(batch_size, 1, 1, seq_len)
            attn_mask = torch.zeros_like(mask_flat, dtype=q.dtype)
            attn_mask.masked_fill_(mask_flat == 0, float('-inf'))
        else:
            attn_mask = None

        # Default scaling for q/k dot product, 1/sqrt(query head dim)
        scale = 1.0 / math.sqrt(self.q_head_dim)

        if extra_kq is not None:
            # Concatenate extra keys/queries (from TAB) onto main K/Q.
            # q, k: (B, H, S, d_head), extra_k, extra_q: (B, H, S, D_extra)
            extra_k, extra_q = extra_kq

            # Pre-scale q and disable the overall scale passed to scaled_dot_product_attention
            # since the different extra q and extra k will have their own scaling.
            # The convention is that their scaling, if any, is already pre-multiplied in.
            q = q * scale
            scale = 1.0

            q = torch.cat([q, extra_q], dim=-1)  # (B, H, S, d_head + D_extra)
            k = torch.cat([k, extra_k], dim=-1)  # (B, H, S, d_head + D_extra)
            # v stays (B, H, S, d_head), scaled_dot_product_attention supports differing channels for v than q/k

        # Hack: record pre-mask logit min/max when capture is enabled
        # (see the commented-out ATTN_LOGIT_STATS_CAPTURE block above the class).
        # if ATTN_LOGIT_STATS_CAPTURE is not None:
        #     _capture_attn_logit_stats(
        #         ATTN_LOGIT_STATS_CAPTURE, self.name, q, k, scale, mask, batch_size, seq_len
        #     )

        # Training-time attention logit penalty (see Model.attn_logit_penalty_cap): record the
        # differentiable per-(batch,head) upper bound on pre-mask attention logit magnitude,
        #   scale * max_i ||q_i|| * max_j ||k_j|| >= max_ij |scale * q_i . k_j|.
        # The max deliberately includes off-board garbage positions, since inference backends
        # compute logits at those positions too before masking, and their magnitudes are what
        # constrain the fp16-safe additive mask bias constants. (Correct for the extra_kq/tab
        # path too: there q is pre-multiplied by the true scale and `scale` is 1.0.)
        #
        # Cost notes: computed on only the first `num_batch_items` samples
        # (see attn_logit_penalty_batch_frac) since the hinge dynamics are slow enough that subsampled
        # gradients suffice, and the dominant cost is the extra backward through every layer's
        # q/k. The explicit .float() BEFORE squaring is required for correctness, not just
        # accumulation: an fp16 square overflows to inf at |q| >= 256 (hot-layer components are
        # already ~65) regardless of any fp32 accumulator, and eager would hit that even though
        # inductor's fused kernels happen to compute fp16 pointwise math in fp32 registers. Under
        # torch.compile the cast fuses into the reduction prologue, so no fp32 copy of q/k is
        # materialized. The sqrt happens after the position amax.
        if block_shared_data is not None and ATTN_LOGIT_UB in block_shared_data:
            ub_state = block_shared_data[ATTN_LOGIT_UB]
            nb = ub_state["num_batch_items"]
            qs = q[:nb].float()
            ks = k[:nb].float()
            ub_qnorm2 = (qs * qs).sum(dim=-1)  # (B', H, S)
            ub_knorm2 = (ks * ks).sum(dim=-1)  # (B', H, S)
            ub_state["ubs"].append(
                scale * torch.sqrt(ub_qnorm2.amax(dim=-1) * ub_knorm2.amax(dim=-1))  # (B', H)
            )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scale,
        )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        attn_output = self.out_proj(attn_output)

        ffn_in = x_in + attn_output
        xn = self.ffn_norm(ffn_in)

        if self.use_swiglu:
            x1 = self.ffn_linear1(xn)
            x1 = self.ffn_act(x1)
            x_gate = self.ffn_linear_gate(xn)
            x1 = x1 * x_gate
        else:
            x1 = self.ffn_linear1(xn)
            x1 = self.ffn_act(x1)
        if self.use_depthwise_conv:
            # Reshape to NCHW for depthwise conv, apply mask, reshape back
            x1_spatial = x1.permute(0, 2, 1).view(batch_size, self.ffn_dim, height, width)
            x1_spatial = self.ffn_dwconv(x1_spatial) * mask
            x1 = x1_spatial.view(batch_size, self.ffn_dim, -1).permute(0, 2, 1)
        x1 = ffn_in + self.ffn_linear2(x1)

        result = x1.permute(0, 2, 1).view(batch_size, channels, height, width)

        return result

class NestedBottleneckTransformerBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(NestedBottleneckTransformerBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.mode = kwargs.get("mode", "fixup")
        self.is_pre_act = kwargs.get("is_pre_act", True)
        self.internal_length = kwargs.get("internal_length", 2)
        assert self.internal_length >= 1, ""
        assert self.bottleneck_channels is not None, ""
        assert self.bottleneck_channels % 2 == 0, ""

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
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=None
        )
        self.blockstack = torch.nn.ModuleList()
        for i in range(self.internal_length):
            self.blockstack.append(TransformerAttentionBlock(channels=self.inner_channels, **kwargs))

        self.post_btl_conv = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.outer_channels,
            kernel_size=1,
            use_gamma=True,
            mode=self.mode,
            is_pre_act=self.is_pre_act,
            placement="in_block",
            activation=self.activation,
            collector=None
        )

    def initialize(self, fixup_scale, se_fixup_scale, xavier_init):
        if xavier_init:
            self.pre_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
            self.post_btl_conv.initialize(scale=1.0, xavier_init=xavier_init)
        else:
            self.pre_btl_conv.initialize(
                scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)), xavier_init=xavier_init)
            self.post_btl_conv.initialize(scale=0.0, xavier_init=xavier_init)

    def add_reg_dict(self, reg_dict):
        self.pre_btl_conv.add_reg_dict(reg_dict)
        for block in self.blockstack:
            block.add_reg_dict(reg_dict)
        self.post_btl_conv.add_reg_dict(reg_dict)

    def forward(self, x, mask, mask_sum_hw, mask_sum, block_shared_data=None):
        out = x
        out = self.pre_btl_conv(out, mask)
        for block in self.blockstack:
            in_feature = out
            out = block(in_feature, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum,
                block_shared_data=block_shared_data)
            out = in_feature + out
        out = self.post_btl_conv(out, mask)
        out = out + x
        return out

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()

        self.layers_collector = list()

        self.cfg = cfg

        self.nntype = cfg.nntype  # default:None

        self.activation = cfg.activation.lower()  # default:"relu"
        self.input_channels = cfg.input_channels  # default:43
        self.residual_channels = cfg.residual_channels  # default:None
        self.xsize = cfg.boardsize  # default:19
        self.ysize = cfg.boardsize  # default:19
        self.pos_len = cfg.boardsize  # default:19
        self.policy_head_channels = cfg.policy_head_channels  # default:None
        self.value_head_channels = cfg.value_head_channels    # default:None
        self.se_ratio = cfg.se_ratio  # default:2
        self.policy_head_type = cfg.policy_head_type  # default:{"Type" : "Normal"}
        if type(self.policy_head_type) == str:
            self.policy_head_type = { "Type" : self.policy_head_type }  # default:{"Type" : "Normal"}
        self.value_misc = 15
        self.policy_outs = 5
        self.stack = cfg.stack  # default:[]
        self.version = 5
        self.mode = cfg.mode  # default:"renorm"
        self.is_pre_act = cfg.is_pre_act # default:False
        self.use_tab = cfg.use_tab # default:False
        self.final_block_cgroup_size = cfg.final_block_cgroup_size  # default:None
        self.attention_qk_norm = cfg.attention_qk_norm  # default:False
        self.transformer_heads = cfg.transformer_heads  # default:3
        self.transformer_kv_heads = cfg.transformer_kv_heads  # default:3
        self.attention_query_head_dim = cfg.attention_query_head_dim  # default:32
        self.attention_value_head_dim = cfg.attention_value_head_dim  # default:32
        self.learned_rope_cast_to_input_dtype = cfg.learned_rope_cast_to_input_dtype  # default:False
        self.transformer_ffn_channels = cfg.transformer_ffn_channels  # default:256
        self.use_swiglu = cfg.use_swiglu        # default:True
        self.transformer_ffn_depthwise_conv = cfg.transformer_ffn_depthwise_conv  # default:False
        self.tab_d1 = cfg.tab_d1    # default:16
        self.tab_d2 = cfg.tab_d2    # default:16
        self.tab_c_z = cfg.tab_c_z  # default:32
        self.tab_num_templates = cfg.tab_num_templates  # default:32
        self.tab_num_freqs = cfg.tab_num_freqs    # default:8
        self.tab_num_blocks = cfg.tab_num_blocks  # default:3
        self.tab_dilation = cfg.tab_dilation      # default:3
        self.opt_name = cfg.optimizer

        self.construct_layers()

        num_total_blocks = len(self.residual_tower)
        xavier_init = self.mode != "fixup"
        with torch.no_grad():
            if self.use_tab:  # default:False
                self.tab_module.initialize()
            self.input_conv.initialize(scale=1.0, xavier_init=xavier_init)
            if self.mode == "fixup":  # default:"renorm"
                fixup_scale = 1.0 / math.sqrt(num_total_blocks)
                se_fixup_scale = math.pow(num_total_blocks, -1.0 / (2 * 4 - 2))
                for block in self.residual_tower:
                    block.initialize(fixup_scale=fixup_scale,
                        se_fixup_scale=se_fixup_scale, xavier_init=xavier_init)
            else:  # default:"renorm"
                fixup_scale = 1.0
                for block in self.residual_tower:
                    block.initialize(fixup_scale=fixup_scale,
                        se_fixup_scale=fixup_scale, xavier_init=xavier_init)

            self.policy_conv.initialize(scale=0.8, xavier_init=xavier_init)
            if self.policy_head_type["Type"] == "RepLK":  # default:"Normal"
                self.policy_depthwise_conv.initialize(scale=1.0, xavier_init=xavier_init)
                self.policy_pointwise_conv.initialize(scale=1.0, xavier_init=xavier_init)
            self.policy_intermediate_fc.initialize(scale=0.6, xavier_init=xavier_init)
            self.pol_misc.initialize(scale=0.3, xavier_init=xavier_init)
            self.pol_misc_pass_fc.initialize(scale=0.3, xavier_init=xavier_init)
            self.value_conv.initialize(scale=1.0, xavier_init=xavier_init)
            self.value_intermediate_fc.initialize(scale=1.0, xavier_init=xavier_init)
            self.ownership_conv.initialize(scale=0.2, xavier_init=xavier_init)
            self.value_misc_fc.initialize(scale=0.2, xavier_init=xavier_init)

    def create_policy_head(self):
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,      # default:None
            out_channels=self.policy_head_channels,  # default:None
            kernel_size=1,
            use_gamma=False,
            mode=self.mode,                          # default:"renorm"
            placement="after_block",
            activation=self.activation,              # default:"relu"
            collector=self.layers_collector
        )
        if self.policy_head_type["Type"] == "Normal":  # default:"Normal"
            pass
        elif self.policy_head_type["Type"] == "RepLK":  # default:"Normal"
            dw_kernel_size = max(self.policy_head_type.get("KernelSize", 7), 7)
            self.policy_depthwise_conv = DepthwiseConvBlock(
                channels=self.policy_head_channels,      # default:None
                kernel_size=dw_kernel_size,
                use_gamma=False,
                mode=self.mode,                          # default:"renorm"
                placement="after_block",
                activation=self.activation,              # default:"relu"
                collector=self.layers_collector
            )
            self.policy_pointwise_conv = ConvBlock(
                in_channels=self.policy_head_channels,   # default:None
                out_channels=self.policy_head_channels,  # default:None
                kernel_size=1,
                use_gamma=True,
                mode=self.mode,                          # default:"renorm"
                placement="after_block",
                activation=self.activation,              # default:"relu"
                collector=self.layers_collector
            )
        else:
            raise Exception("Invalid policy head type.")

        self.policy_intermediate_fc = FullyConnect(
            in_size=self.policy_head_channels * 3,   # default:None
            out_size=self.policy_head_channels,      # default:None
            activation=self.activation,              # default:"relu"
            collector=self.layers_collector
        )
        self.pol_misc = Convolve(
            in_channels=self.policy_head_channels,   # default:None
            out_channels=self.policy_outs,           # fix:5
            kernel_size=1,
            activation="identity",
            collector=self.layers_collector
        )
        self.pol_misc_pass_fc = FullyConnect(
            in_size=self.policy_head_channels,       # default:None
            out_size=self.policy_outs,               # fix:5
            activation="identity",
            collector=self.layers_collector
        )

    def create_value_head(self):
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,      # default:None
            out_channels=self.value_head_channels,   # default:None
            kernel_size=1,
            use_gamma=False,
            mode=self.mode,                          # default:"renorm"
            placement="after_block",
            activation=self.activation,              # default:"relu"
            collector=self.layers_collector
        )
        self.value_intermediate_fc = FullyConnect(
            in_size=self.value_head_channels * 3,    # default:None
            out_size=self.value_head_channels * 3,   # default:None
            activation=self.activation,              # default:"relu"
            collector=self.layers_collector
        )
        self.ownership_conv = Convolve(
            in_channels=self.value_head_channels,    # default:None
            out_channels=1,
            kernel_size=1,
            activation="identity",
            collector=self.layers_collector
        )
        self.value_misc_fc = FullyConnect(
            in_size=self.value_head_channels * 3,    # default:None
            out_size=self.value_misc,                # fix:15
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
        channels = self.residual_channels  # default:None
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
            elif component in ["TransformerBlock", "NestedBottleneckTransformerBlock"]:
                self.is_pre_act = True  # used Transformer
                blockargs["attention_qk_norm"] = self.attention_qk_norm  # default:False
                blockargs["transformer_heads"] = self.transformer_heads  # default:3
                blockargs["transformer_kv_heads"] = self.transformer_kv_heads  # default:3
                blockargs["attention_query_head_dim"] = self.attention_query_head_dim  # default:32
                blockargs["attention_value_head_dim"] = self.attention_value_head_dim  # default:32
                blockargs["learned_rope_cast_to_input_dtype"] = self.learned_rope_cast_to_input_dtype  # default:False
                blockargs["transformer_ffn_channels"] = self.transformer_ffn_channels  # default:256
                blockargs["use_tab"] = self.use_tab  # default:False
                blockargs["tab_d1"] = self.tab_d1    # default:16
                blockargs["tab_d2"] = self.tab_d2    # default:16
                blockargs["tab_c_z"] = self.tab_c_z  # default:None
                blockargs["tab_num_templates"] = self.tab_num_templates  # default:None
                blockargs["tab_num_freqs"] = self.tab_num_freqs  # default:None
                blockargs["tab_num_blocks"] = self.tab_num_blocks  # default:None
                blockargs["tab_dilation"] = self.tab_dilation  # default:None
                blockargs["use_swiglu"] = self.use_swiglu  # default:True
                blockargs["transformer_ffn_depthwise_conv"] = self.transformer_ffn_depthwise_conv  # default:False
                if component == "TransformerBlock":
                    block = TransformerAttentionBlock
                else:
                    blockargs["bottleneck_channels"] = channels // 2
                    assert channels % 2 == 0, ""
                    block = NestedBottleneckTransformerBlock
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
            elif key == "AttentionQKNorm":
                blockargs["attention_qk_norm"] = value
            elif key == "TransformerHeads":
                blockargs["transformer_heads"] = value
            elif key == "TransformerKVHheads":
                blockargs["transformer_kv_heads"] = value
            elif key == "AttentionQueryHeadDim":
                blockargs["attention_query_head_dim"] = value
            elif key == "AttentionValueHeadDim":
                blockargs["attention_value_head_dim"] = value
            elif key == "LearnedRoPECastToInputDtype":
                blockargs["learned_rope_cast_to_input_dtype"] = value
            elif key == "TransformerFFNChannels":
                blockargs["transformer_ffn_channels"] = value
            elif key == "UseSwiGLU":
                blockargs["use_swiglu"] = value
            elif key == "TransformerFFNDepthwiseConv":
                blockargs["transformer_ffn_depthwise_conv"] = value
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
                "activation" : self.activation,            # default:"relu"
                "mode" : self.mode,                        # default:"renorm"
                "is_pre_act" : self.is_pre_act,            # default:False
                "pos_len" : self.pos_len,                  # default:19
                "collector" : self.layers_collector
            }
            block, channels, blockargs = self.parse_blocksetting(blocksetting, blockargs)
            self.residual_tower.append(block(channels=channels, **blockargs))

    def construct_layers(self):
        self.global_pool = GlobalPool(is_value_head=False)
        self.global_pool_val = GlobalPool(is_value_head=True)

        for block in self.stack:
            last_is_tran = False
            components = list()
            if type(block) == str:
                components = block.strip().split('-')
            else:
                components = block["Block"].strip().split('-')
            for component in components:
                if component in ["TransformerBlock", "NestedBottleneckTransformerBlock"]:
                    self.is_pre_act = True  # used Transformer
                    last_is_tran = True

        if self.is_pre_act:
            self.input_conv = Convolve(
                in_channels=self.input_channels,  # default:43
                out_channels=self.residual_channels,  # default:None
                kernel_size=3,
                activation="identity",
                bias=False,
                collector=self.layers_collector
            )
        else:
            self.input_conv = ConvBlock(
                in_channels=self.input_channels,  # default:43
                out_channels=self.residual_channels,   # default:None
                kernel_size=3,
                use_gamma=True,
                mode=self.mode,                        # default:"renorm"
                placement="before_block",
                activation=self.activation,            # default::"relu"
                collector=self.layers_collector
            )

        self.create_residual_tower()

        # Create shared TAB template MLP if any block uses TAB
        if self.use_tab:  # default:False
            self.tab_module = TABModule(
                trunk_channels=self.residual_channels,     # default:None
                tab_c_z=self.tab_c_z,                      # default:32
                tab_num_templates=self.tab_num_templates,  # default:32
                tab_num_freqs=self.tab_num_freqs,          # default:8
                tab_num_blocks=self.tab_num_blocks,        # default:3
                tab_dilation=self.tab_dilation,            # default:3
                activation=self.activation,                # default:"relu"
                pos_len=self.pos_len                       # default:19
            )
        else:
            self.tab_module = None

        # Training-time attention logit penalty. Not part of the model config: set externally
        # (e.g. by train.py from -attn-logit-penalty-cap) before any torch.compile wrapping.
        # When set to a float cap, each forward stores:
        #   self.attn_logit_penalty_per_sample: (B',) sum over attention layers of
        #     mean over heads of relu(logit_upper_bound - cap), differentiable (linear hinge),
        #     over the first B' = ceil(B * attn_logit_penalty_batch_frac) batch items.
        #   self.attn_logit_ub_batch_max: 0-dim detached max of the bound over all layers/samples/heads.
        # See TransformerAttentionBlock for the bound definition (includes off-board positions).
        # attn_logit_penalty_batch_frac < 1 computes the penalty on a fixed slice of the batch,
        # cutting its (mostly-backward) cost proportionally at the price of gradient variance,
        # which the slow hinge dynamics tolerate. The compiled/DDP graph stays static.
        self.attn_logit_penalty_cap = self.cfg.attn_logit_penalty_cap
        self.attn_logit_penalty_batch_frac = self.cfg.attn_logit_penalty_batch_frac

        if self.is_pre_act:  # default:False
            if last_is_tran and self.mode == "fixup" and self.final_block_cgroup_size is not None:
                self.final_block = RMSNormMask(
                    c_in=self.residual_channels,  # default:None
                    cgroup_size=self.final_block_cgroup_size  # default:None
                )
            else:
                self.final_block = BatchNorm2d(
                    num_features=self.residual_channels,  # default:None
                    use_gamma=False,
                    mode=self.mode                        # default:"renorm"
                )
            self.final_act = activation_func(self.activation, inplace=True)  # default:"relu"
        else:
            self.final_block = CustomIdentity()
            self.final_act = nn.Identity()

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
        mask = planes[:, (self.input_channels-1):self.input_channels , :, :].contiguous()
        mask_sum_hw = torch.sum(mask, dim=(1,2,3))
        mask_sum_hw_sqrt = torch.sqrt(mask_sum_hw)
        mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)
        mask_sum_hw_transformer = torch.sum(mask, dim=(2,3), keepdim=True)
        mask_sum_transformer = torch.sum(mask)

        # input layer
        x = self.input_conv(planes, mask)

        # Compute shared block data
        block_shared_data = {}
        if self.tab_module is not None:  # default:None
            tab_keys, tab_queries = self.tab_module(x, mask)
            block_shared_data[TAB_KQ] = TABKeyQueryData(keys=tab_keys, queries=tab_queries)
        if self.attn_logit_penalty_cap is not None:
            pen_batch_items = max(1, int(math.ceil(mask.shape[0] * self.attn_logit_penalty_batch_frac)))
            block_shared_data[ATTN_LOGIT_UB] = {"num_batch_items": pen_batch_items, "ubs": []}

        # residual tower
        for i, block in enumerate(self.residual_tower):
            if isinstance(block, TransformerAttentionBlock) or isinstance(block, NestedBottleneckTransformerBlock):
                x = block(x,
                    mask=mask,
                    mask_sum_hw=mask_sum_hw_transformer,
                    mask_sum=mask_sum_transformer,
                    block_shared_data=block_shared_data
                )
            else:
                x = block(x, mask_buffers)

        if self.attn_logit_penalty_cap is not None:
            ub_list = block_shared_data[ATTN_LOGIT_UB]["ubs"]
            assert len(ub_list) > 0, "attn_logit_penalty_cap set but model has no attention layers"
            ubs = torch.stack(ub_list)  # (num_attn_layers, B', H)
            excess = torch.nn.functional.relu(ubs - self.attn_logit_penalty_cap)
            # Linear hinge: constant-magnitude pull on offending heads regardless of how far
            # above the cap they currently are, gentler than a squared hinge for large excesses.
            self.attn_logit_penalty_per_sample = excess.mean(dim=2).sum(dim=0)  # (B',)
            self.attn_logit_ub_batch_max = ubs.detach().amax()

        # Use original mask for final norm and heads (NCHW format)
        if isinstance(self.final_block, RMSNormMask):
            x = self.final_block(x, mask, mask_sum_hw_transformer, mask_sum_transformer)
        else:
            x = self.final_block(x, mask)
        x = self.final_act(x)

        # Use fp32 for output heads to handle potentially large values
        with autocast("cuda", enabled=False):
            x = x.float()
            mask = mask.float()
            mask_sum_hw = mask_sum_hw.float() if isinstance(mask_sum_hw, torch.Tensor) else mask_sum_hw
            mask_sum_hw_sqrt = mask_sum_hw_sqrt.float() if isinstance(mask_sum_hw_sqrt, torch.Tensor) else mask_sum_hw_sqrt
            mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)
            # policy head
            pol = self.policy_conv(x, mask)
            if self.policy_head_type["Type"] == "RepLK":  # default:"Normal"
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
        if self.is_pre_act:
            info += "Pre Activation : true\n"
        else:
            info += "Pre Activation : false\n"
        info += "Optimizer: {optimizer}\n".format(optimizer=self.opt_name)
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
        reg_dict["normal_tab"] = []
        reg_dict["tab_module"] = []

        self.input_conv.add_reg_dict(reg_dict, placement="before_block")
        for block in self.residual_tower:
            block.add_reg_dict(reg_dict)
        if self.tab_module is not None:        # default:None
            self.tab_module.add_reg_dict(reg_dict)
        if self.is_pre_act:  # default:False
            self.final_block.add_reg_dict(reg_dict, placement="after_block")
        self.policy_conv.add_reg_dict(reg_dict, placement="after_block")
        if self.policy_head_type["Type"] == "RepLK":  # default:"Normal"
            self.policy_depthwise_conv.add_reg_dict(reg_dict, placement="after_block")
            self.policy_pointwise_conv.add_reg_dict(reg_dict, placement="after_block")
        self.policy_intermediate_fc.add_reg_dict(reg_dict, placement="after_block")
        self.pol_misc.add_reg_dict(reg_dict, placement="after_block")
        self.pol_misc_pass_fc.add_reg_dict(reg_dict, placement="after_block")
        self.value_conv.add_reg_dict(reg_dict, placement="after_block")
        self.value_intermediate_fc.add_reg_dict(reg_dict, placement="after_block")
        self.ownership_conv.add_reg_dict(reg_dict, placement="after_block")
        self.value_misc_fc.add_reg_dict(reg_dict, placement="after_block")
