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
        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.linear.weight, gain=compute_gain(self.activation))
        nn.init.zeros_(self.linear.bias)

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

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
        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.conv.weight, gain=compute_gain(self.activation))
        if self.bias:
            nn.init.zeros_(self.conv.bias)

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

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

        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.conv.weight, gain=compute_gain(self.activation))

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

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

        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.conv.weight, gain=compute_gain(self.activation))
        nn.init.xavier_normal_(
            self.rep3x3.weight, gain=compute_gain(self.activation))

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
        if not self.is_pre_act:
            out = out + x
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
        if not self.is_pre_act:
            out = out + x
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
        if not self.is_pre_act:
            out = out + x
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
            if not self.is_pre_act:
                out = out + x
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

class RMSNormMask(nn.Module):
    """RMSNorm applied per spatial position across channels, with masking for off-board positions.
    If spatial=True, computes RMS across both channels and spatial positions (masked), producing
    one scalar RMS per sample instead of per position.
    If spatial=True and cgroup_size is not None, breaks channels into groups of the given size
    and normalizes within each group across channels_in_group x H x W (like group norm but RMS only,
    no mean centering).
    """
    def __init__(self, c_in, spatial, cgroup_size):
        super(RMSNormMask, self).__init__()
        self.c_in = c_in
        self.spatial = spatial
        self.cgroup_size = cgroup_size
        self.eps = 1e-6
        if cgroup_size is not None:
            assert spatial, "cgroup_size requires spatial=True"
            assert c_in % cgroup_size == 0, f"c_in ({c_in}) must be divisible by cgroup_size ({cgroup_size})"
            self.num_groups = c_in // cgroup_size
        if not spatial:
            self.norm = CustomRMSNorm(c_in, eps=self.eps)
        else:
            self.norm = None
            self.gamma = torch.nn.Parameter(torch.ones(c_in))
        self.beta = torch.nn.Parameter(torch.zeros(c_in))

    def add_reg_dict(self, reg_dict, placement):
        if self.norm is not None:
            reg_dict["output"].append(self.norm.weight)
        else:
            reg_dict["output"].append(self.gamma)
        reg_dict["output"].append(self.beta)

    def forward(self, x, mask, mask_sum_hw, mask_sum):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        if not self.spatial:
            # NCHW -> NHWC for RMSNorm across channels, then back
            out = x.permute(0, 2, 3, 1)
            out = self.norm(out)
            out = out.permute(0, 3, 1, 2)
            return (out + self.beta.view(1, -1, 1, 1)) * mask
        else:
            if self.cgroup_size is not None:
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
                mean_sq = torch.sum(
                    x * x * mask,
                    dim=(1, 2, 3),
                    keepdim=True) / (self.c_in * mask_sum_hw + self.eps)
                rms = torch.sqrt(mean_sq + self.eps)
                out = x / rms
            return (out * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)) * mask

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

def apply_learnable_rotary_emb(xq, xk, cos_q, sin_q, cos_k, sin_k):
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
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.reshape(B, S, H, D).type_as(x)

    return _rotate(xq, cos_q, sin_q), _rotate(xk, cos_k, sin_k)

def compute_gab_fourier_features(dr, dc, freqs):
    """Compute Fourier features for relative (dr, dc) offsets.
    dr: (...) float tensor of row offsets
    dc: (...) float tensor of col offsets
    freqs: (num_freqs,) tensor of learnable frequencies
    Returns: (..., 8*num_freqs)
    """
    features = []
    dr_plus_dc = dr + dc
    dr_minus_dc = dr - dc
    for f in freqs:
        features.append(torch.sin(f * dr).unsqueeze(-1))
        features.append(torch.cos(f * dr).unsqueeze(-1))
        features.append(torch.sin(f * dc).unsqueeze(-1))
        features.append(torch.cos(f * dc).unsqueeze(-1))
        features.append(torch.sin(f * dr_plus_dc).unsqueeze(-1))
        features.append(torch.cos(f * dr_plus_dc).unsqueeze(-1))
        features.append(torch.sin(f * dr_minus_dc).unsqueeze(-1))
        features.append(torch.cos(f * dr_minus_dc).unsqueeze(-1))
    return torch.cat(features, dim=-1)

GAB_TEMPLATES = "gab_templates"
TAB_KQ = "tab_kq"

@dataclass
class GABTemplateData:
    """Precomputed GAB template values, shared across all blocks in a forward pass.
    By convention, templates are pre-scaled by 1/sqrt of the appropriate quantity
    so that a weighted combination does not need further scaling.
    """
    templates: torch.Tensor  # (S, S, T) template values for all position pairs

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

class GABTemplateMLP(torch.nn.Module):
    """Shared module that maps relative (dr, dc) offsets to T template values.
    Computed once and shared across all GAB-enabled transformer blocks.
    """
    def __init__(self, gab_num_templates, gab_num_fourier_features, gab_mlp_hidden, pos_len, activation):
        # Let F = gab_num_fourier_features, H = gab_mlp_hidden, T = gab_num_templates
        # S = pos_len * pos_len (max spatial positions)
        super().__init__()
        self.gab_num_templates = gab_num_templates
        self.activation = activation
        self.act = activation_func(activation)
        assert gab_num_fourier_features >= 2, "gab_num_fourier_features must be >= 2"
        fourier_input_dim = 8 * gab_num_fourier_features  # 8*F

        # Geometric initialization from 1 rad/square to 1/50 rad/square
        init_freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1.0 / 50.0), gab_num_fourier_features))
        self.gab_freqs = torch.nn.Parameter(init_freqs)  # (F,)

        self.linear1 = torch.nn.Linear(fourier_input_dim, gab_mlp_hidden)  # (8*F) -> (H)
        self.linear2 = torch.nn.Linear(gab_mlp_hidden, gab_num_templates)  # (H) -> (T)

        S = pos_len * pos_len
        s_idx = torch.arange(S)
        s_r, s_c = s_idx // pos_len, s_idx % pos_len
        offset_dr = (s_r.unsqueeze(1) - s_r.unsqueeze(0)).float()  # (S, S)
        offset_dc = (s_c.unsqueeze(1) - s_c.unsqueeze(0)).float()  # (S, S)
        self.register_buffer("offset_dr", offset_dr, persistent=False)
        self.register_buffer("offset_dc", offset_dc, persistent=False)

    def forward(self, seq_len):
        """Compute templates for all position pairs up to seq_len.
        Returns: (seq_len, seq_len, T)
        """
        dr = self.offset_dr[:seq_len, :seq_len]              # (S, S)
        dc = self.offset_dc[:seq_len, :seq_len]              # (S, S)
        fourier_feats = compute_gab_fourier_features(dr, dc, self.gab_freqs)  # (S, S, 8*F)
        x = self.act(self.linear1(fourier_feats))            # (S, S, H)
        x = self.linear2(x)                                  # (S, S, T)
        scale = 1.0 / math.sqrt(self.gab_num_templates)
        return x * scale

    def initialize(self):
        init_weights(self.linear1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, "identity", scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["noreg"].append(self.gab_freqs)
        reg_dict["gab_mlp"].append(self.linear1.weight)
        reg_dict["noreg"].append(self.linear1.bias)
        reg_dict["gab_mlp"].append(self.linear2.weight)
        reg_dict["noreg"].append(self.linear2.bias)

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

class FrequencyMixingTABBlock(torch.nn.Module):
    """One residual block for frequency-mixing TAB.

    Depthwise convs are per-frequency in the rotated frame (equivariant).
    1x1 convs mix freely across all 2*c_z channels in the unrotated frame (equivariant).
    """
    def __init__(self, c_z, activation, dilation):
        super().__init__()
        self.c_z = c_z
        self.act1 = activation_func(activation)
        self.dw_conv1 = ComplexDepthwiseConv2d(c_z, kernel_size=3, dilation=dilation)
        self.mix1 = torch.nn.Conv2d(2 * c_z, 2 * c_z, kernel_size=1, bias=False)
        self.act2 = activation_func(activation)
        self.dw_conv2 = ComplexDepthwiseConv2d(c_z, kernel_size=3, dilation=1)
        self.mix2 = torch.nn.Conv2d(2 * c_z, 2 * c_z, kernel_size=1, bias=False)

    def forward(self, z, cos_a, sin_a, block_idx):
        """
        z: (N, 2, c_z, H, W) - [real, imag] x c_z frequency channels
        cos_a, sin_a: (1, 1, c_z, H, W) - per-frequency angles, broadcastable
        block_idx: int, for variance normalization
        """
        N, _, c_z, H, W = z.shape
        zskip = z

        # Normalize variance (same logic as TABEquivariantBlock)
        z = z * (1.0 / math.sqrt(block_idx + 1))

        z = self.act1(z)

        # Depthwise conv in rotated frame
        z = tab_rotate(z, cos_a, sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.dw_conv1(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        z = tab_rotate(z, cos_a, -sin_a)

        # 1x1 channel mixing in unrotated frame
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.mix1(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)

        z = self.act2(z)

        # Depthwise conv in rotated frame
        z = tab_rotate(z, cos_a, sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.dw_conv2(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        z = tab_rotate(z, cos_a, -sin_a)

        # 1x1 channel mixing in unrotated frame
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.mix2(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)

        z = z + zskip
        return z

    def initialize(self, activation):
        self.dw_conv1.initialize(activation, scale=1.0)
        self.dw_conv2.initialize(activation, scale=1.0)
        init_weights(self.mix1.weight, activation, scale=1.0)
        init_weights(self.mix2.weight, "identity", scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["tab_module"].append(self.dw_conv1.real_kernel)
        reg_dict["tab_module"].append(self.dw_conv1.imag_kernel)
        reg_dict["tab_module"].append(self.mix1.weight)
        reg_dict["tab_module"].append(self.dw_conv2.real_kernel)
        reg_dict["tab_module"].append(self.dw_conv2.imag_kernel)
        reg_dict["tab_module"].append(self.mix2.weight)

class FrequencyMixingTABModule(torch.nn.Module):
    """TAB module with frequency mixing.

    Unlike TABModule where each frequency has an independent c_z-channel convnet,
    here c_z IS the number of frequencies. Frequencies interact via pointwise (1x1)
    convs in the unrotated frame, spatial mixing happens via depthwise convs in the
    rotated frame. This preserves translational equivariance.
    """
    def __init__(self, trunk_channels, tab_c_z, tab_num_templates, tab_num_blocks, tab_dilation, activation, pos_len):
        super().__init__()
        self.tab_c_z = tab_c_z  # = number of frequencies
        self.tab_num_templates = tab_num_templates
        self.tab_num_blocks = tab_num_blocks
        self.activation = activation

        # 1x1 conv to project trunk channels -> 2*c_z (interpreted as c_z complex values)
        self.input_proj = torch.nn.Conv2d(trunk_channels, 2 * tab_c_z, kernel_size=1, bias=False)

        # Learnable 2D RoPE frequencies: (c_z, 2) for (omega_X, omega_Y)
        # Geometric initialization from 1 rad/square to 1/50 rad/square
        log_lo = math.log(1.0 / 50.0)
        log_hi = math.log(1.0)
        init_freqs = torch.exp(torch.empty(tab_c_z, 2).uniform_(log_lo, log_hi))
        init_freqs = init_freqs * (torch.randint(0, 2, (tab_c_z, 2)) * 2 - 1).float()
        self.rope_freqs = torch.nn.Parameter(init_freqs)

        self.blocks = torch.nn.ModuleList()
        for _ in range(tab_num_blocks):
            self.blocks.append(FrequencyMixingTABBlock(tab_c_z, activation, tab_dilation))

        self.final_act = activation_func(activation)
        self.key_proj = torch.nn.Conv2d(2 * tab_c_z, 2 * tab_c_z, kernel_size=1, bias=False)
        self.query_proj = torch.nn.Conv2d(2 * tab_c_z, 2 * tab_c_z * tab_num_templates, kernel_size=1, bias=False)

    def forward(self, x, mask):
        """
        x: (N, C, H, W) trunk output
        mask: (N, 1, H, W) or None
        Returns: (keys, queries) with keys (N, 2*c_z, 1, S) and queries (N, 2*c_z, T, S), pre-scaled
        """
        N, C, H, W = x.shape
        S = H * W
        c_z = self.tab_c_z
        T = self.tab_num_templates

        z = self.input_proj(x)

        # Apply mask to zero off-board positions
        if mask is not None:
            z = z * mask

        z = z.view(N, 2, c_z, H, W)

        # Precompute angles
        gy = torch.arange(H, device=x.device, dtype=x.dtype)
        gx = torch.arange(W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)
        angles = self.rope_freqs[:, 0:1].unsqueeze(-1) * grid_x.unsqueeze(0) + \
                 self.rope_freqs[:, 1:2].unsqueeze(-1) * grid_y.unsqueeze(0)  # (c_z, H, W)
        # Shape (1, 1, c_z, H, W) to broadcast with (N, 2, c_z, H, W) in tab_rotate
        cos_a = torch.cos(angles).view(1, 1, c_z, H, W)
        sin_a = torch.sin(angles).view(1, 1, c_z, H, W)

        block_idx = 0
        for block in self.blocks:
            z = block(z, cos_a, sin_a, block_idx)
            block_idx += 1

        # Normalize variance
        z = z * (1.0 / math.sqrt(block_idx + 1))

        # Final: activate, project keys/queries in unrotated space, then rotate
        z = self.final_act(z)
        z_flat = z.reshape(N, 2 * c_z, H, W)

        # cos/sin for final rotation: (1, c_z, 1, 1, H, W) -> tile across N samples
        # After folding N*c_z into batch, need (N*c_z, 1, 1, H, W)
        cos_a_out = cos_a.view(1, c_z, 1, 1, H, W).expand(N, c_z, 1, 1, H, W).reshape(N * c_z, 1, 1, H, W)
        sin_a_out = sin_a.view(1, c_z, 1, 1, H, W).expand(N, c_z, 1, 1, H, W).reshape(N * c_z, 1, 1, H, W)

        # Keys: mix in unrotated space first, reshape per-frequency, then rotate
        keys = self.key_proj(z_flat)  # (N, 2*c_z, H, W)
        keys = keys.view(N * c_z, 2, 1, H, W)
        keys = tab_rotate(keys, cos_a_out, sin_a_out)

        # Queries: mix in unrotated space first, reshape per-frequency, then rotate
        queries = self.query_proj(z_flat)  # (N, 2*c_z*T, H, W)
        queries = queries.view(N * c_z, 2, T, H, W)
        queries = tab_rotate(queries, cos_a_out, sin_a_out)

        keys = keys.reshape(N, 2 * c_z, 1, S)
        queries = queries.reshape(N, 2 * c_z, T, S)
        return keys / math.sqrt(c_z), queries / math.sqrt(T)

    def initialize(self):
        init_weights(self.input_proj.weight, self.activation, scale=1.0)
        for block in self.blocks:
            block.initialize(self.activation)
        init_weights(self.key_proj.weight, self.activation, scale=1.0)
        init_weights(self.query_proj.weight, self.activation, scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["tab_module"].append(self.input_proj.weight)
        reg_dict["noreg"].append(self.rope_freqs)
        for block in self.blocks:
            block.add_reg_dict(reg_dict)
        reg_dict["tab_module"].append(self.key_proj.weight)
        reg_dict["tab_module"].append(self.query_proj.weight)

class ComplexDepthwiseConv2d(torch.nn.Module):
    """Depthwise 2D complex convolution.

    Each of the c channels gets its own K x K complex kernel (no cross-channel mixing).
    Stores real_kernel and imag_kernel of shape (c, 1, K, K).

    Computes complex multiplication via two separate depthwise convolutions (groups=c):
        out_real = real_kernel * in_real - imag_kernel * in_imag
        out_imag = imag_kernel * in_real + real_kernel * in_imag

    Input: (*, 2*c, H, W) where channels are [re_0..re_{c-1}, im_0..im_{c-1}].
    Output: same layout.
    """
    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.real_kernel = torch.nn.Parameter(torch.empty(c, 1, kernel_size, kernel_size))
        self.imag_kernel = torch.nn.Parameter(torch.empty(c, 1, kernel_size, kernel_size))

    def forward(self, x):
        # x: (*, 2*c, H, W) laid out as [re_0, ..., re_{c-1}, im_0, ..., im_{c-1}]
        padding = self.dilation * (self.kernel_size // 2)
        x_re = x[..., :self.c, :, :]   # (*, c, H, W)
        x_im = x[..., self.c:, :, :]   # (*, c, H, W)

        # Conv 1: convolve [re; im] with [rk; ik], fully depthwise (groups=2c)
        x_ri = torch.cat([x_re, x_im], dim=-3)              # (*, 2c, H, W)
        k_ri = torch.cat([self.real_kernel, self.imag_kernel], dim=0)  # (2c, 1, K, K)
        conv1 = torch.nn.functional.conv2d(x_ri, k_ri, padding=padding, dilation=self.dilation, groups=2 * self.c)
        # conv1: (*, 2c, H, W) = [rk*re; ik*im]

        # Conv 2: convolve [re; im] with [-ik; rk], fully depthwise (groups=2c)
        k_neg_ir = torch.cat([-self.imag_kernel, self.real_kernel], dim=0)  # (2c, 1, K, K)
        conv2 = torch.nn.functional.conv2d(
            x_ri,
            k_neg_ir,
            padding=padding,
            dilation=self.dilation,
            groups=2 * self.c
        )
        # conv2: (*, 2c, H, W) = [-ik*re; rk*im]

        # out_re = rk*re - ik*im = conv1[:c] - conv1[c:]
        # out_im = ik*re + rk*im = -conv2[:c] + conv2[c:]
        out_re = conv1[..., :self.c, :, :] - conv1[..., self.c:, :, :]
        out_im = conv2[..., self.c:, :, :] - conv2[..., :self.c, :, :]
        return torch.cat([out_re, out_im], dim=-3)

    def initialize(self, activation, scale=1.0):
        init_weights(self.real_kernel, activation, scale=scale / math.sqrt(2.0))
        init_weights(self.imag_kernel, activation, scale=scale / math.sqrt(2.0))

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
        self.pos_len = kwargs.get("pos_len", 9)
        self.positional_encoding = kwargs.get("positional_encoding", "RoPE")
        self.use_rope = False
        self.use_gab = False
        self.use_tab = False
        self.use_tab_freq_mix = False
        if self.positional_encoding in ["RoPE", "RoPE+GAB", "RoPE+TAB", "RoPE+TAB+FreqMix"]:
            self.use_rope = True
        if self.positional_encoding in ["GAB", "RoPE+GAB"]:
            self.use_gab = True
        if self.positional_encoding in ["TAB", "RoPE+TAB"]:
            self.use_tab = True
        if self.positional_encoding in ["TAB+FreqMix", "RoPE+TAB+FreqMix"]:
            self.use_tab_freq_mix = True
        self.learnable_rope = kwargs.get("learnable_rope", False) if self.use_rope else False
        self.use_qk_norm = kwargs.get("attention_qk_norm", False)
        self.num_heads = kwargs.get("transformer_heads", 3)
        self.num_kv_heads = kwargs.get("transformer_kv_heads", self.num_heads)
        # Compute how many query heads each KV head serves (group size)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_head_dim = kwargs.get("attention_query_head_dim", channels // self.num_heads)
        self.v_head_dim = kwargs.get("attention_value_head_dim", channels // self.num_heads)

        if self.use_rope:
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

        if self.use_rope:
            if self.learnable_rope:
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
                self.cos_cached = None
                self.sin_cached = None
            else:
                self.rope_theta = kwargs.get("rope_theta", 100.0)
                assert self.rope_theta > self.pos_len * 2.0, \
                    f"theta={self.rope_theta} of RoPE may be too small for pos_len={self.pos_len}"
                cos_cached, sin_cached = precompute_freqs_cos_sin_2d(self.q_head_dim, self.pos_len, self.rope_theta)
                self.register_buffer("cos_cached", cos_cached, persistent=False)
                self.register_buffer("sin_cached", sin_cached, persistent=False)
        else:
            self.cos_cached = None
            self.sin_cached = None

        if self.use_gab or self.use_tab or self.use_tab_freq_mix:
            gab_d1 = kwargs.get("gab_d1", 16)
            gab_d2 = kwargs.get("gab_d2", 16)
            self.gab_num_templates = kwargs.get("gab_num_templates", 32) if self.use_gab else 0
            self.tab_num_templates = kwargs.get("tab_num_templates", 32) if self.use_tab or self.use_tab_freq_mix else 0
            # Per-head weights: one per GAB template, one per TAB template.
            # TAB weights are per-template (shared across 2*F real/imag freq channels).
            self.total_num_weights = self.gab_num_templates + self.tab_num_templates
            self.gab_proj1 = torch.nn.Linear(channels, gab_d1, bias=False)
            self.gab_proj2 = torch.nn.Linear(gab_d1, gab_d2, bias=False)
            self.gab_norm1 = CustomRMSNorm(gab_d2, eps=1e-6)
            self.gab_proj3 = torch.nn.Linear(gab_d2, self.num_heads * self.total_num_weights, bias=False)
            self.gab_norm2 = CustomRMSNorm(self.num_heads * self.total_num_weights, eps=1e-6)
            self.gab_act1 = activation_func(self.activation, inplace=False)
            self.gab_act2 = activation_func(self.activation, inplace=False)

        self.norm1 =  CustomRMSNorm(channels, eps=1e-6)

    def add_reg_dict(self, reg_dict):
        for name, param in self.named_parameters():
            if "norm" in name or "cached" in name:
                reg_dict["noreg"].append(param)
                continue
            if "weight" in name:
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                    reg_dict["normal_attn"].append(param)
                elif "gab_proj" in name:
                    reg_dict["normal_gab"].append(param)
                else:
                    reg_dict["normal"].append(param)
            else:
                reg_dict["noreg"].append(param)

    def initialize(self, fixup_scale, xavier_init):
        pass

    def _compute_gab_bias(self, x_norm, mask, mask_sum_hw, block_shared_data):
        """Compute attention bias from GAB templates and/or TAB factored keys/queries.
        x_norm: (B, S, C) normalized token representations
        mask: (N, 1, H, W) or None
        mask_sum_hw: (N, 1, 1, 1) or None
        block_shared_data: dict with precomputed template/key-query data
        Returns: (template_bias, extra_kq) where
            template_bias: (B, H, S, S) materialized attention bias, or None
            extra_kq: (extra_k, extra_q) to concatenate onto main K/Q, or None
        """
        batch_size, seq_len, _ = x_norm.shape

        # Per-token projection
        y = self.gab_proj1(x_norm) # (B, S, d1)

        # Masked mean pooling over valid positions
        if mask is not None:
            mask_flat = mask.view(batch_size, seq_len, 1)  # (B, S, 1)
            y = y * mask_flat
            pooled = y.sum(dim=1) / mask_sum_hw.view(batch_size, 1)  # (B, d1)
        else:
            pooled = y.mean(dim=1)                       # (B, d1)

        # Compress + activation + norm
        z = self.gab_act1(self.gab_proj2(pooled))         # (B, d2)
        z = self.gab_norm1(z)

        # Generate per-head weights for all bias mechanisms
        z = self.gab_act2(self.gab_proj3(z))              # (B, H*total_num_weights)
        z = self.gab_norm2(z)
        z = z.view(batch_size, self.num_heads, self.total_num_weights)  # (B, H, W_total)

        bias = None
        extra_k_parts = []
        extra_q_parts = []
        idx = 0

        # GAB contribution: input-independent templates (S, S, T_gab)
        if self.use_gab:
            z_gab = z[:, :, idx:idx + self.gab_num_templates]
            idx += self.gab_num_templates
            gab_data = block_shared_data[GAB_TEMPLATES]
            gab_templates = gab_data.templates
            bias = torch.einsum("bhd,std->bhst", z_gab, gab_templates)

        # TAB contribution: mix templates in K/Q space, then append 2*F_tab dims.
        # Instead of keeping T templates separate (which would need 2*F*T extra dims),
        # we contract over templates before the dot product, yielding one mixed
        # key/query per frequency per head - only 2*F_tab extra dims.
        if self.use_tab or self.use_tab_freq_mix:
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

        return bias, extra_kq

    def forward(self, x, mask, mask_sum_hw, mask_sum, block_shared_data=None):
        """
        Parameters:
        x: NCHW (or NC1S when inline registers are active)
        mask: N1HW (or N11S when inline registers are active)
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1)

        x_norm = self.norm1(x_in)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.q_head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.v_head_dim)

        if self.use_rope:
            if self.learnable_rope:
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
                    cos_q = cos_k.unsqueeze(-2).expand(*cos_k.shape[:-1], self.n_rep, cos_k.shape[-1]).reshape(*cos_k.shape[:-2], self.num_heads, -1)
                    sin_q = sin_k.unsqueeze(-2).expand(*sin_k.shape[:-1], self.n_rep, sin_k.shape[-1]).reshape(*sin_k.shape[:-2], self.num_heads, -1)
                else:
                    cos_q = cos_k
                    sin_q = sin_k
                q, k = apply_learnable_rotary_emb(q, k, cos_q, sin_q, cos_k, sin_k)
            else:
                q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)

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

        template_bias = None
        extra_kq = None
        if self.use_gab or self.use_tab or self.use_tab_freq_mix:
            template_bias, extra_kq = self._compute_gab_bias(x_norm, mask, mask_sum_hw, block_shared_data)

        if mask is not None:
            # For inline registers, mask is N11S and already includes register positions
            # (always 1.0), so seq_len already covers them.
            mask_flat = mask.view(batch_size, 1, 1, seq_len)
            attn_mask = torch.zeros_like(mask_flat, dtype=q.dtype)
            attn_mask.masked_fill_(mask_flat == 0, float('-inf'))
        else:
            attn_mask = None

        if template_bias is not None:
            if attn_mask is not None:
                attn_mask = attn_mask + template_bias
            else:
                attn_mask = template_bias

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

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=scale,
        )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        attn_output = self.out_proj(attn_output)
        result = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        return result

class TransformerFFNBlock(torch.nn.Module):
    """Feed-forward half of a transformer block, with its own residual connection.

    Contains: RMSNorm -> FFN (optionally SwiGLU) -> optional depthwise conv.
    Returns residual only; caller is responsible for adding to trunk.
    """
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(TransformerFFNBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.ffn_dim = kwargs.get("transformer_ffn_channels", 256)
        self.use_swiglu = kwargs.get("use_swiglu", True)
        self.use_depthwise_conv = kwargs.get("transformer_ffn_depthwise_conv", False)
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
        self.norm =  CustomRMSNorm(channels, eps=1e-6)

    def add_reg_dict(self, reg_dict):
        for name, param in self.named_parameters():
            if "norm" in name:
                reg_dict["noreg"].append(param)
                continue
            if "weight" in name:
                reg_dict["normal"].append(param)
            else:
                reg_dict["noreg"].append(param)

    def initialize(self, fixup_scale, xavier_init):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum, block_shared_data=None):
        """
        Parameters:
        x: NCHW (or NC1S when inline registers are active)
        mask: N1HW (or N11S when inline registers are active)
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1)

        xn = self.norm(x_in)

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
        x1 = self.ffn_linear2(x1)
        result = x1.permute(0, 2, 1).view(batch_size, channels, height, width)
        return result

class NestedBottleneckTransformerBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(NestedBottleneckTransformerBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.mode = kwargs.get("mode", "renorm")
        self.is_pre_act = kwargs.get("is_pre_act", True)
        self.positional_encoding = kwargs.get("positional_encoding", "RoPE")
        self.pos_len = kwargs.get("pos_len", 9)
        self.learnable_rope = kwargs.get("learnable_rope", False)
        self.rope_theta = kwargs.get("rope_theta", 100.0)
        self.attention_qk_norm = kwargs.get("attention_qk_norm", False)
        self.gab_d1 = kwargs.get("gab_d1", 16)
        self.gab_d2 = kwargs.get("gab_d2", 16)
        self.gab_num_templates = kwargs.get("gab_num_templates", None)
        self.gab_num_fourier_features = kwargs.get("gab_num_fourier_features", None)
        self.gab_mlp_hidden = kwargs.get("gab_mlp_hidden", None)
        self.tab_c_z = kwargs.get("tab_c_z", None)
        self.tab_num_templates = kwargs.get("tab_num_templates", None)
        self.tab_num_freqs = kwargs.get("tab_num_freqs", None)
        self.tab_num_blocks = kwargs.get("tab_num_blocks", None)
        self.tab_dilation = kwargs.get("tab_dilation", None)
        self.transformer_heads = kwargs.get("transformer_heads", 3)
        self.transformer_kv_heads = kwargs.get("transformer_kv_heads", 3)
        self.attention_query_head_dim = kwargs.get("attention_query_head_dim", 32)
        self.attention_value_head_dim = kwargs.get("attention_value_head_dim", 32)
        self.transformer_ffn_channels = kwargs.get("transformer_ffn_channels", 256)
        self.use_swiglu = kwargs.get("use_swiglu", True)
        self.transformer_ffn_depthwise_conv = kwargs.get("transformer_ffn_depthwise_conv", False)
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
            self.blockstack.append(TransformerAttentionBlock(
                channels=self.inner_channels,
                activation=self.activation,
                pos_len=self.pos_len,
                positional_encoding=self.positional_encoding,
                learnable_rope=self.learnable_rope,
                use_qk_norm=self.attention_qk_norm,
                transformer_heads=self.transformer_heads,
                transformer_kv_heads=self.transformer_kv_heads,
                attention_query_head_dim=self.attention_query_head_dim,
                attention_value_head_dim=self.attention_value_head_dim
            ))
            self.blockstack.append(TransformerFFNBlock(
                channels=self.inner_channels,
                activation=self.activation,
                transformer_ffn_channels=self.transformer_ffn_channels,
                use_swiglu=self.use_swiglu,
                transformer_ffn_depthwise_conv=self.transformer_ffn_depthwise_conv 
            ))
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

    def add_reg_dict(self, reg_dict):
        self.pre_btl_conv.add_reg_dict(reg_dict)
        for block in self.blockstack:
            block.add_reg_dict(reg_dict)
        self.post_btl_conv.add_reg_dict(reg_dict)

    def forward(self, x, mask, mask_sum_hw, mask_sum, block_shared_data=None):
        out = self.pre_btl_conv(x, mask)
        for block in self.blockstack:
            in_feature = out
            out = block(in_feature, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum,
                block_shared_data=block_shared_data)
            out = in_feature + out
        out = self.post_btl_conv(out, mask)
        return out

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()

        self.layers_collector = list()

        self.nntype = cfg.nntype  # default:None

        self.activation = cfg.activation.lower()  # default:"relu"
        self.input_channels = cfg.input_channels  # default:43
        self.residual_channels = cfg.residual_channels  # default:None
        self.xsize = cfg.boardsize  # default:19
        self.ysize = cfg.boardsize  # default:19
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
        self.is_pre_act = cfg.is_pre_act
        self.use_rope = False
        self.use_gab = False
        self.use_tab = False
        self.use_tab_freq_mix = False
        self.positional_encoding = cfg.positional_encoding # default:"RoPE"
        if self.positional_encoding in ["RoPE", "RoPE+GAB", "RoPE+TAB", "RoPE+TAB+FreqMix"]:
            self.use_rope = True
        if self.positional_encoding in ["GAB", "RoPE+GAB"]:
            self.use_gab = True
        if self.positional_encoding in ["TAB", "RoPE+TAB"]:
            self.use_tab = True
        if self.positional_encoding in ["TAB+FreqMix", "RoPE+TAB+FreqMix"]:
            self.use_tab_freq_mix = True
        self.learnable_rope = cfg.learnable_rope  # default:False
        self.rope_theta = cfg.rope_theta  # default:100.0
        self.attention_qk_norm = cfg.attention_qk_norm  # default:False
        self.gab_d1 = cfg.gab_d1    # default:16
        self.gab_d2 = cfg.gab_d2    # default:16
        self.gab_num_templates = cfg.gab_num_templates  # default:None
        self.gab_num_fourier_features = cfg.gab_num_fourier_features  # default:None
        self.gab_mlp_hidden = cfg.gab_mlp_hidden  # default:None
        self.tab_c_z = cfg.tab_c_z  # default:None
        self.tab_num_templates = cfg.tab_num_templates  # default:None
        self.tab_num_freqs = cfg.tab_num_freqs    # default:None
        self.tab_num_blocks = cfg.tab_num_blocks  # default:None
        self.tab_dilation = cfg.tab_dilation      # default:None
        self.transformer_heads = cfg.transformer_heads  # default:3
        self.transformer_kv_heads = cfg.transformer_kv_heads  # default:3
        self.attention_query_head_dim = cfg.attention_query_head_dim  # default:32
        self.attention_value_head_dim = cfg.attention_value_head_dim  # default:32
        self.transformer_ffn_channels = cfg.transformer_ffn_channels  # default:256
        self.use_swiglu = cfg.use_swiglu        # default:True
        self.transformer_ffn_depthwise_conv = cfg.transformer_ffn_depthwise_conv  # default:False
        self.use_trunk_channel_gate = cfg.use_trunk_channel_gate          # default:False
        self.use_trunk_residual_backout = cfg.use_trunk_residual_backout  # default:False
        self.opt_name = cfg.optimizer

        self.construct_layers()

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
        additional_block = None
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
            elif component == "TransformerBlock":
                blockargs["positional_encoding"] = self.positional_encoding  # default:"RoPE"
                blockargs["learnable_rope"] = self.learnable_rope  # default:False
                blockargs["rope_theta"] = self.rope_theta  # default:100.0
                blockargs["attention_qk_norm"] = self.attention_qk_norm  # default:False
                blockargs["gab_d1"] = self.gab_d1    # default:16
                blockargs["gab_d2"] = self.gab_d2    # default:16
                blockargs["gab_num_templates"] = self.gab_num_templates  # default:None
                blockargs["gab_num_fourier_features"] = self.gab_num_fourier_features  # default:None
                blockargs["gab_mlp_hidden"] = self.gab_mlp_hidden  # default:None
                blockargs["tab_c_z"] = self.tab_c_z  # default:None
                blockargs["tab_num_templates"] = self.tab_num_templates  # default:None
                blockargs["tab_num_freqs"] = self.tab_num_freqs  # default:None
                blockargs["tab_num_blocks"] = self.tab_num_blocks  # default:None
                blockargs["tab_dilation"] = self.tab_dilation  # default:None
                blockargs["transformer_heads"] = self.transformer_heads  # default:3
                blockargs["transformer_kv_heads"] = self.transformer_kv_heads  # default:3
                blockargs["attention_query_head_dim"] = self.attention_query_head_dim  # default:32
                blockargs["attention_value_head_dim"] = self.attention_value_head_dim  # default:32
                blockargs["transformer_ffn_channels"] = self.transformer_ffn_channels  # default:256
                blockargs["use_swiglu"] = self.use_swiglu  # default:True
                blockargs["transformer_ffn_depthwise_conv"] = self.transformer_ffn_depthwise_conv  # default:False
                block = TransformerAttentionBlock
                additional_block = TransformerFFNBlock
            elif component == "NestedBottleneckTransformerBlock":
                blockargs["bottleneck_channels"] = channels // 2
                assert channels % 2 == 0, ""
                blockargs["positional_encoding"] = self.positional_encoding  # default:"RoPE"
                blockargs["learnable_rope"] = self.learnable_rope  # default:False
                blockargs["rope_theta"] = self.rope_theta  # default:100.0
                blockargs["attention_qk_norm"] = self.attention_qk_norm  # default:False
                blockargs["gab_d1"] = self.gab_d1    # default:16
                blockargs["gab_d2"] = self.gab_d2    # default:16
                blockargs["gab_num_templates"] = self.gab_num_templates  # default:None
                blockargs["gab_num_fourier_features"] = self.gab_num_fourier_features  # default:None
                blockargs["gab_mlp_hidden"] = self.gab_mlp_hidden  # default:None
                blockargs["tab_c_z"] = self.tab_c_z  # default:None
                blockargs["tab_num_templates"] = self.tab_num_templates  # default:None
                blockargs["tab_num_freqs"] = self.tab_num_freqs  # default:None
                blockargs["tab_num_blocks"] = self.tab_num_blocks  # default:None
                blockargs["tab_dilation"] = self.tab_dilation  # default:None
                blockargs["transformer_heads"] = self.transformer_heads  # default:3
                blockargs["transformer_kv_heads"] = self.transformer_kv_heads  # default:3
                blockargs["attention_query_head_dim"] = self.attention_query_head_dim  # default:32
                blockargs["attention_value_head_dim"] = self.attention_value_head_dim  # default:32
                blockargs["transformer_ffn_channels"] = self.transformer_ffn_channels  # default:256
                blockargs["use_swiglu"] = self.use_swiglu  # default:True
                blockargs["transformer_ffn_depthwise_conv"] = self.transformer_ffn_depthwise_conv  # default:False
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
            elif key == "PositionalEncoding":
                blockargs["positional_encoding"] = value
                assert value in ["RoPE", "GAB", "TAB", "TAB+FreqMix", "RoPE+GAB", "RoPE+TAB", "RoPE+TAB+FreqMix"], ""
                if value in ["RoPE", "RoPE+GAB", "RoPE+TAB", "RoPE+TAB+FreqMix"]:
                    self.use_rope = True
                if value in ["GAB", "RoPE+GAB"]:
                    self.use_gab = True
                if value in ["TAB", "RoPE+TAB"]:
                    assert not self.use_tab_freq_mix, ""
                    self.use_tab = True
                if value in ["TAB+FreqMix", "RoPE+TAB+FreqMix"]:
                    assert not self.use_tab, ""
                    self.use_tab_freq_mix = True
            elif key == "LearnableRoPE":
                blockargs["learnable_rope"] = value
            elif key == "RoPETheta":
                blockargs["rope_theta"] = value
            elif key == "AttentionQKNorm":
                blockargs["attention_qk_norm"] = value
            elif key == "GABD1":
                blockargs["gab_d1"] = value
            elif key == "GABD2":
                blockargs["gab_d2"] = value
            elif key == "TransformerHeads":
                blockargs["transformer_heads"] = value
            elif key == "TransformerKVHheads":
                blockargs["transformer_kv_heads"] = value
            elif key == "AttentionQueryHeadDim":
                blockargs["attention_query_head_dim"] = value
            elif key == "AttentionValueHeadDim":
                blockargs["attention_value_head_dim"] = value
            elif key == "TransformerFFNChannels":
                blockargs["transformer_ffn_channels"] = value
            elif key == "UseSwiGLU":
                blockargs["use_swiglu"] = value
            elif key == "TransformerFFNDepthwiseConv":
                blockargs["transformer_ffn_depthwise_conv"] = value
            else:
                raise Exception("Invalid block setting.")
        return block, channels, blockargs, additional_block

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
                "pos_len" : self.xsize,                    # default:19
                "collector" : self.layers_collector
            }
            block, channels, blockargs, additional_block = self.parse_blocksetting(blocksetting, blockargs)
            self.residual_tower.append(block(channels=channels, **blockargs))
            if additional_block is not None:
                self.residual_tower.append(additional_block(channels=channels, **blockargs))

    def construct_layers(self):
        self.global_pool = GlobalPool(is_value_head=False)
        self.global_pool_val = GlobalPool(is_value_head=True)

        if not self.is_pre_act:
            for block in self.stack:
                components = list()
                if type(block) == str:
                    components = block.strip().split('-')
                else:
                    components = block["Block"].strip().split('-')
                for component in components:
                    if component == "TransformerBlock" or component == "NestedBottleneckTransformerBlock":
                        self.is_pre_act = True  # used Transformer
                        break
                if self.is_pre_act:
                    break

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
                use_gamma=False if self.mode == "fixup" else True,
                mode=self.mode,                        # default:"renorm"
                placement="before_block",
                activation=self.activation,            # default::"relu"
                collector=self.layers_collector
            )

        self.create_residual_tower()

        # Trunk channel gating: per-channel learned gate that interpolates between
        # trunk and residual at each block.
        if self.use_trunk_channel_gate:  # default:False
            num_blocks = len(self.residual_tower)
            self.trunk_channel_gate_logits = torch.nn.ParameterList()
            for k in range(num_blocks):
                self.trunk_channel_gate_logits.append(
                    torch.nn.Parameter(torch.zeros(1, self.residual_channels, 1, 1)))
        # Trunk residual backout: a parallel "backout" trunk accumulates alongside the
        # main trunk using the same per-block (trunk_factor, residual_factor) coefficients,
        # but with each residual's contribution additionally multiplied by a per-channel
        # sigmoid gate. Later blocks (and the final trunk output) can subtract a per-channel
        # sigmoid-gated amount of the backout trunk from the main trunk input to effectively
        # "back out" the influence of earlier blocks.
        if self.use_trunk_residual_backout:  # default:False
            num_blocks = len(self.residual_tower)
            # Controls the amount that the initial embedding (channelwise scaled) forms
            # the initial contents of the backout trunk.
            self.backout_add_logit_embedding = torch.nn.Parameter(torch.zeros(1, self.residual_channels, 1, 1))
            # One gate per block (except the last) controlling how much of each block's
            # residual contributes to the backout trunk. The last block's residual is only
            # consumed by the final output, so letting it contribute to a backout the final
            # output could subtract would be incoherent; hence no entry for it.
            self.backout_add_logits = torch.nn.ParameterList()
            for k in range(num_blocks - 1):
                self.backout_add_logits.append(torch.nn.Parameter(torch.zeros(1, self.residual_channels, 1, 1)))
            # One gate per block after the first, controlling how much of the backout
            # trunk is subtracted from the trunk before feeding to that block. Indexed
            # shifted by one: backout_use_logits[i] is consumed by block i+1 (so block 1
            # uses backout_use_logits[0]). The first block sees the raw embedding.
            # Initialized to -2 so we start out only backing out a tiny bit
            # (sigmoid(-2) ~= 0.12).
            self.backout_use_logits = torch.nn.ParameterList()
            for k in range(num_blocks - 1):
                self.backout_use_logits.append(
                    torch.nn.Parameter(torch.full((1, self.residual_channels, 1, 1), -2.0)))
            # Controls how much of the backout trunk is subtracted from the trunk before
            # the final trunk norm+act feeding into the heads. Same -2 init rationale.
            self.backout_use_logit_final = torch.nn.Parameter(torch.full((1, self.residual_channels, 1, 1), -2.0))

        # Create shared GAB template MLP if any block uses GAB
        if self.use_gab:  # default:False
            self.gab_template_mlp = GABTemplateMLP(
                gab_num_templates=self.gab_num_templates,  # default:None
                gab_num_fourier_features=self.gab_num_fourier_features,  # default:None
                gab_mlp_hidden=self.gab_mlp_hidden,  # default:None
                pos_len=self.xsize,  # default:19
                activation=self.activation,  # default:"relu"
            )
        else:
            self.gab_template_mlp = None

        if self.use_tab_freq_mix:  # default:False
            self.tab_module = FrequencyMixingTABModule(
                trunk_channels=self.residual_channels,     # default:None
                tab_c_z=self.tab_c_z,                      # default:None
                tab_num_templates=self.tab_num_templates,  # default:None
                tab_num_blocks=self.tab_num_blocks,        # default:None
                tab_dilation=self.tab_dilation,            # default:None
                activation=self.activation,                # default:"relu"
                pos_len=self.xsize                         # default:19
            )
        elif self.use_tab:  # default:False
            self.tab_module = TABModule(
                trunk_channels=self.residual_channels,     # default:None
                tab_c_z=self.tab_c_z,                      # default:None
                tab_num_templates=self.tab_num_templates,  # default:None
                tab_num_freqs=self.tab_num_freqs,          # default:None
                tab_num_blocks=self.tab_num_blocks,        # default:None
                tab_dilation=self.tab_dilation,            # default:None
                activation=self.activation,                # default:"relu"
                pos_len=self.xsize                         # default:19
            )
        else:
            self.tab_module = None

        if self.is_pre_act:  # default:False
            self.final_block = BatchNorm2d(
                num_features=self.residual_channels,  # default:None
                use_gamma=False,
                mode="fixup"  # Better results are obteined without the norm than by speciflying self.mode
            )
            self.final_act = activation_func(self.activation, inplace=True)  # default:"relu"
        else:
            self.final_block = CustomIdentity()
            self.final_act = nn.Identity()

        self.create_policy_head()
        self.create_value_head()

    def _trunk_residual_factors(self, block_idx):
        """Return (trunk_factor, residual_factor) for combining trunk and residual at block_idx.

        If trunk channel gating is enabled, these are per-channel (1, C, 1, 1) tensors,
        otherwise both factors are the scalar 1.0 (plain residual addition).
        """
        if self.use_trunk_channel_gate:  # default:False
            gate_logit = 0.5 * self.trunk_channel_gate_logits[block_idx]
            w = ((block_idx+2) / (block_idx+1)) / ((1.0 / (block_idx+1)) + torch.exp(-gate_logit))
            trunk_factor = (1.0/(block_idx+1)) * ((block_idx+2) - w)
            residual_factor = w
            return trunk_factor, residual_factor
        else:
            return 1.0, 1.0

    def _run_block_with_backout(self, block, out, backout, block_idx, is_first_block_of_trunk,
        mask_buffers=None, mask=None, mask_sum_hw=None, mask_sum=None, block_shared_data=None):
        """Run a single trunk block, maintaining the parallel backout trunk.

        Returns (new_out, new_backout).

        For every block after the very first one, the block input is
            block_in = out - sigmoid(backout_use_logits[block_idx - 1]) * backout
        so later blocks can learn to subtract out the influence of earlier blocks.
        The backout_use_logits list is shifted by one (no entry for the first block).

        The backout trunk then accumulates with the same per-block (trunk_factor,
        residual_factor) coefficients used for the main trunk, except each residual
        contribution to the backout is additionally multiplied by
        sigmoid(backout_add_logits[block_idx]). The last block does not contribute
        to the backout (its residual is only consumed by the final output).
        """
        if is_first_block_of_trunk:
            block_in = out
        else:
            block_in = out - torch.sigmoid(self.backout_use_logits[block_idx - 1]) * backout

        if (isinstance(block, TransformerAttentionBlock) or
            isinstance(block, TransformerFFNBlock) or
            isinstance(block, NestedBottleneckTransformerBlock)):
            residual = block(
                block_in,
                mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum,
                block_shared_data=block_shared_data
            )
        else:
            residual = block(
                block_in,
                mask_buffers=mask_buffers
            )

        trunk_factor, residual_factor = self._trunk_residual_factors(block_idx)
        new_out = trunk_factor * block_in + residual_factor * residual
        is_last_block_of_trunk = (block_idx == len(self.residual_tower) - 1)
        if is_last_block_of_trunk:
            # Last block's residual is only seen by the final output; don't add it to backout.
            new_backout = trunk_factor * backout
        else:
            backout_residual_gate = torch.sigmoid(self.backout_add_logits[block_idx])
            new_backout = trunk_factor * backout + residual_factor * backout_residual_gate * residual
        return new_out, new_backout

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
        if self.gab_template_mlp is not None:  # default:None
            seq_len = mask.shape[2] * mask.shape[3]  # H * W
            templates = self.gab_template_mlp(seq_len)
            block_shared_data[GAB_TEMPLATES] = GABTemplateData(templates=templates)
        if self.tab_module is not None:  # default:None
            tab_keys, tab_queries = self.tab_module(x, mask)
            block_shared_data[TAB_KQ] = TABKeyQueryData(keys=tab_keys, queries=tab_queries)

        # residual tower
        # Initialize the parallel backout trunk if enabled.
        if self.use_trunk_residual_backout:  # default:False
            backout = torch.sigmoid(self.backout_add_logit_embedding) * x
        else:
            backout = None
        # for block in self.residual_tower:
        for i, block in enumerate(self.residual_tower):
            if (isinstance(block, TransformerAttentionBlock) or
                isinstance(block, TransformerFFNBlock) or
                isinstance(block, NestedBottleneckTransformerBlock)):
                if self.use_trunk_residual_backout:  # default:False
                    x, backout = self._run_block_with_backout(
                        block, x, backout, i, is_first_block_of_trunk=(i == 0),
                        mask=mask,
                        mask_sum_hw=mask_sum_hw_transformer,
                        mask_sum=mask_sum_transformer,
                        block_shared_data=block_shared_data
                    )
                else:
                    residual = block(x,
                        mask=mask,
                        mask_sum_hw=mask_sum_hw_transformer,
                        mask_sum=mask_sum_transformer,
                        block_shared_data=block_shared_data
                    )
                    if self.use_trunk_channel_gate:
                        trunk_factor, residual_factor = self._trunk_residual_factors(i)
                        x = trunk_factor * x + residual_factor * residual
                    else:
                        x = x + residual
            elif self.is_pre_act:  # default:False
                if self.use_trunk_residual_backout:  # default:False
                    x, backout = self._run_block_with_backout(
                        block, x, backout, i, is_first_block_of_trunk=(i == 0),
                        mask_buffers=mask_buffers
                    )
                else:
                    residual = block(x, mask_buffers)
                    if self.use_trunk_channel_gate:
                        trunk_factor, residual_factor = self._trunk_residual_factors(i)
                        x = trunk_factor * x + residual_factor * residual
                    else:
                        x = x + residual
            else:
                x = block(x, mask_buffers)

        if self.use_trunk_residual_backout:  # default:False
            x = x - torch.sigmoid(self.backout_use_logit_final) * backout

        x = self.final_block(x, mask)
        x = self.final_act(x)

        with autocast("cuda", enabled=False):
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
        reg_dict["normal_gab"] = []
        reg_dict["output"] = []
        reg_dict["noreg"] = []
        reg_dict["output_noreg"] = []
        reg_dict["gab_mlp"] = []
        reg_dict["tab_module"] = []

        self.input_conv.add_reg_dict(reg_dict, placement="before_block")
        for block in self.residual_tower:
            block.add_reg_dict(reg_dict)
        if self.gab_template_mlp is not None:  # default:None
            self.gab_template_mlp.add_reg_dict(reg_dict)
        if self.tab_module is not None:        # default:None
            self.tab_module.add_reg_dict(reg_dict)
        if self.use_trunk_channel_gate:        # default:None
            for gate_logit in self.trunk_channel_gate_logits:
                reg_dict["normal_gamma"].append(gate_logit)
        if self.use_trunk_residual_backout:    # default:None
            backout_reg = "normal_gamma"
            reg_dict[backout_reg].append(self.backout_add_logit_embedding)
            for logit in self.backout_add_logits:
                reg_dict[backout_reg].append(logit)
            for logit in self.backout_use_logits:
                reg_dict[backout_reg].append(logit)
            reg_dict[backout_reg].append(self.backout_use_logit_final)
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
