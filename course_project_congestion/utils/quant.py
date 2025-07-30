# this file is adapted from the BinaryBERT project
# https://github.com/ht-zhou/binary-bert/blob/master/BinaryBERT/transformer/utils_quant.py

# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        if num_bits < 32:
            input = torch.where(input < clip_val[1], input, clip_val[1]) # Max clipping
            input = torch.where(input > clip_val[0], input, clip_val[0]) # Min clipping

            if layerwise:
                max_input = torch.max(torch.abs(input)).expand_as(input)
            else:
                if input.ndimension() <= 3: # weight & hidden layer
                    max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
                elif input.ndimension() == 4: # NHWC or NCHW
                    tmp = input.view(input.shape[0], input.shape[1], -1) # N, C, H*W
                    max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
                else:
                    raise ValueError("Unsupported tensor dimension: {}".format(input.ndimension()))

            s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-8) # Add epsilon for stability
            output = torch.round(input * s).div(s)
        else:
            output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        num_bits = ctx.num_bits
        grad_input = grad_output.clone()
        grad_clip = None
        if num_bits < 32:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
            grad_clip_pos = (grad_output * input.ge(clip_val[1]).float()).sum()
            grad_clip_neg = (grad_output * input.le(clip_val[0]).float()).sum()
            grad_clip = torch.tensor([grad_clip_neg, grad_clip_pos], device=input.device)
        return grad_input, grad_clip, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        if num_bits < 32:
            input = torch.where(input < clip_val[1], input, clip_val[1]) # Max clipping
            input = torch.where(input > clip_val[0], input, clip_val[0]) # Min clipping

            if layerwise:
                alpha = (input.max() - input.min()).detach()
                beta = input.min().detach()
            else:
                if input.ndimension() <= 3:
                    alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                    beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
                elif input.ndimension() == 4:
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                             tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                    beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
                else:
                    raise ValueError("Unsupported tensor dimension: {}".format(input.ndimension()))

            input_normalized = (input - beta) / (alpha + 1e-8) # Add epsilon
            s = (2 ** num_bits - 1)
            quant_input = torch.round(input_normalized * s).div(s) # div(s + 1e-8) might be safer if s can be 0
            output = quant_input * (alpha + 1e-8) + beta
        else:
            output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        num_bits = ctx.num_bits
        grad_input = grad_output.clone()
        grad_clip = None
        if num_bits < 32:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
            grad_clip_pos = (grad_output * input.ge(clip_val[1]).float()).sum()
            grad_clip_neg = (grad_output * input.le(clip_val[0]).float()).sum()
            grad_clip = torch.tensor([grad_clip_neg, grad_clip_pos], device=input.device)
        return grad_input, grad_clip, None, None


class LaqQuantizer(torch.autograd.Function): # can only be used for weight
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise): # clip_val not used by LAQ's formula
        ctx.save_for_backward(input)
        if num_bits < 32:
            D = torch.ones_like(input) # Simplified Hessian approximation
            if layerwise:
                if num_bits == 1:
                    alpha = (D * input).abs().sum() / (D.sum() + 1e-8)
                    b = input.sign()
                    output = alpha * b
                else:
                    n_levels = 2 ** (num_bits - 1) - 1
                    b = input.sign()
                    alpha = (b * D * input).abs().sum() / ((b * D).abs().sum() + 1e-8)
                    b = ((input / (alpha + 1e-8)).clamp(-1., 1.) * n_levels).round() / (n_levels + 1e-8)
                    for _ in range(10): # Iterative refinement
                        alpha = (b * D * input).abs().sum() / ((b * D).abs().sum() + 1e-8)
                        b = ((input / (alpha + 1e-8)).clamp(-1., 1.) * n_levels).round() / (n_levels + 1e-8)
                    output = alpha * b
            else: # Not layerwise (e.g. per-channel for weights)
                sum_dims = list(range(1, input.dim())) # Sum over all dims except the first (batch/output_channel)
                if num_bits == 1:
                    alpha = (D * input).abs().sum(dim=sum_dims, keepdim=True) / (D.sum(dim=sum_dims, keepdim=True) + 1e-8)
                    b = input.sign()
                    output = alpha * b
                else:
                    n_levels = 2 ** (num_bits - 1) - 1
                    b = input.sign()
                    alpha = (b * D * input).abs().sum(dim=sum_dims, keepdim=True) / ((b * D).abs().sum(dim=sum_dims, keepdim=True) + 1e-8)
                    b = ((input / (alpha + 1e-8)).clamp(-1., 1.) * n_levels).round() / (n_levels + 1e-8)
                    for _ in range(10):
                        alpha = (b * D * input).abs().sum(dim=sum_dims, keepdim=True) / ((b * D).abs().sum(dim=sum_dims, keepdim=True) + 1e-8)
                        b = ((input / (alpha + 1e-8)).clamp(-1., 1.) * n_levels).round() / (n_levels + 1e-8)
                    output = alpha * b
        else:
            output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class LsqStepSize(nn.Parameter):
    # __new__ is used for immutable types or when subclassing nn.Parameter in this specific way
    # to set the data directly. For most nn.Module subclasses, __init__ is standard.
    def __new__(cls, initial_value_tensor):
        # It's generally safer to use super().__new__(cls) if you override __new__
        # For nn.Parameter, this specific pattern is sometimes seen to directly initialize data.
        param = super(LsqStepSize, cls).__new__(cls, data=initial_value_tensor.clone().detach(), requires_grad=True)
        return param

    def __init__(self, initial_value_tensor):
        # The actual initialization of 'initialized' should happen in __init__
        # even if __new__ handles data creation.
        # However, since __new__ returns the instance, __init__ might be called on an already
        # partially initialized object. Standard practice is to do most logic in __init__.
        # For nn.Parameter subclassing, if __new__ handles 'data', then __init__
        # primarily calls super().__init__() and sets other attributes.
        # Let's ensure super().__init__() is called correctly for nn.Parameter.
        # nn.Parameter's __init__ itself might not take arguments other than data and requires_grad,
        # which are often handled in __new__ when subclassing it this way.

        # The original code used super(LsqStepSize, self).__new__(nn.Parameter, data=...)
        # which is a bit unusual. A more common way to subclass nn.Parameter if you need
        # custom behavior is:
        # class LsqStepSize(nn.Parameter):
        #     def __init__(self, data, requires_grad=True):
        #         super().__init__(data, requires_grad)
        #         self.initialized = False
        # For this specific case, the original code structure is a bit unique for nn.Parameter.
        # We'll stick to the logic provided, assuming `initial_value_tensor` is the intended data.
        self.initialized = False # This should be set here.

    def _initialize(self, init_val_tensor):
        if not isinstance(init_val_tensor, torch.Tensor):
            init_val_tensor = torch.tensor(init_val_tensor, device=self.data.device, dtype=self.data.dtype)

        if self.data.numel() == 1 and init_val_tensor.numel() == 1:
            self.data.fill_(init_val_tensor.item())
        elif self.data.shape == init_val_tensor.shape:
            self.data.copy_(init_val_tensor)
        else:
            raise ValueError(f"Shape mismatch for LSQ step size. Current: {self.data.shape}, Init: {init_val_tensor.shape}")

        # logging.info(f'LsqStepSize initialized to: {self.data.item() if self.data.numel()==1 else self.data}')
        self.initialized = True

    def initialize_wrapper(self, tensor_to_quant, num_bits, symmetric, init_method='default'):
        if self.initialized:
            return

        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** num_bits - 1
        if Qp == 0: Qp = 1

        if init_method == 'default':
            mean_abs = tensor_to_quant.abs().mean()
            if mean_abs.item() < 1e-9:
                init_val_scalar = torch.tensor(1e-5, device=tensor_to_quant.device, dtype=tensor_to_quant.dtype)
            else:
                init_val_scalar = (2 * mean_abs / math.sqrt(Qp)) if symmetric \
                                  else (4 * mean_abs / math.sqrt(Qp))
            init_val_scalar = torch.clamp(init_val_scalar, min=1e-5)

        elif init_method == 'uniform':
            init_val_scalar = torch.tensor(1.0 / (2 * Qp + 1) if symmetric else 1.0 / Qp, device=tensor_to_quant.device, dtype=tensor_to_quant.dtype if hasattr(tensor_to_quant,'dtype') else torch.float32)
            init_val_scalar = torch.clamp(init_val_scalar, min=1e-5)
        else:
            raise ValueError(f"Unknown LSQ init_method: {init_method}")

        self._initialize(init_val_scalar)


class SymLsqQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha_param, num_bits, layerwise):
        if not layerwise:
            raise NotImplementedError("SymLsqQuantizer currently supports layerwise=True only")
        ctx.num_bits = num_bits
        if num_bits >= 32:
            return input

        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1

        alpha_val = alpha_param.data
        if not alpha_param.initialized:
             alpha_param.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
             alpha_val = alpha_param.data

        if not alpha_val.item() > 0: # Added check for item() in case alpha_val is a 0-dim tensor
            logging.error(f'LSQ alpha must be positive. Got: {alpha_val.item() if alpha_val.numel() == 1 else alpha_val}')
            # Fallback or raise error
            alpha_val.data.fill_(1e-5) # Fallback to a small positive
            # raise ValueError(f'LSQ alpha must be positive. Got: {alpha_val.item()}')


        grad_scale = 1.0

        ctx.save_for_backward(input, alpha_val)
        ctx.other = grad_scale, Qn, Qp

        q_input = (input / alpha_val).round().clamp(Qn, Qp)
        w_q = q_input * alpha_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 32:
            return grad_output, None, None, None

        input_tensor, alpha_val = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other

        q_input_unclamped = input_tensor / alpha_val

        indicate_middle = (q_input_unclamped.ge(Qn) & q_input_unclamped.le(Qp)).float()
        grad_input = indicate_middle * grad_output

        term1 = indicate_middle * (-q_input_unclamped + q_input_unclamped.round())
        term2 = (q_input_unclamped < Qn).float() * Qn
        term3 = (q_input_unclamped > Qp).float() * Qp

        grad_alpha_tensor = ((term1 + term2 + term3) * grad_output * grad_scale).sum().unsqueeze(dim=0)

        return grad_input, grad_alpha_tensor, None, None


class AsymLsqQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha_param, num_bits, layerwise):
        if not layerwise:
            raise NotImplementedError("AsymLsqQuantizer currently supports layerwise=True only")
        ctx.num_bits = num_bits
        if num_bits >= 32:
            return input

        Qn = 0
        Qp = 2 ** num_bits - 1

        min_val = input.min().item()
        input_shifted = input - min_val

        alpha_val = alpha_param.data
        if not alpha_param.initialized:
            alpha_param.initialize_wrapper(input_shifted, num_bits, symmetric=False, init_method='default')
            alpha_val = alpha_param.data

        if not alpha_val.item() > 0: # Added check
            logging.error(f'LSQ alpha must be positive. Got: {alpha_val.item() if alpha_val.numel() == 1 else alpha_val}')
            alpha_val.data.fill_(1e-5) # Fallback
            # raise ValueError(f'LSQ alpha must be positive. Got: {alpha_val.item()}')


        grad_scale = 1.0

        ctx.save_for_backward(input_shifted, alpha_val)
        ctx.other = grad_scale, Qn, Qp, min_val

        q_input_shifted = (input_shifted / alpha_val).round().clamp(Qn, Qp)
        w_q_shifted = q_input_shifted * alpha_val
        w_q = w_q_shifted + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 32:
            return grad_output, None, None, None

        input_shifted, alpha_val = ctx.saved_tensors
        grad_scale, Qn, Qp, min_val = ctx.other

        q_input_shifted_unclamped = input_shifted / alpha_val

        indicate_middle = (q_input_shifted_unclamped.ge(Qn) & q_input_shifted_unclamped.le(Qp)).float()
        grad_input = indicate_middle * grad_output

        term1 = indicate_middle * (-q_input_shifted_unclamped + q_input_shifted_unclamped.round())
        term2 = (q_input_shifted_unclamped < Qn).float() * Qn
        term3 = (q_input_shifted_unclamped > Qp).float() * Qp

        grad_alpha_tensor = ((term1 + term2 + term3) * grad_output * grad_scale).sum().unsqueeze(dim=0)

        return grad_input, grad_alpha_tensor, None, None


class BwnQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        ctx.save_for_backward(input)
        if layerwise:
            scaling_factor = input.abs().mean()
            result = input.sign() * scaling_factor
        else:
            sum_dims = list(range(1, input.dim()))
            scaling_factor = input.abs().mean(dim=sum_dims, keepdim=True)
            result = input.sign() * scaling_factor
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class TwnQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        ctx.save_for_backward(input)
        if layerwise:
            delta_threshold = 0.7 * input.abs().mean()
            mask = input.abs().gt(delta_threshold)
            alpha_sum_val = (input.abs() * mask.float()).sum()
            mask_sum_val = mask.float().sum()
            alpha = alpha_sum_val / (mask_sum_val + 1e-8) if mask_sum_val > 0 else torch.tensor(0.0, device=input.device, dtype=input.dtype)


            output = torch.zeros_like(input)
            output[input > delta_threshold] = alpha
            output[input < -delta_threshold] = -alpha
        else:
            sum_dims = list(range(1, input.dim()))
            delta_threshold = 0.7 * input.abs().mean(dim=sum_dims, keepdim=True)

            mask = input.abs().gt(delta_threshold)
            alpha_sum_val = (input.abs() * mask.float()).sum(dim=sum_dims, keepdim=True)
            mask_sum_val = mask.float().sum(dim=sum_dims, keepdim=True)
            alpha = alpha_sum_val / (mask_sum_val + 1e-8) # Broadcasting should handle if mask_sum_val has 0s where alpha_sum_val is also 0


            output = torch.zeros_like(input)
            # Need to be careful with broadcasting alpha if it becomes 0 for some channels
            alpha_expanded = alpha.expand_as(output)
            output[input > delta_threshold] = alpha_expanded[input > delta_threshold]
            output[input < -delta_threshold] = -alpha_expanded[input < -delta_threshold] # Negative sign should be handled by alpha if it's scalar

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def act_quant_fn(input, clip_val_or_alpha, num_bits, symmetric, quant_method, layerwise):
    if num_bits >= 32:
        return input

    if quant_method == "uniform":
        quant_fn = SymQuantizer if symmetric else AsymQuantizer
    elif quant_method == "lsq":
        quant_fn = SymLsqQuantizer if symmetric else AsymLsqQuantizer
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "twn" and num_bits == 2:
        quant_fn = TwnQuantizer
    else:
        raise ValueError(f"Unknown activation quant_method: {quant_method} or incompatible num_bits: {num_bits}")

    output = quant_fn.apply(input, clip_val_or_alpha, num_bits, layerwise)
    return output


def weight_quant_fn(weight, clip_val_or_alpha, num_bits, symmetric, quant_method, layerwise):
    if num_bits >= 32:
        return weight

    if quant_method == "uniform":
        quant_fn = SymQuantizer if symmetric else AsymQuantizer
    elif quant_method == "lsq":
        quant_fn = SymLsqQuantizer if symmetric else AsymLsqQuantizer
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "twn" and num_bits == 2:
        quant_fn = TwnQuantizer
    elif quant_method == "laq":
        quant_fn = LaqQuantizer
    else:
        raise ValueError(f"Unknown weight quant_method: {quant_method} or incompatible num_bits: {num_bits}")

    output = quant_fn.apply(weight, clip_val_or_alpha, num_bits, layerwise)
    return output


class QuantizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 # Quantization specific parameters
                 clip_val=2.5, weight_bits=8, input_bits=8,
                 learnable_clip=False,
                 symmetric_weights=True, symmetric_acts=False,
                 weight_layerwise=True, input_layerwise=True,
                 weight_quant_method="uniform", input_quant_method="uniform"):
        super().__init__(in_features, out_features, bias) # MODIFIED: super call

        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable_clip = learnable_clip

        self.symmetric_weights = symmetric_weights
        self.symmetric_acts = symmetric_acts

        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method

        self.weight_clip_param = QuantizeLinear._create_quant_param( # MODIFIED: Call static method
            self.weight_quant_method, self.learnable_clip, clip_val, for_weights=True
        )
        self.input_clip_param = QuantizeLinear._create_quant_param( # MODIFIED: Call static method
            self.input_quant_method, self.learnable_clip, clip_val, for_weights=False
        )

    @staticmethod # MODIFIED: Added @staticmethod
    def _create_quant_param(quant_method, learnable_clip_flag, init_val, for_weights): # MODIFIED: Removed self
        """ Helper to create quantization parameter (clip_val tensor or LsqStepSize object) """
        if quant_method == 'uniform':
            if isinstance(init_val, (int, float)):
                clip_tensor = torch.tensor([-float(init_val), float(init_val)])
            elif isinstance(init_val, (list, tuple)) and len(init_val) == 2:
                clip_tensor = torch.tensor([float(init_val[0]), float(init_val[1])])
            else:
                logging.warning(f"Uniform clip_val init expected scalar or 2-elem list, got {init_val}. Defaulting to symmetric 2.5.")
                clip_tensor = torch.tensor([-2.5, 2.5])

            if learnable_clip_flag:
                return nn.Parameter(clip_tensor)
            else:
                return clip_tensor

        elif quant_method == 'lsq':
            if not learnable_clip_flag: # Simplified warning
                 logging.warning(f"LSQ for {'weights' if for_weights else 'activations'} is typically used with learnable_clip=True (learnable alpha).")
            return LsqStepSize(torch.tensor(1.0))

        elif quant_method in ["bwn", "twn", "laq"]:
            return None
        else:
            raise ValueError(f"Unknown quant_method '{quant_method}' for _create_quant_param")

    def forward(self, input):
        if torch.is_tensor(self.weight_clip_param) and self.weight_clip_param.device != self.weight.device:
            self.weight_clip_param = self.weight_clip_param.to(self.weight.device)
        elif isinstance(self.weight_clip_param, LsqStepSize) and self.weight_clip_param.data.device != self.weight.device:
            # For LsqStepSize (nn.Parameter), move the parameter itself, not just .data
            self.weight_clip_param = self.weight_clip_param.to(self.weight.device)


        if torch.is_tensor(self.input_clip_param) and self.input_clip_param.device != input.device:
            self.input_clip_param = self.input_clip_param.to(input.device)
        elif isinstance(self.input_clip_param, LsqStepSize) and self.input_clip_param.data.device != input.device:
            self.input_clip_param = self.input_clip_param.to(input.device)


        quantized_weight = weight_quant_fn(self.weight, self.weight_clip_param,
                                           num_bits=self.weight_bits,
                                           symmetric=self.symmetric_weights,
                                           quant_method=self.weight_quant_method,
                                           layerwise=self.weight_layerwise)

        quantized_input = act_quant_fn(input, self.input_clip_param,
                                       num_bits=self.input_bits,
                                       symmetric=self.symmetric_acts,
                                       quant_method=self.input_quant_method,
                                       layerwise=self.input_layerwise)

        output = F.linear(quantized_input, quantized_weight, self.bias)
        return output


class QuantizeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, # MODIFIED: Renamed from init
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 clip_val=2.5, weight_bits=8,
                 learnable_clip=False,
                 symmetric_weights=True,
                 embed_layerwise=False,
                 weight_quant_method="uniform"):
        super().__init__( # MODIFIED: super call
            num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
            scale_grad_by_freq, sparse, _weight)

        self.weight_bits = weight_bits
        self.learnable_clip = learnable_clip
        self.symmetric_weights = symmetric_weights
        self.embed_layerwise = embed_layerwise
        self.weight_quant_method = weight_quant_method

        self.embed_clip_param = QuantizeLinear._create_quant_param(
            self.weight_quant_method, self.learnable_clip, clip_val, for_weights=True
        )

    def forward(self, input):
        if torch.is_tensor(self.embed_clip_param) and self.embed_clip_param.device != self.weight.device:
            self.embed_clip_param = self.embed_clip_param.to(self.weight.device)
        elif isinstance(self.embed_clip_param, LsqStepSize) and self.embed_clip_param.data.device != self.weight.device:
            self.embed_clip_param = self.embed_clip_param.to(self.weight.device)


        quantized_weight = weight_quant_fn(self.weight, self.embed_clip_param,
                                           num_bits=self.weight_bits,
                                           symmetric=self.symmetric_weights,
                                           quant_method=self.weight_quant_method,
                                           layerwise=self.embed_layerwise)

        return F.embedding(
            input, quantized_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class QuantizeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, # MODIFIED: Renamed from init
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 clip_val=2.5, weight_bits=8, input_bits=8,
                 learnable_clip=False,
                 symmetric_weights=True, symmetric_acts=False,
                 weight_layerwise=True, input_layerwise=True,
                 weight_quant_method="uniform", input_quant_method="uniform"):
        super().__init__( # MODIFIED: super call
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable_clip = learnable_clip

        self.symmetric_weights = symmetric_weights
        self.symmetric_acts = symmetric_acts

        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method

        self.weight_clip_param = QuantizeLinear._create_quant_param(
            self.weight_quant_method, self.learnable_clip, clip_val, for_weights=True
        )
        self.input_clip_param = QuantizeLinear._create_quant_param(
            self.input_quant_method, self.learnable_clip, clip_val, for_weights=False
        )

    def forward(self, input):
        if torch.is_tensor(self.weight_clip_param) and self.weight_clip_param.device != self.weight.device:
            self.weight_clip_param = self.weight_clip_param.to(self.weight.device)
        elif isinstance(self.weight_clip_param, LsqStepSize) and self.weight_clip_param.data.device != self.weight.device:
            self.weight_clip_param = self.weight_clip_param.to(self.weight.device)


        if torch.is_tensor(self.input_clip_param) and self.input_clip_param.device != input.device:
            self.input_clip_param = self.input_clip_param.to(input.device)
        elif isinstance(self.input_clip_param, LsqStepSize) and self.input_clip_param.data.device != input.device:
            self.input_clip_param = self.input_clip_param.to(input.device)


        quantized_weight = weight_quant_fn(self.weight, self.weight_clip_param,
                                            num_bits=self.weight_bits,
                                            symmetric=self.symmetric_weights,
                                            quant_method=self.weight_quant_method,
                                            layerwise=self.weight_layerwise)

        quantized_input = act_quant_fn(input, self.input_clip_param,
                                       num_bits=self.input_bits,
                                       symmetric=self.symmetric_acts,
                                       quant_method=self.input_quant_method,
                                       layerwise=self.input_layerwise)

        output = F.conv2d(quantized_input, quantized_weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output


class QuantizeConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, # MODIFIED: Renamed from init
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros',
                 clip_val=2.5, weight_bits=8, input_bits=8,
                 learnable_clip=False,
                 symmetric_weights=True, symmetric_acts=False,
                 weight_layerwise=True, input_layerwise=True,
                 weight_quant_method="uniform", input_quant_method="uniform"):
        super().__init__( # MODIFIED: super call
            in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation, padding_mode)

        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable_clip = learnable_clip

        self.symmetric_weights = symmetric_weights
        self.symmetric_acts = symmetric_acts

        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method

        self.weight_clip_param = QuantizeLinear._create_quant_param(
            self.weight_quant_method, self.learnable_clip, clip_val, for_weights=True
        )
        self.input_clip_param = QuantizeLinear._create_quant_param(
            self.input_quant_method, self.learnable_clip, clip_val, for_weights=False
        )

    def forward(self, input, output_size=None):
        if torch.is_tensor(self.weight_clip_param) and self.weight_clip_param.device != self.weight.device:
            self.weight_clip_param = self.weight_clip_param.to(self.weight.device)
        elif isinstance(self.weight_clip_param, LsqStepSize) and self.weight_clip_param.data.device != self.weight.device:
            self.weight_clip_param = self.weight_clip_param.to(self.weight.device)


        if torch.is_tensor(self.input_clip_param) and self.input_clip_param.device != input.device:
            self.input_clip_param = self.input_clip_param.to(input.device)
        elif isinstance(self.input_clip_param, LsqStepSize) and self.input_clip_param.data.device != input.device:
            self.input_clip_param = self.input_clip_param.to(input.device)


        quantized_weight = weight_quant_fn(self.weight, self.weight_clip_param,
                                            num_bits=self.weight_bits,
                                            symmetric=self.symmetric_weights,
                                            quant_method=self.weight_quant_method,
                                            layerwise=self.weight_layerwise)

        quantized_input = act_quant_fn(input, self.input_clip_param,
                                       num_bits=self.input_bits,
                                       symmetric=self.symmetric_acts,
                                       quant_method=self.input_quant_method,
                                       layerwise=self.input_layerwise)

        output = F.conv_transpose2d(quantized_input, quantized_weight, self.bias, self.stride,
                                    self.padding, self.output_padding, self.groups, self.dilation)
        return output

class QuantizeInstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, # MODIFIED: Renamed from init
                 track_running_stats=False,
                 clip_val=2.5, param_bits=8, input_bits=8,
                 learnable_clip=False,
                 symmetric_params=True, symmetric_acts=False,
                 param_layerwise=True, input_layerwise=True,
                 param_quant_method="uniform", input_quant_method="uniform"):
        super().__init__( # MODIFIED: super call
            num_features, eps, momentum, affine, track_running_stats)

        self.param_bits = param_bits
        self.input_bits = input_bits
        self.learnable_clip = learnable_clip
        self.symmetric_params = symmetric_params
        self.symmetric_acts = symmetric_acts
        self.param_layerwise = param_layerwise
        self.input_layerwise = input_layerwise
        self.param_quant_method = param_quant_method
        self.input_quant_method = input_quant_method

        if self.affine:
            self.gamma_clip_param = QuantizeLinear._create_quant_param(
                self.param_quant_method, self.learnable_clip, clip_val, for_weights=True
            )
            self.beta_clip_param = QuantizeLinear._create_quant_param(
                self.param_quant_method, self.learnable_clip, clip_val, for_weights=True
            )
        else: # Ensure attributes exist even if not affine, to prevent AttributeError
            self.gamma_clip_param = None
            self.beta_clip_param = None


        self.input_clip_param = QuantizeLinear._create_quant_param(
            self.input_quant_method, self.learnable_clip, clip_val, for_weights=False
        )

    def forward(self, input):
        q_input_clip = self.input_clip_param
        if torch.is_tensor(q_input_clip) and q_input_clip.device != input.device:
            q_input_clip = q_input_clip.to(input.device)
        elif isinstance(q_input_clip, LsqStepSize) and q_input_clip.data.device != input.device:
            q_input_clip = q_input_clip.to(input.device)


        quantized_input = act_quant_fn(input, q_input_clip,
                                       num_bits=self.input_bits,
                                       symmetric=self.symmetric_acts,
                                       quant_method=self.input_quant_method,
                                       layerwise=self.input_layerwise)

        q_gamma = self.weight
        q_beta = self.bias

        if self.affine:
            if self.weight is not None:
                gamma_clip = self.gamma_clip_param
                if torch.is_tensor(gamma_clip) and gamma_clip.device != self.weight.device:
                    gamma_clip = gamma_clip.to(self.weight.device)
                elif isinstance(gamma_clip, LsqStepSize) and gamma_clip.data.device != self.weight.device:
                    gamma_clip = gamma_clip.to(self.weight.device)

                q_gamma = weight_quant_fn(self.weight, gamma_clip,
                                          num_bits=self.param_bits,
                                          symmetric=self.symmetric_params,
                                          quant_method=self.param_quant_method,
                                          layerwise=self.param_layerwise)

            if self.bias is not None:
                beta_clip = self.beta_clip_param
                if torch.is_tensor(beta_clip) and beta_clip.device != self.bias.device:
                    beta_clip = beta_clip.to(self.bias.device)
                elif isinstance(beta_clip, LsqStepSize) and beta_clip.data.device != self.bias.device:
                    beta_clip = beta_clip.to(self.bias.device)


                q_beta = weight_quant_fn(self.bias, beta_clip,
                                         num_bits=self.param_bits,
                                         symmetric=self.symmetric_params,
                                         quant_method=self.param_quant_method,
                                         layerwise=self.param_layerwise)

        return F.instance_norm(
            quantized_input, self.running_mean, self.running_var,
            q_gamma, q_beta,
            self.training or not self.track_running_stats,
            self.momentum, self.eps
        )

class QuantizeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, # MODIFIED: Renamed from init
                 track_running_stats=True,
                 clip_val=2.5, param_bits=8, input_bits=8,
                 learnable_clip=False,
                 symmetric_params=True, symmetric_acts=False,
                 param_layerwise=True, input_layerwise=True,
                 param_quant_method="uniform", input_quant_method="uniform"):
        super().__init__( # MODIFIED: super call
            num_features, eps, momentum, affine, track_running_stats)

        self.param_bits = param_bits
        self.input_bits = input_bits
        self.learnable_clip = learnable_clip
        self.symmetric_params = symmetric_params
        self.symmetric_acts = symmetric_acts
        self.param_layerwise = param_layerwise
        self.input_layerwise = input_layerwise
        self.param_quant_method = param_quant_method
        self.input_quant_method = input_quant_method

        if self.affine:
            self.gamma_clip_param = QuantizeLinear._create_quant_param(
                self.param_quant_method, self.learnable_clip, clip_val, for_weights=True
            )
            self.beta_clip_param = QuantizeLinear._create_quant_param(
                self.param_quant_method, self.learnable_clip, clip_val, for_weights=True
            )
        else: # Ensure attributes exist
            self.gamma_clip_param = None
            self.beta_clip_param = None

        self.input_clip_param = QuantizeLinear._create_quant_param(
            self.input_quant_method, self.learnable_clip, clip_val, for_weights=False
        )

    def forward(self, input):
        q_input_clip = self.input_clip_param
        if torch.is_tensor(q_input_clip) and q_input_clip.device != input.device:
            q_input_clip = q_input_clip.to(input.device)
        elif isinstance(q_input_clip, LsqStepSize) and q_input_clip.data.device != input.device:
            q_input_clip = q_input_clip.to(input.device)


        quantized_input = act_quant_fn(input, q_input_clip,
                                       num_bits=self.input_bits,
                                       symmetric=self.symmetric_acts,
                                       quant_method=self.input_quant_method,
                                       layerwise=self.input_layerwise)

        q_gamma = self.weight
        q_beta = self.bias

        if self.affine:
            if self.weight is not None:
                gamma_clip = self.gamma_clip_param
                if torch.is_tensor(gamma_clip) and gamma_clip.device != self.weight.device:
                    gamma_clip = gamma_clip.to(self.weight.device)
                elif isinstance(gamma_clip, LsqStepSize) and gamma_clip.data.device != self.weight.device:
                    gamma_clip = gamma_clip.to(self.weight.device)

                q_gamma = weight_quant_fn(self.weight, gamma_clip,
                                          num_bits=self.param_bits,
                                          symmetric=self.symmetric_params,
                                          quant_method=self.param_quant_method,
                                          layerwise=self.param_layerwise)

            if self.bias is not None:
                beta_clip = self.beta_clip_param
                if torch.is_tensor(beta_clip) and beta_clip.device != self.bias.device:
                    beta_clip = beta_clip.to(self.bias.device)
                elif isinstance(beta_clip, LsqStepSize) and beta_clip.data.device != self.bias.device:
                    beta_clip = beta_clip.to(self.bias.device)

                q_beta = weight_quant_fn(self.bias, beta_clip,
                                         num_bits=self.param_bits,
                                         symmetric=self.symmetric_params,
                                         quant_method=self.param_quant_method,
                                         layerwise=self.param_layerwise)

        return F.batch_norm(
            quantized_input, self.running_mean, self.running_var, q_gamma, q_beta,
            self.training or not self.track_running_stats,
            self.momentum, self.eps
        )

SUPPORTED_QUANTIZABLE_MODULES = {
    nn.Conv2d: QuantizeConv2d,
    nn.ConvTranspose2d: QuantizeConvTranspose2d,
    nn.Linear: QuantizeLinear,
    nn.InstanceNorm2d: QuantizeInstanceNorm2d, 
    nn.BatchNorm2d: QuantizeBatchNorm2d, 
}

def get_default_quant_config(
    # 权重相关
    weight_bits=8, 
    weight_quant_method="uniform",
    symmetric_weights=True, 
    # 激活相关 (即输入到可量化层的激活)
    input_bits=32, # 默认不量化激活
    input_quant_method="uniform",
    symmetric_acts=False, 
    # Norm层参数相关 (gamma/beta)
    param_bits=8, # affine参数的比特数
    param_quant_method="uniform", # affine参数的量化方法
    symmetric_params=True, # affine参数的对称性
    # 通用
    learnable_clip=False, 
    clip_val_init=2.5, 
    layerwise_weights=True, # 对应原TA代码的weight_layerwise
    layerwise_acts=True,    # 对应原TA代码的input_layerwise
    layerwise_params=True   # Norm层参数的layerwise
    ):
    return {
        'weight_bits': weight_bits,
        'weight_quant_method': weight_quant_method,
        'symmetric_weights': symmetric_weights,
        'weight_layerwise': layerwise_weights, # Renamed for clarity

        'input_bits': input_bits,
        'input_quant_method': input_quant_method,
        'symmetric_acts': symmetric_acts,
        'input_layerwise': layerwise_acts, # Renamed for clarity

        'param_bits': param_bits, # For InstanceNorm/BatchNorm affine params
        'param_quant_method': param_quant_method,
        'symmetric_params': symmetric_params,
        'param_layerwise': layerwise_params, # Renamed for clarity

        'learnable_clip': learnable_clip, # General flag for clip_val learnability
        'clip_val_init': clip_val_init,   # Initial clip_val for uniform/LSQ placeholder
    }

def _replace_module_with_quantized_version(fp32_module, quant_config):
    """Helper to replace a single FP32 module with its quantized version."""
    for fp32_type, quant_type in SUPPORTED_QUANTIZABLE_MODULES.items():
        if isinstance(fp32_module, fp32_type):
            q_module_instance = None
            # 根据类型构造Quantize版本
            if fp32_type == nn.Conv2d:
                q_module_instance = quant_type( # QuantizeConv2d
                    fp32_module.in_channels, fp32_module.out_channels, fp32_module.kernel_size,
                    stride=fp32_module.stride, padding=fp32_module.padding, dilation=fp32_module.dilation,
                    groups=fp32_module.groups, bias=fp32_module.bias is not None, padding_mode=fp32_module.padding_mode,
                    clip_val=quant_config['clip_val_init'], learnable_clip=quant_config['learnable_clip'],
                    weight_bits=quant_config['weight_bits'], input_bits=quant_config['input_bits'],
                    symmetric_weights=quant_config['symmetric_weights'], symmetric_acts=quant_config['symmetric_acts'],
                    weight_layerwise=quant_config['weight_layerwise'], input_layerwise=quant_config['input_layerwise'],
                    weight_quant_method=quant_config['weight_quant_method'], input_quant_method=quant_config['input_quant_method']
                )
            elif fp32_type == nn.ConvTranspose2d:
                q_module_instance = quant_type( # QuantizeConvTranspose2d
                    fp32_module.in_channels, fp32_module.out_channels, fp32_module.kernel_size,
                    stride=fp32_module.stride, padding=fp32_module.padding, output_padding=fp32_module.output_padding,
                    groups=fp32_module.groups, bias=fp32_module.bias is not None, dilation=fp32_module.dilation,
                    padding_mode=fp32_module.padding_mode,
                    clip_val=quant_config['clip_val_init'], learnable_clip=quant_config['learnable_clip'],
                    weight_bits=quant_config['weight_bits'], input_bits=quant_config['input_bits'],
                    symmetric_weights=quant_config['symmetric_weights'], symmetric_acts=quant_config['symmetric_acts'],
                    weight_layerwise=quant_config['weight_layerwise'], input_layerwise=quant_config['input_layerwise'],
                    weight_quant_method=quant_config['weight_quant_method'], input_quant_method=quant_config['input_quant_method']
                )
            elif fp32_type == nn.Linear:
                q_module_instance = quant_type( # QuantizeLinear
                    fp32_module.in_features, fp32_module.out_features, bias=fp32_module.bias is not None,
                    clip_val=quant_config['clip_val_init'], learnable_clip=quant_config['learnable_clip'],
                    weight_bits=quant_config['weight_bits'], input_bits=quant_config['input_bits'],
                    # Assuming QuantizeLinear uses symmetric_weights for its main 'symmetric' flag
                    symmetric_weights=quant_config['symmetric_weights'], symmetric_acts=quant_config['symmetric_acts'],
                    weight_layerwise=quant_config['weight_layerwise'], input_layerwise=quant_config['input_layerwise'],
                    weight_quant_method=quant_config['weight_quant_method'], input_quant_method=quant_config['input_quant_method']
                )
            elif fp32_type == nn.InstanceNorm2d:
                 q_module_instance = quant_type( # QuantizeInstanceNorm2d
                    fp32_module.num_features, eps=fp32_module.eps, momentum=fp32_module.momentum, 
                    affine=fp32_module.affine, track_running_stats=fp32_module.track_running_stats,
                    clip_val=quant_config['clip_val_init'], learnable_clip=quant_config['learnable_clip'],
                    param_bits=quant_config['param_bits'], input_bits=quant_config['input_bits'],
                    symmetric_params=quant_config['symmetric_params'], symmetric_acts=quant_config['symmetric_acts'],
                    param_layerwise=quant_config['param_layerwise'], input_layerwise=quant_config['input_layerwise'],
                    param_quant_method=quant_config['param_quant_method'], input_quant_method=quant_config['input_quant_method']
                )
            elif fp32_type == nn.BatchNorm2d:
                q_module_instance = quant_type( # QuantizeBatchNorm2d
                    fp32_module.num_features, eps=fp32_module.eps, momentum=fp32_module.momentum,
                    affine=fp32_module.affine, track_running_stats=fp32_module.track_running_stats,
                    clip_val=quant_config['clip_val_init'], learnable_clip=quant_config['learnable_clip'],
                    param_bits=quant_config['param_bits'], input_bits=quant_config['input_bits'],
                    symmetric_params=quant_config['symmetric_params'], symmetric_acts=quant_config['symmetric_acts'],
                    param_layerwise=quant_config['param_layerwise'], input_layerwise=quant_config['input_layerwise'],
                    param_quant_method=quant_config['param_quant_method'], input_quant_method=quant_config['input_quant_method']
                )
            
            if q_module_instance:
                # Copy weights and bias (and running_mean/var for Norm layers)
                if hasattr(fp32_module, 'weight') and fp32_module.weight is not None:
                    q_module_instance.weight.data.copy_(fp32_module.weight.data)
                if hasattr(fp32_module, 'bias') and fp32_module.bias is not None:
                    if hasattr(q_module_instance, 'bias') and q_module_instance.bias is not None:
                        q_module_instance.bias.data.copy_(fp32_module.bias.data)
                if hasattr(fp32_module, 'running_mean') and fp32_module.running_mean is not None:
                    q_module_instance.running_mean.data.copy_(fp32_module.running_mean)
                if hasattr(fp32_module, 'running_var') and fp32_module.running_var is not None:
                    q_module_instance.running_var.data.copy_(fp32_module.running_var)
                # num_batches_tracked is not typically critical for PTQ eval
                return q_module_instance
    return None # Not a supported type or error in creation


def convert_to_quantized_model(model_fp32, quant_config, 
                               target_module_paths=None, current_path=""):
    """
    Recursively replaces modules in model_fp32 with their quantized versions.
    - model_fp32: The FP32 model instance (or a submodule of it).
    - quant_config: Dictionary of quantization parameters.
    - target_module_paths: A list of dot-separated paths to modules that should be quantized.
                           If None, all supported modules are quantized.
                           E.g., ['encoder', 'decoder.conv1']
    - current_path: Used internally for recursion to track the path to the current module.
    """
    for name, child_module in model_fp32.named_children():
        child_path = f"{current_path}.{name}" if current_path else name
        
        # Decision logic: Should this child_module or its children be processed for quantization?
        process_this_child_branch = False
        if target_module_paths is None: # No specific targets, try to quantize everything supported
            process_this_child_branch = True
        else:
            for target_path in target_module_paths:
                if child_path == target_path or child_path.startswith(target_path + "."):
                    process_this_child_branch = True
                    break
        
        if process_this_child_branch and type(child_module) in SUPPORTED_QUANTIZABLE_MODULES:
            quantized_child = _replace_module_with_quantized_version(child_module, quant_config)
            if quantized_child:
                setattr(model_fp32, name, quantized_child)
                child_module = quantized_child # Update to the quantized version

        convert_to_quantized_model(child_module, quant_config, target_module_paths, current_path=child_path)

def calibrate_model_no_activation_data(model, device, quant_config):
    model.eval()
    model.to(device)

    for name, module in model.named_modules():
        if isinstance(module, (QuantizeConv2d, QuantizeConvTranspose2d, QuantizeLinear)):
            if module.weight_bits < 32 and hasattr(module, 'weight_clip_param') and module.weight_clip_param is not None:
                clip_param = module.weight_clip_param
                data_tensor = module.weight.data.detach()
                if module.weight_quant_method == "uniform":
                    if module.symmetric_weights: max_abs = data_tensor.abs().max(); clip_param.data[0], clip_param.data[1] = -max_abs, max_abs
                    else: clip_param.data[0], clip_param.data[1] = data_tensor.min(), data_tensor.max()
                elif module.weight_quant_method == "lsq" and isinstance(clip_param, LsqStepSize) and not clip_param.initialized:
                    clip_param.initialize_wrapper(data_tensor, module.weight_bits, module.symmetric_weights)
        
        if isinstance(module, (QuantizeInstanceNorm2d, QuantizeBatchNorm2d)) and module.affine:
            if module.param_bits < 32:
                if module.weight is not None and hasattr(module, 'gamma_clip_param') and module.gamma_clip_param is not None:
                    gamma_clip, gamma_data = module.gamma_clip_param, module.weight.data.detach()
                    if module.param_quant_method == "uniform":
                        if module.symmetric_params: max_abs = gamma_data.abs().max(); gamma_clip.data[0],gamma_clip.data[1] = -max_abs, max_abs
                        else: gamma_clip.data[0],gamma_clip.data[1] = gamma_data.min(), gamma_data.max()
                    elif module.param_quant_method == "lsq" and isinstance(gamma_clip, LsqStepSize) and not gamma_clip.initialized:
                        gamma_clip.initialize_wrapper(gamma_data, module.param_bits, module.symmetric_params)
                if module.bias is not None and hasattr(module, 'beta_clip_param') and module.beta_clip_param is not None:
                    beta_clip, beta_data = module.beta_clip_param, module.bias.data.detach()
                    if module.param_quant_method == "uniform":
                        if module.symmetric_params: max_abs = beta_data.abs().max(); beta_clip.data[0],beta_clip.data[1] = -max_abs, max_abs
                        else: beta_clip.data[0],beta_clip.data[1] = beta_data.min(), beta_data.max()
                    elif module.param_quant_method == "lsq" and isinstance(beta_clip, LsqStepSize) and not beta_clip.initialized:
                        beta_clip.initialize_wrapper(beta_data, module.param_bits, module.symmetric_params)

    if quant_config['input_bits'] < 32:
        for name, module in model.named_modules():
            if isinstance(module, (QuantizeConv2d, QuantizeConvTranspose2d, QuantizeLinear, QuantizeInstanceNorm2d, QuantizeBatchNorm2d)):
                if module.input_bits < 32 and module.input_quant_method == "lsq" and hasattr(module, 'input_clip_param'):
                    clip_param = module.input_clip_param
                    if isinstance(clip_param, LsqStepSize) and not clip_param.initialized:
                        # LSQ alpha for activations needs initialization. Since we have no data, use a generic range.
                        # This is highly suboptimal for LSQ but necessary if no calibration data.
                        default_init_range = quant_config.get('clip_val_init', 2.5)
                        dummy_act_tensor = torch.tensor([-default_init_range, default_init_range], device=device) if module.symmetric_acts else torch.tensor([0, default_init_range], device=device)
                        clip_param.initialize_wrapper(dummy_act_tensor, module.input_bits, module.symmetric_acts)