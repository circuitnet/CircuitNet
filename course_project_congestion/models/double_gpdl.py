# Copyright 2022 CircuitNet. All rights reserved. (Enhanced version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict, Any

# --- Helper Functions ---
def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys


# --- Core Building Blocks ---

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channel: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # Broadcast to match input dimensions

class ASPPConv(nn.Sequential):
    """Convolutional block for ASPP with specified dilation."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

class ASPPPooling(nn.Module):
    """Image Pooling branch for ASPP."""
    def __init__(self, in_channels: int, out_channels: int, pool_target_size: Tuple[int, int] = (4, 4)):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_target_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        pooled_features = self.pool_conv(x)
        return F.interpolate(pooled_features, size=target_size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module."""
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int] = [6, 12, 18]):
        super().__init__()
        
        num_atrous_branches = len(atrous_rates)
        num_total_branches = num_atrous_branches + 2  # + 1x1 conv + image pooling
        
        # Ensure branch_channels is at least 1, avoid division by zero if out_channels is small
        branch_channels = max(1, out_channels // num_total_branches)
        
        # If out_channels is not perfectly divisible, adjust the first branch to compensate
        # This helps ensure the projected output matches `out_channels` more closely.
        first_branch_channels = branch_channels + (out_channels % num_total_branches)

        branches = []
        # 1x1 convolution branch
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, first_branch_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(first_branch_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # Atrous convolution branches
        for rate in atrous_rates:
            branches.append(ASPPConv(in_channels, branch_channels, rate))

        # Image pooling branch
        branches.append(ASPPPooling(in_channels, branch_channels))
        
        self.convs = nn.ModuleList(branches)

        project_in_channels = first_branch_channels + (num_atrous_branches + 1) * branch_channels
        self.project = nn.Sequential(
            nn.Conv2d(project_in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.05)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [conv_branch(x) for conv_branch in self.convs]
        concatenated_features = torch.cat(branch_outputs, dim=1)
        return self.project(concatenated_features)

class ConvBlock(nn.Module):
    """Standard double convolution block, optionally with Squeeze-and-Excitation and Dropout."""
    def __init__(self, dim_in: int, dim_out: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = True, use_se: bool = False, se_reduction: int = 8,
                 use_dropout: bool = False, dropout_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.norm1 = nn.InstanceNorm2d(dim_out, affine=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.se_block = SEBlock(dim_out, reduction=se_reduction) if use_se else None
        self.dropout = nn.Dropout2d(dropout_rate) if use_dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
            
        if self.se_block:
            x = self.se_block(x)

        if self.dropout:
            x = self.dropout(x)
        return x

class UpConvBlock(nn.Sequential):
    """Upconvolution block using ConvTranspose2d."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

# --- U-Net Components ---

class Encoder(nn.Module):
    def __init__(self,
                 in_dim: int = 3,
                 base_channels: int = 32,
                 num_stages: int = 2,
                 bottleneck_channels_factor: float = 2.0,
                 use_se_in_stages: Union[bool, List[bool]] = False,
                 se_reduction: int = 8,
                 use_aspp_bottleneck: bool = False,
                 aspp_rates: Optional[List[int]] = None,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self._skip_channels: List[int] = []

        current_channels = in_dim
        
        if isinstance(use_se_in_stages, bool):
            use_se_per_stage = [use_se_in_stages] * num_stages
        elif isinstance(use_se_in_stages, list) and len(use_se_in_stages) == num_stages:
            use_se_per_stage = use_se_in_stages
        else:
            raise ValueError("use_se_in_stages must be a bool or a list of bools matching num_stages")

        for i in range(num_stages):
            stage_out_channels = base_channels * (2**i)
            self.encoder_blocks.append(
                ConvBlock(current_channels, stage_out_channels,
                          use_se=use_se_per_stage[i], se_reduction=se_reduction,
                          use_dropout=use_dropout, dropout_rate=dropout_rate)
            )
            self._skip_channels.append(stage_out_channels)
            if i < num_stages - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = stage_out_channels
        
        self.final_pool_for_bottleneck = nn.MaxPool2d(kernel_size=2, stride=2)
        bottleneck_in_channels = current_channels
        self.final_bottleneck_channels = int(bottleneck_in_channels * bottleneck_channels_factor)
            
        if use_aspp_bottleneck:
            aspp_rates_default = [6, 12, 18] if aspp_rates is None else aspp_rates
            self.bottleneck_processor = ASPP(bottleneck_in_channels, self.final_bottleneck_channels, aspp_rates_default)
        else:
            use_se_in_bottleneck_block = use_se_per_stage[-1]
            self.bottleneck_processor = ConvBlock(bottleneck_in_channels, self.final_bottleneck_channels,
                                                  use_se=use_se_in_bottleneck_block, se_reduction=se_reduction,
                                                  use_dropout=use_dropout, dropout_rate=dropout_rate * 1.5)  # 瓶颈层用更高dropout
        
        self.bottleneck_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.pools): # Apply pooling if not the last encoder block
                x = self.pools[i](x)
        
        # Pool before bottleneck processing
        x_pooled_for_bottleneck = self.final_pool_for_bottleneck(x)
        bottleneck_features_processed = self.bottleneck_processor(x_pooled_for_bottleneck)
        bottleneck_output = self.bottleneck_activation(bottleneck_features_processed)
        
        return bottleneck_output, skips[::-1]  # Reverse skips to match decoder order

class Decoder(nn.Module):
    """U-Net style decoder."""
    def __init__(self,
                 out_channels_final: int,
                 bottleneck_channels_in: int,
                 skip_channels_list: List[int],
                 decoder_upconv_out_channels_list: Optional[List[int]] = None,
                 use_se_in_stages: Union[bool, List[bool]] = False,
                 se_reduction: int = 8,
                 final_activation_type: Optional[str] = 'sigmoid',
                 use_dropout: bool = False,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        num_decoder_stages = len(skip_channels_list)
        
        self.upconvs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        current_channels_dec = bottleneck_channels_in
        
        # Determine SE usage per stage
        if isinstance(use_se_in_stages, bool):
            use_se_per_stage = [use_se_in_stages] * num_decoder_stages
        elif isinstance(use_se_in_stages, list) and len(use_se_in_stages) == num_decoder_stages:
            use_se_per_stage = use_se_in_stages
        else:
            raise ValueError("use_se_in_stages must be a bool or a list matching num_decoder_stages")

        for i in range(num_decoder_stages):
            skip_ch = skip_channels_list[i]
            
            # Determine upconv output channels for this stage
            if decoder_upconv_out_channels_list and i < len(decoder_upconv_out_channels_list):
                upconv_out_ch = decoder_upconv_out_channels_list[i]
            else: # Default logic: try to match skip connection channels, or halve current
                upconv_out_ch = skip_ch if skip_ch > 0 else current_channels_dec // 2
            upconv_out_ch = max(16, upconv_out_ch) # Ensure a minimum number of channels

            self.upconvs.append(UpConvBlock(current_channels_dec, upconv_out_ch))
            
            conv_in_ch = upconv_out_ch + skip_ch
            # The conv_block in decoder typically reduces channels back to upconv_out_ch or similar
            conv_out_ch = upconv_out_ch 
            self.conv_blocks.append(
                ConvBlock(conv_in_ch, conv_out_ch,
                          use_se=use_se_per_stage[i], se_reduction=se_reduction,
                          use_dropout=use_dropout, dropout_rate=dropout_rate)
            )
            current_channels_dec = conv_out_ch
            
        self.final_conv = nn.Conv2d(current_channels_dec, out_channels_final, kernel_size=1)
        
        if final_activation_type == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation_type == 'softmax':
            self.final_activation = nn.Softmax(dim=1) # Ensure dim is correct for your task
        elif final_activation_type is None or final_activation_type.lower() == 'none':
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported final_activation_type: {final_activation_type}")

    def forward(self, bottleneck_features: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = bottleneck_features
        
        for i in range(len(self.conv_blocks)):
            x = self.upconvs[i](x)
            
            if i < len(skips):
                skip_connection = skips[i]
                # Ensure spatial dimensions match for concatenation
                if x.shape[-2:] != skip_connection.shape[-2:]:
                    x = F.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip_connection], dim=1)
            
            x = self.conv_blocks[i](x)
            
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

# --- Main GPDL Model ---

class GPDL(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 2,
                 encoder_base_channels: int = 32,
                 encoder_num_stages: int = 2,
                 encoder_bottleneck_channels_factor: float = 2.0,
                 encoder_use_se_in_stages: Union[bool, List[bool]] = False,
                 encoder_se_reduction: int = 8,
                 encoder_use_aspp_bottleneck: bool = False,
                 encoder_aspp_rates: Optional[List[int]] = None,
                 decoder_upconv_out_channels_list: Optional[List[int]] = None,
                 decoder_use_se_in_stages: Union[bool, List[bool]] = False,
                 decoder_se_reduction: int = 8,
                 final_activation: Optional[str] = 'sigmoid',
                 use_dropout: bool = False,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__()
        
        self.encoder = Encoder(
            in_dim=in_channels,
            base_channels=encoder_base_channels,
            num_stages=encoder_num_stages,
            bottleneck_channels_factor=encoder_bottleneck_channels_factor,
            use_se_in_stages=encoder_use_se_in_stages,
            se_reduction=encoder_se_reduction,
            use_aspp_bottleneck=encoder_use_aspp_bottleneck,
            aspp_rates=encoder_aspp_rates,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate
        )
        
        self.decoder = Decoder(
            out_channels_final=out_channels,
            bottleneck_channels_in=self.encoder.final_bottleneck_channels,
            skip_channels_list=self.encoder._skip_channels[::-1],
            decoder_upconv_out_channels_list=decoder_upconv_out_channels_list,
            use_se_in_stages=decoder_use_se_in_stages,
            se_reduction=decoder_se_reduction,
            final_activation_type=final_activation,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate * 0.8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_output, skip_connections = self.encoder(x)
        return self.decoder(bottleneck_output, skip_connections)

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
# --- Double GPDL Structure ---
class DoubleGPDL(nn.Module):
    """
    Double GPDL structure where the output of the first GPDL (net1)
    modulates the input to the second GPDL (net2).
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 2, # Final output channels from net2
                 net1_params: Optional[Dict[str, Any]] = None,
                 net2_params: Optional[Dict[str, Any]] = None,
                 # For auxiliary loss, if net1_out is also returned
                 return_net1_output_on_train: bool = False,
                 **kwargs):
        super().__init__()
        self.return_net1_output_on_train = return_net1_output_on_train

        # --- High-performance default configurations ---
        default_net1_config = {
            'out_channels': 1,
            'encoder_base_channels': 48,                     # Increased to 48 (from 32)
            'encoder_num_stages': 3,                         # Increased to 3 stages (from 2)
            'encoder_bottleneck_channels_factor': 1.5,       # Increased bottleneck capacity (from 0.5)
            'encoder_use_se_in_stages': [False, True, True], # Use SE in deeper layers
            'encoder_se_reduction': 4,                       # Stronger SE expression (from 8)
            'encoder_use_aspp_bottleneck': False,            # Keep Net1 simple, focus on mask generation
            'decoder_use_se_in_stages': [False, False, True], # Only use SE in final stage
            'decoder_se_reduction': 4,
            'decoder_upconv_out_channels_list': [96, 48, 24], # Match 3-stage structure
            'final_activation': 'sigmoid',
        }

        default_net2_config = {
            'out_channels': out_channels,
            'encoder_base_channels': 64,                     # Significantly increased to 64 (from 32)
            'encoder_num_stages': 3,                         # Increased to 3 stages
            'encoder_bottleneck_channels_factor': 2.0,       # Strong bottleneck expression (from 0.5)
            'encoder_use_se_in_stages': [False, True, True], # Use SE attention in deeper layers
            'encoder_se_reduction': 4,                       # Strong SE expression
            'encoder_use_aspp_bottleneck': True,             # Use ASPP for multi-scale enhancement
            'encoder_aspp_rates': [2, 4, 8, 16],            # Richer dilation rates
            'decoder_use_se_in_stages': [False, True, True], # Use SE in deeper decoder stages
            'decoder_se_reduction': 4,
            'decoder_upconv_out_channels_list': [128, 64, 32], # High-capacity decoder
            'final_activation': 'sigmoid',
        }

        # --- Apply user-provided params over defaults ---
        actual_net1_params = default_net1_config.copy()
        if net1_params:
            actual_net1_params.update(net1_params)
        # Ensure net1's out_channels is suitable for a mask if not specified for that purpose
        if 'out_channels' not in actual_net1_params or actual_net1_params.get('out_channels', 1) != 1:
            print(f"Warning: DoubleGPDL net1 out_channels is {actual_net1_params.get('out_channels')}. "
                  "Typically 1 for mask generation. Overriding to 1.")
            actual_net1_params['out_channels'] = 1


        actual_net2_params = default_net2_config.copy()
        if net2_params:
            actual_net2_params.update(net2_params)
        actual_net2_params['out_channels'] = out_channels # Crucial: ensure net2 produces final num channels

        # print(f"DoubleGPDL Initializing net1 with params: {actual_net1_params}")
        self.net1 = GPDL(in_channels=in_channels, **actual_net1_params)
        
        # print(f"DoubleGPDL Initializing net2 with params: {actual_net2_params}")
        # net2's input channels remain the same as the original input,
        # as it's element-wise multiplied by net1's mask.
        self.net2 = GPDL(in_channels=in_channels, **actual_net2_params)

    def forward(self, x_original: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out_net1_mask = self.net1(x_original)
        
        # Ensure mask is broadcastable if x_original has more channels than mask
        if x_original.shape[1] > out_net1_mask.shape[1] and out_net1_mask.shape[1] == 1:
            input_net2 = x_original * out_net1_mask # Broadcasting handles this
        else: # If mask has same channels or more (unusual for mask), direct multiplication
            input_net2 = x_original * out_net1_mask
            
        out_net2_final = self.net2(input_net2)
        
        if self.training and self.return_net1_output_on_train:
            return out_net1_mask, out_net2_final
        return out_net2_final

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
        
# --- Double GPDL distill Structure ---
class DoubleGPDLdistill(nn.Module):
    """
    Double GPDL structure where the output of the first GPDL (net1)
    modulates the input to the second GPDL (net2).
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 2, # Final output channels from net2
                 net1_params: Optional[Dict[str, Any]] = None,
                 net2_params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__()

        # --- Default configurations for net1 and net2 ---
        default_net1_config = {
            'out_channels': 1,
            'encoder_base_channels': 16,
            'encoder_num_stages': 3,
            'encoder_bottleneck_channels_factor': 1.0,
            'encoder_use_se_in_stages': [True, True, True],
            'encoder_se_reduction': 4,
            'encoder_use_aspp_bottleneck': True,
            'encoder_aspp_rates': [6, 12, 18],
            'decoder_use_se_in_stages': [True, True, True],
            'decoder_se_reduction': 4,
            'decoder_upconv_out_channels_list': None,
            'final_activation': 'sigmoid',
            'use_dropout': False,
        }

        default_net2_config = {
            'out_channels': out_channels,
            'encoder_base_channels': 32,
            'encoder_num_stages': 2,
            'encoder_bottleneck_channels_factor': 1.0,
            'encoder_use_se_in_stages': [True, True],
            'encoder_se_reduction': 8,
            'encoder_use_aspp_bottleneck': True,
            'encoder_aspp_rates': [6, 12, 18],
            'decoder_use_se_in_stages': [True, True],
            'decoder_se_reduction': 8,
            'decoder_upconv_out_channels_list': None,
            'final_activation': 'sigmoid',
            'use_dropout': True,
            'dropout_rate': 0.2,
        }

        # --- Apply user-provided params over defaults ---
        actual_net1_params = default_net1_config.copy()
        if net1_params:
            actual_net1_params.update(net1_params)
        # Ensure net1's out_channels is suitable for a mask if not specified for that purpose
        if 'out_channels' not in actual_net1_params or actual_net1_params.get('out_channels', 1) != 1:
            print(f"Warning: DoubleGPDL net1 out_channels is {actual_net1_params.get('out_channels')}. "
                  "Typically 1 for mask generation. Overriding to 1.")
            actual_net1_params['out_channels'] = 1


        actual_net2_params = default_net2_config.copy()
        if net2_params:
            actual_net2_params.update(net2_params)
        actual_net2_params['out_channels'] = out_channels # Crucial: ensure net2 produces final num channels

        # print(f"DoubleGPDL Initializing net1 with params: {actual_net1_params}")
        self.net1 = GPDL(in_channels=in_channels, **actual_net1_params)
        
        # print(f"DoubleGPDL Initializing net2 with params: {actual_net2_params}")
        # net2's input channels remain the same as the original input,
        # as it's element-wise multiplied by net1's mask.
        self.net2 = GPDL(in_channels=in_channels, **actual_net2_params)

        self.fusion_weight = nn.Parameter(torch.tensor(0.99))

    def forward(self, x_original: torch.Tensor) -> torch.Tensor:
        attention_weights = self.net1(x_original)  # net1 produces a mask

        fusion_strength = torch.sigmoid(self.fusion_weight)  # sigmoid to ensure values are in [0, 1]
        enhanced_input = x_original * (attention_weights * fusion_strength + (1 - fusion_strength))

        output = self.net2(enhanced_input)  # net2 processes the modulated input
        
        return output

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained )}.')


if __name__ == '__main__':
    import time
    
    def get_param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("="*50)
    print("GPDL Model Testing")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_input = torch.randn(16, 3, 256, 256).to(device)
    print(f"Input shape: {test_input.shape}")

    # --- Test DoubleGPDL with various configurations ---
    print("\n--- Testing DoubleGPDL ---")
    
    try: 
        model = DoubleGPDL().to(device)
        print(model)
        model.init_weights()
        model.train()

        params = get_param_count(model)
        print(f"Total parameters: {params:,}")

        start_time = time.time()
        output = model(test_input)
        elapsed_time = time.time() - start_time

        print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else 'Tuple'}")
        print(f"Forward time: {elapsed_time:.4f}s")
        print("Status: OK")

        from torchviz import make_dot
        graph = make_dot(output, params=dict(model.named_parameters()))
        graph.render("double_gpdl_model", format="pdf", cleanup=True)

        # import netron
        # modelData = "./demo.onnx"
        # torch.onnx.export(model, test_input, modelData)
        # netron.start(modelData)  


    except Exception as e:
        print(f"Status: FAILED - {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("Model Testing completed!")
    print("="*50)