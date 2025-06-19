# Imports
import os
import numpy as np
import time

# PyTorch
import torch
from torch import Tensor
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

# Groups
from .groups import Group

# Kernels
from .kernels import InterpolativeLiftingKernel, InterpolativeGroupKernel



####################
# Includes the various equivariant layers used to implement Group Equivariant CNNs.
####################



# Lifting Layer
class LiftingConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int, # number of input feature channels (e.g., 3 for RGB)
        out_channels: int, # number of output channels per group element h
        kernel_size: int,
        group: Group, # custom group object implementing H
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        padding_mode: str = 'zeros'
    ) -> None:
        super().__init__()

        self.kernel = InterpolativeLiftingKernel(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            group=group
        )

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.num_group_elements = group.elements().shape[0]

        # Enable bias parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))  # only one bias per G-feature map
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)


    def forward(self, input: Tensor) -> Tensor:
        # Sample convolution kernels
        conv_kernels = self.kernel.sample()

        # Convolve
        input = F.conv2d(
            input=input, # [C_out, |H|, C_in, kernel_size, kernel_size]
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.num_group_elements,
                self.kernel.in_channels,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ), # [C_out * |H|, C_in, kernel_size, kernel_size]
            padding=self.padding,
            stride=self.stride,
            bias=None
        )

        #print(input.shape)

        # Reshape to ensure compatibility with Conv2d
        input = input.view(
            -1,
            self.kernel.out_channels,
            self.num_group_elements,
            input.shape[-1],
            input.shape[-2]
        )

        # Add bias term
        if self.bias is not None:
            input += self.bias.view(1, -1, 1, 1, 1)  # broadcast over G, H, W

        return input


# Group Convolution Layer
class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int, # number of input feature channels (e.g., 3 for RGB)
        out_channels: int, # number of output channels per group element h
        kernel_size: int,
        group: Group, # custom group object implementing H
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None:
        super().__init__()

        # Get kernel
        self.kernel = InterpolativeGroupKernel(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            group=group
        )

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.num_group_elements = group.elements().shape[0]

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))  # one per G-feature map
            fan_in = in_channels * kernel_size * kernel_size * self.num_group_elements
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)


    def forward(self, input: Tensor) -> Tensor:
        # Fold the group dim of the input into the input channel dim
        input = input.reshape(
            -1,
            input.shape[1] * input.shape[2], # C_in * |H|
            input.shape[3],
            input.shape[4]
        )

        # Sample convolution kernels
        conv_kernels = self.kernel.sample()

        # Convolve
        input = F.conv2d(
            input=input,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.num_group_elements,
                self.kernel.in_channels * self.num_group_elements,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding,
            stride=self.stride,
            bias=None
        )

        input = input.view(
            -1,
            self.kernel.out_channels,
            self.num_group_elements,
            input.shape[-1],
            input.shape[-2]
        )

        if self.bias is not None:
            input += self.bias.view(1, -1, 1, 1, 1)  # [B, C_out, G, H, W]

        return input