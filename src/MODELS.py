## MODELS.py ##

# PyTorch
from torch import Tensor
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

# e2cnn
from e2cnn import gspaces
from e2cnn import nn as e2nn

from e2cnn.group import SO2


# Groups
from subfiles.groups import Group

# Layers
from subfiles.layers import LiftingConv2d, GroupConv2d



class GECNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            num_hidden: int,
            hidden_channels: int,
            padding: int=None,
            group: Group,
            bias: bool=True
        ) -> None:
        super().__init__()
        padding = kernel_size // 2 if padding == None else padding

        # Lifting convolution layer
        self.lifting_conv = LiftingConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            group=group,
            bias=bias
        )

        # Set of group convolutions
        self.gconvs = nn.ModuleList()

        for _ in range(num_hidden):
            self.gconvs.append(
                GroupConv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    group=group,
                    bias=bias
                )
            )
        
        # Projection layer
        self.projection_layer = nn.AdaptiveAvgPool3d(output_size=1)

        # Final linear layer for classification
        self.final_linear = nn.Linear(
            in_features=hidden_channels,
            out_features=out_channels
        )

    def forward(self, input: Tensor) -> Tensor:
        # Lift and disentangle features in the input
        input = self.lifting_conv(input)
        input = F.layer_norm(
            input,
            input.shape[-4:]
        )
        input = F.relu(input=input)

        # Apply group convolutions
        for gconv in self.gconvs:
            input = gconv(input)
            input = F.layer_norm(input, input.shape[-4:])
            input = F.relu(input=input)

        # Ensure equivariance, apply pooling over group and spatial dims
        input = self.projection_layer(input).squeeze()

        input = self.final_linear(input)

        return input



class CNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            num_hidden: int,
            hidden_channels: int,
            padding: int=None,
            bias: bool = True,
        ) -> None:
        super().__init__()
        padding = kernel_size // 2 if padding == None else padding

        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

        self.convs = nn.ModuleList()
        for _ in range(num_hidden):
            self.convs.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias
                )
            )

        self.final_linear = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):

        x = self.first_conv(x)
        x = F.layer_norm(x, x.shape[-3:])
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x)
            x = F.layer_norm(x, x.shape[-3:])
            x = F.relu(x)
        
        # Apply average pooling over remaining spatial dimensions.
        x = F.adaptive_avg_pool2d(x, 1).squeeze()

        x = self.final_linear(x)
        return x
