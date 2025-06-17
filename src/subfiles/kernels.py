## subfiles/kernels.py ##

# Imports
import numpy as np

# PyTorch
import torch
import torch.nn as nn

# Groups
from .groups import Group

# Interpolation
from .utils import bilinear_interpolation, trilinear_interpolation



# Lifting kernel base (optimized)
class LiftingKernel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            group: Group
        ) -> None:
        super().__init__()

        # Save config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group = group
        self.num_group_elements = self.group.elements().numel() # |H|
        self.device = self.group.identity.device # reiable way to get device

        # Create normalized 2D from -1 to 1
        grid_1d = torch.linspace(start=-1., end=1., steps=self.kernel_size, device=self.device)
        grid_2d = torch.stack(
            torch.meshgrid(
                grid_1d, # direction i
                grid_1d, # direction j
                indexing="ij"
            )
        )

        # Register buffers (not trainable)
        self.register_buffer(name="kernel_grid", tensor=grid_2d)
        self.register_buffer(name="transformed_grid", tensor=self.transform_grid()) # Transform the grid by group elements h ∈ H

    def transform_grid(self):
        elements = self.group.elements()  # (|H|,)
        h_inv = self.group.inverse(elements)  # (|H|,)

        # Batched transform: (2, |H|, H, W)
        return self.group.left_regular_representation(h_inv, self.kernel_grid)
    
    def sample(self, elements):
        """
        Sample convolution kernels using group elements h ∈ H.

        Returns kernels, i.e. a kernel bank (stack of kernels) over all input channels,
        containing kernels transformed for all output group elements.
        """
        raise NotImplementedError()



# Interpolative lifting kernel, able to interpolate pixels/weights in transformations that are unfit
# for the frame (e.g. rotations)
class InterpolativeLiftingKernel(LiftingKernel):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            group
        ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, group)
        
        # Setup empty kernel weight parameter
        self.weight = nn.Parameter(
            data=torch.empty((
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ), device=self.device),
            requires_grad=True
        ) # (C_out, C_in, kernel_size, kernel_size)

        # Initialize weights using Kaiming uniform initialization (same as default Conv2d)
        nn.init.kaiming_uniform_(
            tensor=self.weight.data,
            a=np.sqrt(5)
        )
    
    def sample(self):
        # Combine channels into one dimension
        weight = self.weight.view(
            self.in_channels * self.out_channels,
            self.kernel_size,
            self.kernel_size
        )  # [C_combined, H, W]

        signal = weight.unsqueeze(0).repeat(
            self.num_group_elements,
            1,
            1,
            1
        )

        # transformed_grid: [2, |H|, H, W] → [|H|, 2, H, W]
        grid = self.transformed_grid.permute(1, 0, 2, 3)  # [|H|, 2, H, W]

        # Apply interpolation over batched signal/grid
        sampled = bilinear_interpolation(signal, grid)  # [|H|, C_combined, H, W]

        # Reshape and reorder to match expected kernel format
        sampled = sampled.view(
            self.num_group_elements,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )  # [|H|, C_out, C_in, H, W]

        return sampled.transpose(0, 1)  # [C_out, |H|, C_in, H, W]



# Base group convolution kernel
class GroupKernel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            group: Group
        ) -> None:
        super().__init__()
        
        # Save config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group = group
        self.num_group_elements = self.group.elements().numel() # |H|
        self.device = self.group.identity.device # reiable way to get device

        # Create normalized 2D from -1 to 1
        grid_1d = torch.linspace(start=-1., end=1., steps=self.kernel_size, device=self.device)
        grid_2d = torch.stack(
            torch.meshgrid(
                grid_1d, # direction i
                grid_1d, # direction j
                indexing="ij"
            )
        )

        # Register buffers (not trainable)
        self.register_buffer(name="grid_R2", tensor=grid_2d)                                      # ℝ² grid
        self.register_buffer(name="grid_H", tensor=self.group.elements())                         # group elements
        self.register_buffer(name="transformed_affine_grid", tensor=self.transform_affine_grid()) # ℝ² ⋊ H
    
    # Grid: ℝ² ⋊ H
    def transform_affine_grid(self):
        elements = self.group.elements() # all h ∈ H
        h_inv = self.group.inverse(elements) # all h⁻¹ ∈ H

        # Transform ℝ² with left regular action: result shape (2, |H|, H, W)
        transformed_grid_R2 = self.group.left_regular_representation(h=h_inv, x=self.grid_R2)

        #print("h_inv before product:", h_inv)
        #print("self.grid_H before product:", self.grid_H)

        # Transform group grid H: compute full pairwise product table (|H|, |H|)
        #h_inv = h_inv.view(-1, 1)
        #h_grid = self.grid_H.view(1, -1)
        #transformed_grid_H = self.group.product(
        #    h_inv.expand(-1, self.num_group_elements),
        #    h_grid.expand(self.num_group_elements, -1)
        #)
        transformed_grid_H = self.group.product(
            h_inv.unsqueeze(-1),   # Shape (|H|, 1)
            self.grid_H.unsqueeze(0) # Shape (1, |H|)
        )

        # Transpose the (|H|, |H|) matrix to get the right order
        transformed_grid_H = transformed_grid_H.T 

        # Normalize to please PyTorch
        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)

        # Reshape and expand H part: (1, |H|, |H|, H, W)
        transformed_grid_H = transformed_grid_H.view(
            1, self.num_group_elements, self.num_group_elements, 1, 1
        ).expand(-1, -1, -1, self.kernel_size, self.kernel_size)

        # Reshape and expand R2 part: (2, |H|, |H|, H, W)
        transformed_grid_R2 = transformed_grid_R2.unsqueeze(2).expand(
            -1, self.num_group_elements, self.num_group_elements, -1, -1
        )

        # Concatenate along first dim: final shape (3, |H|, |H|, H, W)
        transformed_grid = torch.cat(
            [transformed_grid_R2, transformed_grid_H], dim=0
        )

        return transformed_grid

    def sample(self, sampled_group_elements):
        raise NotImplementedError()
    


# Group convolutional kernel with interpolation
class InterpolativeGroupKernel(GroupKernel):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            group: Group
        ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, group)
        
        # Initialize kernel weights
        self.weight = nn.Parameter(
            data=torch.empty((
                out_channels,
                in_channels,
                self.num_group_elements, # add group elements as well
                kernel_size,
                kernel_size
            ), device=self.device),
            requires_grad=True
        ) # (C_out, C_in, kernel_size, kernel_size)

        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(
            tensor=self.weight.data,
            a=np.sqrt(5)
        )
    
    # Sample group convolution kernels
    def sample(self):
        # Fold the output channel dim into the input channel dim
        # this enables us to transform the entire kernel bank in one go using the
        # torch grid_sample function
        weight = self.weight.view(
            self.in_channels * self.out_channels,
            self.num_group_elements,
            self.kernel_size,
            self.kernel_size
        ) # (C_in * C_in, kernel_size, kernel_size)

        signal_for_trilinear = weight.unsqueeze(0).expand(
            self.num_group_elements,
            -1, -1, -1, -1
        )

        transformed_weight = trilinear_interpolation(
            signal=signal_for_trilinear,
            grid=self.transformed_affine_grid
        )
    
        # Separate input and output channels, reshape back
        transformed_weight = transformed_weight.view(
            self.num_group_elements,
            self.out_channels,
            self.in_channels,
            self.num_group_elements,
            self.kernel_size,
            self.kernel_size
        ) # (|H|, C_out, C_in, |H|, kernel_size, kernel_size)

        # To make it work with Conv2d, transpose it so that out_channel dim is before group dim
        transformed_weight = transformed_weight.transpose(0, 1) # (C_out, |H|, C_in, |H|, kernel_size, kernel_size)

        return transformed_weight