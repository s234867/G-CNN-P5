## subfiles/kernels.py ##

# Imports
import math

# PyTorch
import torch
import torch.nn as nn

# Groups
from .groups import Group

# Interpolation
from .utils import bilinear_interpolation, trilinear_interpolation



####################


# Base kernel
class BaseKernel(nn.Module):
    """
    Base class for convolution kernels, with initialization and grid creation.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            group: Group
        ) -> None:
        super().__init__()

        # Save config
        self.in_channels = in_channels                           # C_in
        self.out_channels = out_channels                         # C_out
        self.kernel_size = kernel_size                           # K
        self.group = group
        self.num_group_elements = self.group.elements().shape[0] # |H|
        self.device = self.group.device

        # Initialize spatial grid
        self.register_buffer(name="grid_R2", tensor=self._init_grids()) # ℝ² grid

    def _init_grids(self) -> torch.Tensor:
        """
        Initializes common spatial grids for the kernel.
        """
        grid_1d = torch.linspace(start=-1., end=1., steps=self.kernel_size, device=self.device)
        grid_2d = torch.stack(
            torch.meshgrid(
                grid_1d, # direction i
                grid_1d, # direction j
                indexing="ij"
            )
        ) # (2, K, K)

        return grid_2d

    def sample(self) -> torch.Tensor:
        """
        Samples convolution kernels. Must be implemented by subclasses.
        """
        raise NotImplementedError()


# Lifting kernel
class LiftingKernel(BaseKernel):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            group: Group
        ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, group)

        # Transform the grid by the left-regular action of group H
        self.register_buffer(name="transformed_grid", tensor=self.transform_grid())

    def transform_grid(self) -> torch.Tensor:
        """
        Apply inverse group elements h⁻¹ to the spatial grid via left-regular representation.

        returns:
            Transformed grid of shape (2, |H|, k, k), where 2 = spatial dim.
        """
        elements = self.group.elements()  # (|H|,)
        h_inv = self.group.inverse(elements)  # (|H|,)

        # Batched transform
        batched_transform = self.group.left_regular_representation(h_inv, self.grid_R2) # (|H|, 2, K, K)

        # Tranpose to get the correct shape for PyTorch Conv2d (batch first)
        # (|H|, 2, K, K) -> (2, |H|, K, K)
        batched_transform = batched_transform.transpose(0, 1)

        return batched_transform
    
    def sample(self) -> torch.Tensor:
        """
        Sample convolution kernels using group elements h ∈ H.

        Returns kernels, i.e. a kernel bank (stack of kernels) over all input channels,
        containing kernels transformed for all output group elements.
        """
        raise NotImplementedError()


class InterpolativeLiftingKernel(LiftingKernel):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            group
        ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, group)
        
        # Learnable base kernel (before transformation)
        self.weight = nn.Parameter(
            data=torch.empty((
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ), device=self.device),
            requires_grad=True
        ) # (C_out, C_in, K, K)

        # Kaiming initialization (like Conv2d)
        nn.init.kaiming_uniform_(tensor=self.weight.data, a=math.sqrt(5))
    
    def sample(self):
        """
        Interpolates the base kernel over transformed group grids to produce group-equivariant lifted kernels.
        """
        # Flatten channel dims for joint interpolation
        # enables us to transform the entire kernel bank simultaneously using grid_sample
        weight = self.weight.view(
            self.in_channels * self.out_channels,
            self.kernel_size,
            self.kernel_size
        )  # (C_in, C_out, K, K) -> (C_in * C_out, K, K)

        # Transpose grid: (|H|, 2, K, K) -> (|H|, K, K, 2)
        grid = self.transformed_grid

        # Interpolate transformed grid
        sampled = bilinear_interpolation(weight, grid)

        # Reshape and reorder to match expected kernel format
        sampled = sampled.view(
            self.num_group_elements,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )  # (|H|, C_out, C_in, K, K)

        return sampled.transpose(0, 1) # (C_out, |H|, C_in, K, K)


# Base group convolution kernel
class GroupKernel(BaseKernel):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            group: Group
        ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, group)

        # Register buffers (not trainable)
        self.register_buffer(name="grid_H", tensor=self.group.elements())                         # H grid
        self.register_buffer(name="transformed_affine_grid", tensor=self.transform_affine_grid()) # ℝ² ⋊ H grid

    def transform_affine_grid(self):
        elements = self.group.elements() # all h ∈ H
        h_inv = self.group.inverse(elements) # all h⁻¹ ∈ H

        ### Spatial part

        # Transform ℝ² (spatial part) with left regular action
        # (|H|, 2, K, K)
        transformed_grid_R2 = self.group.left_regular_representation(h=h_inv, x=self.grid_R2)

        # Add an extra dimension for the second group element and expand it
        # (|H|, 2, 1, K, K)
        transformed_grid_R2 = transformed_grid_R2.unsqueeze(2) 

        # Expand to (|H|, 2, |H|, K, K)
        transformed_grid_R2 = transformed_grid_R2.expand(-1, -1, self.num_group_elements, -1, -1)

        # Permute to bring the spatial dimensions to the front: (2, |H|, |H|, K, K)
        transformed_grid_R2 = transformed_grid_R2.permute(1, 0, 2, 3, 4) 


        ### Group part

        # Compute pairwise product table (circulant matrix)
        # (|H|, |H|, group_dim)
        transformed_grid_H = self.group.product(
            h_inv.unsqueeze(1), # (|H|, 1, D) or (|H|, 1)
            self.grid_H.unsqueeze(0) # (1, |H|, D) or (1, |H|)
        ) # -> (|H|, |H|, D) or (|H|, |H|)
        
        # For CyclicGroup, ensure it has a last dimension for group.dimension=1
        if self.group.dimension == 1 and transformed_grid_H.ndim == 2:
            transformed_grid_H = transformed_grid_H.unsqueeze(-1) # (|H, |H|, 1)
        
        # Transpose grid
        transformed_grid_H = transformed_grid_H.transpose(0, 1)

        # Normalize group elements so they fit with normalized ℝ² grid
        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)
        
        # Permute to bring the group dimensions to the front: (group.dimension, |H|, |H|)
        transformed_grid_H = transformed_grid_H.permute(2, 0, 1)

        # Reshape and expand H part: (group.dimension, |H|, |H|, 1, 1) -> (group.dimension, |H|, |H|, K, K)
        transformed_grid_H = transformed_grid_H.view(
            self.group.dimension, self.num_group_elements, self.num_group_elements, 1, 1
        ).expand(-1, -1, -1, self.kernel_size, self.kernel_size)

        # Concatenate
        transformed_grid = torch.cat([transformed_grid_R2, transformed_grid_H], dim=0)

        return transformed_grid
    
    def sample(self):
        """
        Sample convolution kernels using group elements h ∈ H.
        """
        raise NotImplementedError()


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
        ) # (C_out, C_in, |H|, K, K)

        # Initialize weights using Kaiming uniform initialization (like Conv2d)
        nn.init.kaiming_uniform_(tensor=self.weight.data, a=math.sqrt(5))
    
    def sample(self):
        """
        Sample convolution kernels for the group kernel.
        """
        # Flatten channel dims for joint interpolation
        # enables us to transform the entire kernel bank simultaneously using grid_sample
        weight = self.weight.view(
            self.in_channels * self.out_channels,
            self.num_group_elements,
            self.kernel_size,
            self.kernel_size
        )  # (C_in, C_out, |H|, K, K) -> (C_in * C_out, |H|, K, K)

        # Interpolation paths
        if self.group.dimension == 1: # Trilinear interpolation over spatial grid and group axis

            # Kernel signal: (C_in * C_out, |H|, K, K) -> (|H|, C_out*C_in, |H|, K, K)
            signal = weight.unsqueeze(0).expand(
                self.num_group_elements, -1, -1, -1, -1
            )

            # Affine grid to be transformed
            grid = self.transformed_affine_grid

            # Interpolated grid
            sampled = trilinear_interpolation(signal, grid)

            # Reshape back to (|H|, C_out, C_in, |H|, K, K)
            sampled = sampled.view(
                self.num_group_elements,
                self.out_channels,
                self.in_channels,
                self.num_group_elements,
                self.kernel_size,
                self.kernel_size
            )

            # Transpose back to support Conv2d: (C_out, |H|, C_in, |H|, K, K)
            return sampled.transpose(0, 1)

        else: # Bilinear interpolation over spatial grid
            # Reshape for batch processing: (|H|, C_out*C_in, K, K)
            signal = weight.permute(1, 0, 2, 3)

            # Repeat signal for each transformation (H transformations per group element)
            # shape: (|H|*|H|, C_out*C_in, K, K)
            signal = signal.repeat(
                self.num_group_elements, 1, 1, 1
            )

            # Prepare grid: (2, |H|, |H|, K, K) -> (|H|*|H|, K, K, 2)
            grid = self.transformed_affine_grid[:2] \
                .permute(1, 2, 3, 4, 0) \
                .reshape(self.num_group_elements * self.num_group_elements, self.kernel_size, self.kernel_size, 2)
            
            # Bilinear grid sampling
            sampled = nn.functional.grid_sample(
                input=signal,
                grid=grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )  # -> (|H|*|H|, C_out*C_in, K, K)

            # Reshape back and permute to desired output shape:
            # (|H|, |H|, C_out, C_in, K, K) -> (C_out, |H|, C_in, |H|, K, K)
            sampled = sampled.view(
                self.num_group_elements,
                self.num_group_elements,
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            ).permute(2, 0, 3, 1, 4, 5)

            return sampled
