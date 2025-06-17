## subfiles/utils.py ##

# PyTorch
import torch
import torch.nn.functional as F


# To interpolate pixel values when rotating in the interpolative lifting kernel
def bilinear_interpolation(signal, grid):
    """
    Perform bilinear interpolation using grid_sample.

    @param signal: Tensor [C, H, W] or [B, C, H, W]
    @param grid: Tensor [2, H, W] or [B, 2, H, W]
    @returns: Interpolated tensor [B, C, H, W]
    """
    if signal.ndim == 3:
        signal = signal.unsqueeze(0)  # [1, C, H, W]
    if grid.ndim == 3:
        grid = grid.unsqueeze(0)  # [1, 2, H, W]

    # Convert to [B, H, W, 2] format (YX order)
    grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
    grid = torch.roll(grid, shifts=1, dims=-1)  # Swap X ↔ Y

    return F.grid_sample(
        input=signal,
        grid=grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )



# Interpolate pixel values in the group convolution kernel
def trilinear_interpolation(signal, grid):
    """ 
    @param signal: Tensor containing pixel values (N, C, D_in, H_in, W_in)
    @param grid: Tensor containing coordinate values (N, D_out, H_out, W_out, 3)
    """
    grid = grid.permute(1, 2, 3, 4, 0)

    grid = torch.roll(grid, shifts=1, dims=-1)
    
    return F.grid_sample(
        signal,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )