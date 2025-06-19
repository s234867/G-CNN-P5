# PyTorch
import torch
import torch.nn.functional as F



def bilinear_interpolation(signal, grid):
    """
    Batched bilinear interpolation using torch.grid_sample.

    Args:
        signal: Tensor of shape [C, H, W] or [N, C, H, W]
        grid: Tensor of shape [2, N, H, W] (coords XY)

    Returns:
        Interpolated tensor: [N, C, H, W]
    """
    if signal.dim() == 3:
        signal = signal.unsqueeze(0)  # [1, C, H, W]

    if grid.dim() == 3:
        grid = grid.unsqueeze(1)  # [2, 1, H, W]

    # grid_sample expects grid shape [N, H, W, 2]
    grid = grid.permute(1, 2, 3, 0)  # [N, H, W, 2]

    # grid_sample expects YX coords, but you have XY, so swap last dim
    grid = torch.roll(grid, shifts=1, dims=-1)

    # Now expand signal batch dimension to match grid batch size N
    if signal.size(0) == 1 and grid.size(0) > 1:
        signal = signal.expand(grid.size(0), -1, -1, -1)

    return F.grid_sample(
        input=signal,
        grid=grid,
        mode='bilinear',
        padding_mode='zeros',
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