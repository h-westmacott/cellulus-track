import torch
import torch.nn.functional as F
from typing import Union, Tuple, List, Optional

def upsample_4d_nearest(
    x: torch.Tensor, 
    size: Optional[Union[Tuple[int, ...], List[int]]] = None, 
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None
) -> torch.Tensor:
    """
    Upsamples a 6D tensor using 4D nearest-neighbor interpolation.

    Args:
        x (torch.Tensor): The input tensor of shape (N, C, D1, D2, D3, D4).
                          We use (N, C, T, D, H, W) for clarity.
        size (tuple or list, optional): The target output size for the 4 spatial
                                        dimensions (T_out, D_out, H_out, W_out).
        scale_factor (float or tuple, optional): The multiplier for the spatial size.
                                                  Must be a float or a 4-element tuple.

    Returns:
        torch.Tensor: The upsampled tensor.
    """
    # --- Input Validation ---
    if x.dim() != 6:
        raise ValueError(f"Expected 6D input tensor, but got {x.dim()}D")
    if size is None and scale_factor is None:
        raise ValueError("Either 'size' or 'scale_factor' must be specified")
    if size is not None and scale_factor is not None:
        raise ValueError("Only one of 'size' or 'scale_factor' should be specified")

    N, C, T, D, H, W = x.shape

    # --- Determine Target Size ---
    if size is not None:
        T_out, D_out, H_out, W_out = size
    else: # scale_factor is not None
        if isinstance(scale_factor, (float, int)):
            sf_t, sf_d, sf_h, sf_w = (scale_factor,) * 4
        else:
            assert len(scale_factor) == 4, "scale_factor must be a float or 4-element tuple"
            sf_t, sf_d, sf_h, sf_w = scale_factor
        
        T_out = int(T * sf_t)
        D_out = int(D * sf_d)
        H_out = int(H * sf_h)
        W_out = int(W * sf_w)

    # --- Step 1: Upsample the first spatial dimension (T) ---
    # To use interpolate, we treat T as the spatial dimension of a 3D tensor.
    # We permute and reshape to isolate T.
    # (N, C, T, D, H, W) -> (N, C, D, H, W, T)
    x_permuted = x.permute(0, 1, 3, 4, 5, 2)
    # (N, C, D, H, W, T) -> (N*C*D*H*W, 1, T)
    x_reshaped = x_permuted.reshape(-1, 1, T)
    
    # Apply 1D nearest interpolation
    x_interp_t = F.interpolate(x_reshaped, size=T_out, mode='nearest')
    
    # Reshape and permute back
    # (N*C*D*H*W, 1, T_out) -> (N, C, D, H, W, T_out)
    y = x_interp_t.view(N, C, D, H, W, T_out)
    # (N, C, D, H, W, T_out) -> (N, C, T_out, D, H, W)
    y = y.permute(0, 1, 5, 2, 3, 4)
    
    # --- Step 2: Upsample the other three spatial dimensions (D, H, W) ---
    # Merge N and T_out to use the standard 3D interpolate function.
    # (N, C, T_out, D, H, W) -> (N * T_out, C, D, H, W)
    y_reshaped = y.reshape(N * T_out, C, D, H, W)
    
    # Apply 3D nearest interpolation
    interp_3d = F.interpolate(y_reshaped, size=(D_out, H_out, W_out), mode='nearest')
    
    # Reshape back to the final 6D shape
    # (N * T_out, C, D_out, H_out, W_out) -> (N, C, T_out, D_out, H_out, W_out)
    final_output = interp_3d.view(N, C, T_out, D_out, H_out, W_out)
    
    return final_output


class Upsample4DNearest(torch.nn.Module):
    def __init__(self, size: Optional[Union[Tuple[int, ...], List[int]]] = None, 
                 scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
                 mode: str = 'nearest'):
        super(Upsample4DNearest, self).__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample_4d_nearest(x, self.size, self.scale_factor)