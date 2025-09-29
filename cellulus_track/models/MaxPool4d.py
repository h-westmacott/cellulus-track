import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPooling4d(nn.Module):
    """
    A 4D Max Pooling layer implementation for 6D tensors (N, C, D1, D2, D3, D4).
    """
    def __init__(self, kernel_size, stride=None, padding=0, 
                 dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPooling4d, self).__init__()
        
        # Ensure kernel_size and stride are 4-element tuples or lists
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * 4
        else:
            assert len(kernel_size) == 4, "kernel_size must be a 4-element tuple"
            self.kernel_size = kernel_size
            
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride,) * 4
        else:
            assert len(stride) == 4, "stride must be a 4-element tuple"
            self.stride = stride

        # For simplicity, this implementation only supports padding=0 and dilation=1
        # A more complex implementation would be needed to support them.
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 4D max pooling.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D1, D2, D3, D4).
                              We'll use (N, C, T, D, H, W) for clarity.

        Returns:
            torch.Tensor: Pooled output tensor.
        """
        # Ensure input is 6D
        assert x.dim() == 6, f"Expected 6D input tensor, but got {x.dim()}D"
        
        N, C, T, D, H, W = x.shape
        
        # 1. Unfold the first spatial dimension (T) to create sliding windows.
        # The unfold operation creates an additional dimension at the end.
        # Input: (N, C, T, D, H, W)
        # Output: (N, C, T_out, D, H, W, kernel_size[0])
        x_unfolded = x.unfold(2, self.kernel_size[0], self.stride[0])
        
        # 2. Take the max over the window dimension (the one we just created).
        # This effectively performs pooling over the first spatial dimension.
        # Input: (N, C, T_out, D, H, W, kernel_size[0])
        # Output: (N, C, T_out, D, H, W)
        x_pooled_t, _ = torch.max(x_unfolded, dim=-1)
        
        # The number of output frames in the T dimension is now T_out
        T_out = x_pooled_t.shape[2]
        
        # 3. Reshape and apply 3D Max Pooling to the remaining dimensions (D, H, W).
        # We merge the batch (N) and T_out dimensions to use the efficient max_pool3d.
        # Input: (N, C, T_out, D, H, W) -> (N * T_out, C, D, H, W)
        x_reshaped = x_pooled_t.reshape(N * T_out, C, D, H, W)
        
        # Apply standard 3D max pooling
        kernel_3d = self.kernel_size[1:]
        stride_3d = self.stride[1:]
        
        pooled_output = F.max_pool3d(x_reshaped, kernel_size=kernel_3d, stride=stride_3d)
        
        # 4. Reshape the tensor back to its final 6D shape.
        # The D, H, W dimensions have been pooled.
        D_out, H_out, W_out = pooled_output.shape[2:]
        # Input: (N * T_out, C, D_out, H_out, W_out) -> (N, T_out, C, D_out, H_out, W_out)
        final_output = pooled_output.view(N, T_out, C, D_out, H_out, W_out)
        
        # 5. Permute to restore the original channel dimension order (N, C, ...).
        # Input: (N, T_out, C, D_out, H_out, W_out) -> (N, C, T_out, D_out, H_out, W_out)
        final_output = final_output.permute(0, 2, 1, 3, 4, 5)
        
        return final_output