"""
Hybrid Convolution and Group Resampling Layer for 3D Group Equivariant CNNs

This module implements a hybrid layer that combines group convolution with group resampling,
providing a unified interface for both same-group and cross-group operations in 3D Group
Equivariant CNNs.

MATHEMATICAL FOUNDATIONS:
- Group Convolution: f * ψ where f ∈ L^2(G), ψ ∈ L^2(G)
- Group Resampling: S: L^2(G) → L^2(H) where H ⊆ G
- Channel Calculation: total_channels = base_channels × group_order
- Equivariance: Maintains f(g·x) = g·f(x) property

ARCHITECTURE:
- rnConv: Group convolution that maintains group order
- SubgroupDownsample: Group resampling that changes group order
- Anti-aliasing: Prevents aliasing artifacts during group downsampling

GROUP PROCESSING:
- Same Group Order: Only applies rnConv (no resampling)
- Decreasing Group Order: rnConv + subgroup downsampling
- Increasing Group Order: rnConv + subgroup upsampling

USAGE:
    layer = HybridConvGroupResample3D(
        in_group_type="octahedral",
        in_group_order=24,
        in_num_features=32,
        out_group_type="cycle",
        out_group_order=4,
        out_num_features=64,
        apply_antialiasing=True
    )
"""

import torch
import torch.nn as nn
from gsampling.layers.rnconv import rnConv  # Group convolution layer
from gsampling.layers.downsampling import SubgroupDownsample  # Group downsampling layer


class HybridConvGroupResample3D(nn.Module):
    """
    Hybrid layer combining group convolution with group resampling for 3D Group Equivariant CNNs.
    
    This class implements a composite layer that combines:
    - Group convolution (rnConv) that maintains group order
    - Group resampling (SubgroupDownsample) that changes group order
    - Anti-aliasing for preventing artifacts during group downsampling
    
    The layer provides a unified interface for both same-group and cross-group operations,
    making it easier to build complex Group Equivariant CNN architectures.
    
    MATHEMATICAL OPERATIONS:
        1. Group Convolution: f * ψ where f ∈ L^2(G), ψ ∈ L^2(G)
        2. Group Resampling: S: L^2(G) → L^2(H) where H ⊆ G
        3. Channel Calculation: total_channels = base_channels × group_order
        4. Equivariance: Maintains f(g·x) = g·f(x) property
        
    GROUP PROCESSING MODES:
        - Same Group Order (out_group_order == in_group_order): Only applies rnConv
        - Decreasing Group Order (out_group_order < in_group_order): rnConv + subgroup downsampling
        - Increasing Group Order (out_group_order > in_group_order): rnConv + subgroup upsampling
        
    ANTI-ALIASING:
        When group downsampling is applied, anti-aliasing can be enabled to prevent
        aliasing artifacts. This is particularly important for maintaining signal
        quality during group order reduction.
    """

    def __init__(
        self,
        *,
        in_group_type: str,
        in_group_order: int,
        in_num_features: int,
        out_group_type: str,
        out_group_order: int,
        out_num_features: int,
        representation: str = "regular",
        kernel_size: int = 3,
        domain: int = 3,
        apply_antialiasing: bool = False,
        anti_aliasing_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize hybrid convolution and group resampling layer.
        
        This constructor builds a composite layer that combines group convolution
        with group resampling, providing a unified interface for both same-group
        and cross-group operations in 3D Group Equivariant CNNs.
        
        ARCHITECTURE OVERVIEW:
            - rnConv: Group convolution that maintains group order
            - SubgroupDownsample: Group resampling that changes group order
            - Anti-aliasing: Prevents aliasing artifacts during group downsampling
            
        GROUP PROCESSING FLOW:
            1. Group Convolution: (batch, in_channels×|G_in|, depth, height, width)
            2. Group Resampling: (batch, out_channels×|G_out|, depth, height, width)
            3. Output: (batch, out_channels×|G_out|, depth, height, width)
            
        MATHEMATICAL FOUNDATIONS:
            - Group Convolution: (f * ψ)(g) = Σ_{h∈G} f(h)ψ(h^-1g)
            - Group Resampling: S: L^2(G) → L^2(H) where H ⊆ G
            - Channel Calculation: total_channels = base_channels × group_order
            - Equivariance: f(g·x) = g·f(x) for all group elements g
            
        Args:
            in_group_type (str): Input group type
                - "octahedral": Octahedral group (24 elements)
                - "cycle": Cyclic group (C4, 4 elements)
                - "dihedral": Dihedral group (8 elements)
                
            in_group_order (int): Input group order (|G_in|)
                - For octahedral: 24
                - For C4: 4
                - For dihedral: 8
                
            in_num_features (int): Number of input features per group element
                - Total input channels = in_num_features × in_group_order
                - Example: 32 features × 24 group order = 768 total channels
                
            out_group_type (str): Output group type
                - Can be same as input or different
                - Determines the group structure of the output
                
            out_group_order (int): Output group order (|G_out|)
                - Can be same as input, smaller (downsampling), or larger (upsampling)
                - Total output channels = out_num_features × out_group_order
                
            out_num_features (int): Number of output features per group element
                - Total output channels = out_num_features × out_group_order
                - Example: 64 features × 4 group order = 256 total channels
                
            representation (str): Group representation type
                - "regular": Regular representation (default)
                - "trivial": Trivial representation
                - Determines how group elements are represented
                
            kernel_size (int): 3D convolution kernel size
                - Common values: 3 (3×3×3 kernels), 5 (5×5×5 kernels)
                - Must be odd numbers for proper padding
                
            domain (int): Spatial domain dimension
                - Must be 3 for 3D data
                - Determines convolution dimensionality
                
            apply_antialiasing (bool): Whether to apply anti-aliasing
                - True: Apply spectral anti-aliasing during group downsampling
                - False: Skip anti-aliasing (may cause artifacts)
                
            anti_aliasing_kwargs (dict, optional): Anti-aliasing parameters
                - smoothness_loss_weight (float): Weight for smoothness regularization
                - iterations (int): Number of optimization iterations
                - mode (str): Optimization mode ("gpu_optim", "linear_optim")
                
            device (str | torch.device): Device for computation
                - "cpu": CPU computation
                - "cuda": GPU computation (if available)
                
            dtype (torch.dtype): Data type for computations
                - torch.float32: Single precision (default)
                - torch.float16: Half precision (memory efficient)
                - torch.float64: Double precision (higher accuracy)
                
        Raises:
            ValueError: If group types are invalid or incompatible
            AssertionError: If group order calculations result in invalid values
            
        Example:
            >>> layer = HybridConvGroupResample3D(
            ...     in_group_type="octahedral",
            ...     in_group_order=24,
            ...     in_num_features=32,
            ...     out_group_type="cycle",
            ...     out_group_order=4,
            ...     out_num_features=64,
            ...     apply_antialiasing=True
            ... )
            >>> x = torch.randn(2, 32*24, 8, 8, 8)  # Input: (batch, channels×group_order, depth, height, width)
            >>> output = layer(x)  # Output: (batch, 64*4, 8, 8, 8)
        """
        super().__init__()

        self.in_group_type = in_group_type
        self.in_group_order = in_group_order
        self.out_group_type = out_group_type
        self.out_group_order = out_group_order
        self.apply_antialiasing = apply_antialiasing
        self.anti_aliasing_kwargs = anti_aliasing_kwargs or {}

        # We'll build two convs depending on direction to keep orders equal across each conv
        self.conv_increase = None
        self.conv_decrease = None

        # Prepare resampler if orders differ
        self.resampler: SubgroupDownsample | None = None
        if out_group_order != in_group_order:
            if out_group_order < in_group_order:
                # Downsample in group space
                subsampling_factor = in_group_order // out_group_order
                self.resampler = SubgroupDownsample(
                    group_type=in_group_type,
                    order=in_group_order,
                    sub_group_type=out_group_type,
                    subsampling_factor=subsampling_factor,
                    num_features=out_num_features,
                    generator="r-s",
                    device=device,
                    dtype=dtype,
                    sample_type="sample",
                    apply_antialiasing=apply_antialiasing,
                    anti_aliasing_kwargs=self.anti_aliasing_kwargs,
                )
                # Conv at higher (input) order
                self.conv_decrease = rnConv(
                    in_group_type=in_group_type,
                    in_order=in_group_order,
                    in_num_features=in_num_features,
                    in_representation=representation,
                    out_group_type=in_group_type,
                    out_num_features=out_num_features,
                    out_representation=representation,
                    domain=domain,
                    kernel_size=kernel_size,
                ).to(device=device, dtype=dtype)
            else:
                # Upsample in group space
                upsampling_factor = out_group_order // in_group_order
                self.resampler = SubgroupDownsample(
                    group_type=out_group_type,
                    order=out_group_order,
                    sub_group_type=in_group_type,
                    subsampling_factor=upsampling_factor,
                    num_features=out_num_features,
                    generator="r-s",
                    device=device,
                    dtype=dtype,
                    sample_type="sample",
                    apply_antialiasing=apply_antialiasing,
                    anti_aliasing_kwargs=self.anti_aliasing_kwargs,
                )
                # Conv at higher (output) order
                self.conv_increase = rnConv(
                    in_group_type=out_group_type,
                    in_order=out_group_order,
                    in_num_features=in_num_features,
                    in_representation=representation,
                    out_group_type=out_group_type,
                    out_num_features=out_num_features,
                    out_representation=representation,
                    domain=domain,
                    kernel_size=kernel_size,
                ).to(device=device, dtype=dtype)
        else:
            # No resampling, single conv at this order
            self.conv_decrease = rnConv(
                in_group_type=in_group_type,
                in_order=in_group_order,
                in_num_features=in_num_features,
                in_representation=representation,
                out_group_type=in_group_type,
                out_num_features=out_num_features,
                out_representation=representation,
                domain=domain,
                kernel_size=kernel_size,
            ).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid convolution and group resampling layer.
        
        This method performs the complete forward pass, applying group convolution
        and group resampling in the appropriate order based on the group order
        relationship between input and output.
        
        MATHEMATICAL FLOW:
            1. Same Group Order (out_group_order == in_group_order):
               - Apply only group convolution: x → conv(x)
               - No group resampling needed
               
            2. Decreasing Group Order (out_group_order < in_group_order):
               - Apply group convolution at high order: x → conv(x)
               - Apply group downsampling: conv(x) → downsample(conv(x))
               
            3. Increasing Group Order (out_group_order > in_group_order):
               - Apply group upsampling: x → upsample(x)
               - Apply group convolution at high order: upsample(x) → conv(upsample(x))
               
        GROUP EQUIVARIANCE:
            The entire forward pass maintains group equivariance f(g·x) = g·f(x).
            Both group convolution and group resampling preserve this property.
            
        CHANNEL FLOW:
            - Input: (batch, in_channels×|G_in|, depth, height, width)
            - After convolution: (batch, out_channels×|G_in|, depth, height, width)
            - After resampling: (batch, out_channels×|G_out|, depth, height, width)
            
        Args:
            x (torch.Tensor): Input tensor with shape (batch, in_channels×|G_in|, depth, height, width)
                - batch: Batch size
                - in_channels×|G_in|: Input channels × input group order
                - depth, height, width: 3D spatial dimensions
                
        Returns:
            torch.Tensor: Output tensor with shape (batch, out_channels×|G_out|, depth, height, width)
                - batch: Batch size
                - out_channels×|G_out|: Output channels × output group order
                - depth, height, width: 3D spatial dimensions (same as input)
                
        Example:
            >>> layer = HybridConvGroupResample3D(...)
            >>> x = torch.randn(2, 32*24, 8, 8, 8)  # Input: (batch, 32×24, 8, 8, 8)
            >>> output = layer(x)  # Output: (batch, 64×4, 8, 8, 8)
        """
        # Debug information for troubleshooting (commented out for production)
        # print(f"Hybrid forward: in_order={self.in_group_order}, out_order={self.out_group_order}, x_ch={x.shape[1]}")
        
        if self.resampler is None:
            # CASE 1: Same Group Order - Only apply group convolution
            # This is the simplest case where no group resampling is needed
            # print("Path: no-resample -> conv_decrease")
            return self.conv_decrease(x)
        
        # CASE 2: Decreasing Group Order - Convolution first, then downsampling
        # This applies group convolution at the high (input) order, then downsamples
        if self.out_group_order < self.in_group_order:
            # print("Path: decrease -> conv_decrease -> downsample")
            # Step 1: Apply group convolution at high order
            x = self.conv_decrease(x)  # (batch, out_channels×|G_in|, depth, height, width)
            
            # Step 2: Apply group downsampling
            # Note: SubgroupDownsample returns a tuple (x, v), so we unpack it
            x, _ = self.resampler(x)  # (batch, out_channels×|G_out|, depth, height, width)
            return x
        
        # CASE 3: Increasing Group Order - Upsampling first, then convolution
        # This applies group upsampling first, then convolution at the high (output) order
        # print("Path: increase -> upsample -> conv_increase")
        # Step 1: Apply group upsampling
        x = self.resampler.upsample(x)  # (batch, out_channels×|G_out|, depth, height, width)
        
        # Step 2: Apply group convolution at high order
        return self.conv_increase(x)  # (batch, out_channels×|G_out|, depth, height, width)


