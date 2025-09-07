import torch
import torch.nn as nn
from einops import rearrange


class SamplingLayer(nn.Module):
    def __init__(
        self,
        *,
        sampling_factor: int = 2,
        nodes: list = None,
        subsample_nodes: list = None,
        type: str = "sample"
    ):
        """
        Perform subsampling operation from group to subgroup.
        Args:
            sampling_factor (int): Subsampling factor.
            nodes (list): List of nodes in the group. Nodes are represented as indices in the adjacency matrix.
            subsample_nodes (list): List of nodes in the subgroup.
            type (str): Type of sampling operation. Can be 'sample' or 'pool'.
        """

        super().__init__()
        self.sampling_factor = sampling_factor
        self.nodes = nodes
        self.subsample_nodes = subsample_nodes
        self.type = type
        self.sampled_nodes = None

        if self.type == "sample":
            # ========================================================================
            # SAMPLING MATRIX CONSTRUCTION
            # ========================================================================
            # This section constructs the sampling matrix S that maps from the main group G
            # to the subgroup H. The sampling matrix is the core of group downsampling.
            #
            # Mathematical Foundation:
            # - Main group G has |G| = len(self.nodes) elements
            # - Subgroup H has |H| = len(self.subsample_nodes) elements
            # - Sampling matrix S: R^|H| → R^|G| maps subgroup features to main group features
            # - Up-sampling matrix U: R^|G| → R^|H| maps main group features to subgroup features
            #
            # Matrix Construction:
            # - S[i, j] = 1 if subsample_nodes[i] = nodes[j], 0 otherwise
            # - This creates a |H| × |G| matrix where each row selects one element from G
            # - U is the pseudo-inverse of S, providing the best least-squares reconstruction
            #
            # Channel Reduction Formula:
            # Input:  (batch, C * |G|, spatial_dims)  - C features × |G| group elements
            # Output: (batch, C * |H|, spatial_dims)  - C features × |H| subgroup elements
            # Reduction factor: |G| / |H| = subsampling_factor
            
            # Initialize sampling matrix as |H| × |G| zero matrix
            sampling_matrix = torch.zeros(len(self.subsample_nodes), len(self.nodes))
            
            # Fill sampling matrix: S[i, j] = 1 if subsample_nodes[i] = nodes[j]
            # This creates a selection matrix that picks out the subgroup elements
            for i, node in enumerate(self.subsample_nodes):
                sampling_matrix[i, node] = 1

            # ========================================================================
            # UP-SAMPLING MATRIX CONSTRUCTION
            # ========================================================================
            # The up-sampling matrix U is the pseudo-inverse of the sampling matrix S
            # Mathematical: U = S^+ (Moore-Penrose pseudo-inverse)
            # 
            # Properties:
            # - U provides the best least-squares reconstruction from subgroup to main group
            # - U * S = I_|H| (identity on subgroup space)
            # - S * U is the orthogonal projection onto the subgroup space
            #
            # Usage in upsampling:
            # - To upsample from H to G: x_G = U * x_H
            # - This gives the best reconstruction of the full group representation
            
            up_sampling_matrix = torch.linalg.pinv(sampling_matrix)

            self.register_buffer(
                "sampling_matrix", torch.tensor(sampling_matrix).clone()
            )
            self.register_buffer(
                "up_sampling_matrix", torch.tensor(up_sampling_matrix).clone()
            )
        elif self.type == "pool":
            self.pool = nn.MaxPool1d(
                kernel_size=self.sampling_factor, stride=self.sampling_factor
            )
        else:
            raise ValueError("Unknown type")

    def forward(self, x):
        """
        Forward pass for group downsampling.
        
        Mathematical Operation:
        - Input: x ∈ R^(batch × C×|G| × spatial_dims) where C is number of features
        - Output: y ∈ R^(batch × C×|H| × spatial_dims) where |H| = |G| / subsampling_factor
        - Operation: y = S * x where S is the |H| × |G| sampling matrix
        
        Channel Reduction:
        - Input channels:  C * |G|  (C features × |G| group elements)
        - Output channels: C * |H|  (C features × |H| subgroup elements)  
        - Reduction factor: |G| / |H| = subsampling_factor
        """
        if self.type == "sample":
            if len(x.shape) == 1:
                # ====================================================================
                # 1D CASE: Simple vector downsampling
                # ====================================================================
                # Input: x ∈ R^|G| (single group representation)
                # Output: y ∈ R^|H| (subgroup representation)
                # Operation: y = S * x
                return self.sampling_matrix @ x
            elif len(x.shape) == 4:
                # ====================================================================
                # 4D CASE: 2D spatial data with group dimension
                # ====================================================================
                # Input: x ∈ R^(batch × C×|G| × height × width)
                # Process: Reshape to separate group dimension, apply sampling, reshape back
                # 
                # Step 1: Reshape from (batch, C×|G|, H, W) to (batch, C, |G|, H, W)
                x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.nodes))
                
                # Step 2: Apply sampling matrix using Einstein summation
                # einsum("fg,bcghw->bcfhw", S, x) where:
                # - f: subgroup dimension (|H|)
                # - g: main group dimension (|G|)  
                # - b: batch, c: channels, h: height, w: width
                # Result: (batch, C, |H|, H, W)
                x = torch.einsum("fg,bcghw->bcfhw", self.sampling_matrix, x)
                
                # Step 3: Reshape back to (batch, C×|H|, H, W)
                x = rearrange(x, "b c g h w -> b (c g) h w")
                return x
            elif len(x.shape) == 5:
                # ====================================================================
                # 5D CASE: 3D spatial data with group dimension
                # ====================================================================
                # Input: x ∈ R^(batch × C×|G| × depth × height × width)
                # Process: Reshape to separate group dimension, apply sampling, reshape back
                #
                # Step 1: Reshape from (batch, C×|G|, D, H, W) to (batch, C, |G|, D, H, W)
                x = rearrange(x, "b (c g) h w d -> b c g h w d", g=len(self.nodes))
                
                # Step 2: Apply sampling matrix using Einstein summation
                # einsum("fg,bcghwd->bcfhwd", S, x) where:
                # - f: subgroup dimension (|H|)
                # - g: main group dimension (|G|)
                # - b: batch, c: channels, h: height, w: width, d: depth
                # Result: (batch, C, |H|, D, H, W)
                x = torch.einsum("fg,bcghwd->bcfhwd", self.sampling_matrix, x)
                
                # Step 3: Reshape back to (batch, C×|H|, D, H, W)
                x = rearrange(x, "b c g h w d -> b (c g) h w d")
                return x
        elif self.type == "pool":
            x = x.unsqueeze(0)
            x = self.pool(x)
            return x.squeeze(0)
        else:
            raise ValueError("Unknown type")

    def up_sample(self, x):
        if len(x.shape) == 4:
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.subsample_nodes))
            x_upsampled = torch.einsum("fg,bcghw->bcfhw", self.up_sampling_matrix, x)
            x_upsampled = rearrange(x_upsampled, "b c g h w -> b (c g) h w")
        elif len(x.shape) == 5:
            x = rearrange(x, "b (c g) h w d -> b c g h w d", g=len(self.subsample_nodes))
            x_upsampled = torch.einsum("fg,bcghwd->bcfhwd", self.up_sampling_matrix, x)
            x_upsampled = rearrange(x_upsampled, "b c g h w d -> b (c g) h w d")
        else:
            x_upsampled = self.up_sampling_matrix @ x
        return x_upsampled
