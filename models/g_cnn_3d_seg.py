"""
4D U-Net for 3D Medical Image Segmentation with Group Equivariance

This module implements a 4D U-Net architecture that combines:
- 3D spatial processing (depth, height, width)
- Group axis processing (group equivariance)
- Encoder-decoder architecture with skip connections
- Group downsampling and upsampling

MATHEMATICAL FOUNDATIONS:
- Group Theory: Implements equivariance under octahedral group O (24 elements)
- Cyclic Subgroup: Uses C4 cyclic subgroup (4 elements) for 90° rotations around z-axis
- 4D Processing: Combines spatial (3D) and group dimensions
- Skip Connections: Concatenate encoder and decoder features for better reconstruction

ARCHITECTURE:
- Encoder: 3D GCNN for feature extraction with group downsampling
- Bottleneck: Deepest features with group processing
- Decoder: Feature reconstruction with group upsampling + skip connections
- Final Conv: Output segmentation mask

GROUP PROCESSING:
- Input: (batch, 1, depth, height, width) - trivial representation
- Encoder: (batch, channels×|G|, depth/8, height/8, width/8) - regular representation
- Decoder: (batch, channels×|G|, depth, height, width) - upsampled features
- Output: (batch, num_classes, depth, height, width) - segmentation mask

EQUIVARIANCE PROPERTY:
The network maintains group equivariance: f(g·x) = g·f(x) for all group elements g.
This means that rotating the input by 90° around the z-axis will produce a
correspondingly rotated segmentation mask.

USAGE:
    model = Gcnn3DSegmentation(
        num_layers=4,
        num_channels=[1, 32, 64, 128, 256],
        dwn_group_types=[["octahedral", "octahedral"], ...],
        subsampling_factors=[1, 1, 1, 1],
        init_group_order=24,
        num_classes=4
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsampling.layers.rnconv import rnConv  # Group convolution layers
from gsampling.layers.downsampling import SubgroupDownsample  # Group downsampling
from gsampling.utils.group_utils import get_group  # Group creation utilities
from einops import rearrange  # Tensor reshaping operations
from .g_cnn_3d import Gcnn3D  # 3D GCNN encoder
from .hybrid import HybridConvGroupResample3D  # Hybrid convolution + group resampling


class Gcnn3DSegmentation(nn.Module):
    """
    4D U-Net for 3D Medical Image Segmentation with Group Equivariance.
    
    This class implements a complete 4D U-Net architecture that processes both
    spatial (3D) and group dimensions for 3D medical image segmentation. The
    architecture maintains group equivariance throughout the forward pass.
    """
    
    def __init__(
        self,
        *,
        num_layers: int,
        num_channels: list[int],
        kernel_sizes: list[int],
        num_classes: int,
        dwn_group_types: list,
        init_group_order: int,
        spatial_subsampling_factors: list[int],
        subsampling_factors,
        domain,
        apply_antialiasing,
        antialiasing_kwargs,
        dropout_rate,
        dtype=torch.float32,
        device="cpu",
        layer_kwargs={}
    ):
        """
        Initialize 4D U-Net for 3D Medical Image Segmentation with Group Equivariance.
        
        This constructor builds a complete 4D U-Net architecture that combines:
        - 3D spatial processing (depth, height, width)
        - Group axis processing (group equivariance)
        - Encoder-decoder architecture with skip connections
        - Group downsampling and upsampling
        
        ARCHITECTURE OVERVIEW:
        - Encoder: 3D GCNN for feature extraction with group downsampling
        - Bottleneck: Deepest features with group processing
        - Decoder: Feature reconstruction with group upsampling + skip connections
        - Final Conv: Output segmentation mask
        
        GROUP PROCESSING FLOW:
        1. Input: (batch, 1, depth, height, width) - trivial representation
        2. Encoder: (batch, channels×|G|, depth/8, height/8, width/8) - regular representation
        3. Decoder: (batch, channels×|G|, depth, height, width) - upsampled features
        4. Output: (batch, num_classes, depth, height, width) - segmentation mask
        
        MATHEMATICAL FOUNDATIONS:
        - Group Equivariance: f(g·x) = g·f(x) for all group elements g
        - 4D Processing: Combines spatial (3D) and group dimensions
        - Skip Connections: Concatenate encoder and decoder features
        - Group Pooling: Collapse group dimension for final output
        
        Args:
            num_layers (int): Number of encoder/decoder layers
                - Must be ≥ 1
                - Each layer processes both spatial and group dimensions
                
            num_channels (list[int]): Channel progression for each layer
                - Length must be num_layers + 1
                - Format: [input_channels, layer1_channels, ..., output_channels]
                - Example: [1, 32, 64, 128, 256] for 4 layers
                
            kernel_sizes (list[int]): 3D convolution kernel sizes for each layer
                - Length must be num_layers
                - Common values: 3 (3×3×3 kernels), 5 (5×5×5 kernels)
                - Must be odd numbers for proper padding
                
            num_classes (int): Number of segmentation classes
                - For ACDC: 4 classes (background + 3 cardiac structures)
                - For binary segmentation: 2 classes
                - Determines output tensor shape: (batch, num_classes, depth, height, width)
                
            dwn_group_types (list): Group type transitions for each layer
                - Length must be num_layers
                - Format: [["input_group", "output_group"], ...]
                - Examples: [["octahedral", "octahedral"], ["octahedral", "cycle"]]
                - Group types: "octahedral" (24 elements), "cycle" (C4, 4 elements)
                
            init_group_order (int): Initial group order (|G₀|)
                - For octahedral group: 24
                - For cyclic group: 4 (C4)
                - Determines initial channel expansion: input_channels × init_group_order
                
            spatial_subsampling_factors (list[int]): Spatial downsampling factors
                - Length must be num_layers
                - Values: 1 (no downsampling), 2 (2× downsampling), etc.
                - Applied via BlurPool3d for anti-aliased spatial downsampling
                
            subsampling_factors (list[int]): Group downsampling factors
                - Length must be num_layers
                - Values: 1 (no group downsampling), 6 (O→C4), etc.
                - Reduces group order: |G_out| = |G_in| / subsampling_factor
                
            domain (int): Spatial domain dimension
                - Must be 3 for 3D data
                - Determines convolution dimensionality
                
            apply_antialiasing (bool): Whether to apply anti-aliasing
                - True: Apply spectral anti-aliasing during group downsampling
                - False: Skip anti-aliasing (may cause artifacts)
                
            antialiasing_kwargs (dict): Anti-aliasing parameters
                - smoothness_loss_weight (float): Weight for smoothness regularization
                - iterations (int): Number of optimization iterations
                - mode (str): Optimization mode ("gpu_optim", "linear_optim")
                
            dropout_rate (float): Dropout probability for regularization
                - Range: [0.0, 1.0]
                - Applied after each convolutional layer
                - 0.0: No dropout, 1.0: Complete dropout
                
            dtype (torch.dtype): Data type for computations
                - torch.float32: Single precision (default)
                - torch.float16: Half precision (memory efficient)
                - torch.float64: Double precision (higher accuracy)
                
            device (str): Device for computation
                - "cpu": CPU computation
                - "cuda": GPU computation (if available)
                
            layer_kwargs (dict): Additional layer-specific parameters
                - Currently unused, reserved for future extensions

        Raises:
            ValueError: If parameter dimensions don't match or invalid values provided
            AssertionError: If group order calculations result in invalid values
            
        Example:
            >>> model = Gcnn3DSegmentation(
            ...     num_layers=4,
            ...     num_channels=[1, 32, 64, 128, 256],
            ...     kernel_sizes=[3, 3, 3, 3],
            ...     num_classes=4,
            ...     dwn_group_types=[["octahedral", "octahedral"], ...],
            ...     subsampling_factors=[1, 1, 1, 1],
            ...     init_group_order=24,
            ...     spatial_subsampling_factors=[2, 2, 2, 1],
            ...     domain=3,
            ...     apply_antialiasing=True,
            ...     antialiasing_kwargs={"smoothness_loss_weight": 0.1}
            ... )
            >>> x = torch.randn(2, 1, 32, 32, 32)  # (batch, channels, depth, height, width)
            >>> output = model(x)  # (batch, num_classes, depth, height, width)
        """
        super().__init__()
        
        # Store configuration
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        self.dwn_group_types = dwn_group_types
        self.init_group_order = init_group_order
        self.spatial_subsampling_factors = spatial_subsampling_factors
        self.subsampling_factors = subsampling_factors
        self.domain = domain
        self.apply_antialiasing = apply_antialiasing
        self.antialiasing_kwargs = antialiasing_kwargs
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        self.device = device
        
        # Create encoder using existing Gcnn3D architecture
        self.encoder = Gcnn3D(
            num_layers=num_layers,
            num_channels=num_channels,
            kernel_sizes=kernel_sizes,
            num_classes=num_classes,  # Will be overridden
            dwn_group_types=dwn_group_types,
            init_group_order=init_group_order,
            spatial_subsampling_factors=spatial_subsampling_factors,
            subsampling_factors=subsampling_factors,
            domain=domain,
            pooling_type="max",  # Required parameter (won't be used due to fully_convolutional=True)
            apply_antialiasing=apply_antialiasing,
            antialiasing_kwargs=antialiasing_kwargs,
            dropout_rate=dropout_rate,
            dtype=dtype,
            device=device,
            fully_convolutional=True,  # Don't add final pooling/linear layers
            layer_kwargs=layer_kwargs
        )
        
        # Calculate group orders at each layer (reuse encoder's logic)
        self.group_orders = [init_group_order]
        current_group_order = init_group_order
        
        for i in range(num_layers):
            if subsampling_factors[i] > 1:
                # Calculate next group order after subsampling
                next_group_type = dwn_group_types[i][1]
                if next_group_type in ["octahedral", "full_octahedral"]:
                    next_group_order = get_group(next_group_type, 1).order()
                else:
                    next_group_order = current_group_order // subsampling_factors[i]
                current_group_order = next_group_order
            self.group_orders.append(current_group_order)
        
        # Build decoder (reverse of encoder)
        self._build_decoder()
        
        # Build output layer
        self._build_output_layer()

    def _build_decoder(self):
        """Build decoder layers to reverse the encoder operations."""
        self.decoder_upsampling = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Process layers in reverse order
        for i in reversed(range(self.num_layers)):
            # We will resample group order inside HybridConvGroupResample3D after concatenation
            self.decoder_upsampling.append(None)
            
            # Spatial upsampling (reverse of encoder's spatial downsampling)
            if self.spatial_subsampling_factors[i] > 1:
                spatial_upsampling = SpatialUpsample3D(
                    scale_factor=self.spatial_subsampling_factors[i]
                )
            else:
                spatial_upsampling = nn.Identity()
            
            # Store spatial upsampling for later use
            if not hasattr(self, 'decoder_spatial_upsampling'):
                self.decoder_spatial_upsampling = nn.ModuleList()
            self.decoder_spatial_upsampling.append(spatial_upsampling)
            
            # Decoder hybrid block: conv at current group order then resample to next (higher) group order for previous level
            in_features = self.num_channels[i + 1] * 2
            out_features = self.num_channels[i] if i > 0 else self.num_channels[1]
            in_group_order = self.group_orders[i + 1]
            out_group_order = self.group_orders[i]
            in_group_type = self.dwn_group_types[i][1]  # subgroup at this depth
            out_group_type = self.dwn_group_types[i][0]  # parent group for next level up

            print(f"Build decoder block i={i}: in_type={in_group_type}({in_group_order}), out_type={out_group_type}({out_group_order}), in_features={in_features}, out_features={out_features}")

            block = HybridConvGroupResample3D(
                in_group_type=in_group_type,
                in_group_order=in_group_order,
                in_num_features=in_features,
                out_group_type=out_group_type,
                out_group_order=out_group_order,
                out_num_features=out_features,
                representation="regular",
                kernel_size=self.kernel_sizes[i],
                domain=self.domain,
                apply_antialiasing=self.apply_antialiasing,
                anti_aliasing_kwargs=self.antialiasing_kwargs,
                device=self.device,
                dtype=self.dtype,
            ).to(device=self.device, dtype=self.dtype)
            self.decoder_blocks.append(block)
        
        # Keep order: deepest -> shallowest to align with forward
        self.decoder_upsampling = nn.ModuleList(self.decoder_upsampling)
        self.decoder_spatial_upsampling = nn.ModuleList(self.decoder_spatial_upsampling)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    def _build_output_layer(self):
        """Build final segmentation output layer."""
        # Output layer: convert to segmentation classes and pool over group dimension
        final_group_type = self.dwn_group_types[0][0]
        final_group_order = self.group_orders[0]
        
        self.output_conv = rnConv(
            in_group_type=final_group_type,
            in_order=final_group_order,
            in_num_features=self.num_channels[1],  # Features from decoder
            in_representation="regular",
            out_group_type=final_group_type,
            out_num_features=self.num_classes,
            out_representation="regular",
            domain=self.domain,
            kernel_size=1,
        )
        self.output_conv = self.output_conv.to(device=self.device, dtype=self.dtype)
        
        # Group pooling to remove group dimension
        self.group_pool = GroupPooling3D(
            group_order=final_group_order,
            pooling_type="mean"
        )

    def forward(self, x):
        """
        Complete forward pass through the 4D U-Net for 3D medical image segmentation.
        
        This method performs the complete forward pass from input to segmentation mask,
        including encoder feature extraction, skip connections, and decoder reconstruction.
        
        MATHEMATICAL FLOW:
            1. Encoder Path: Feature extraction with group downsampling
                - Input: (batch, 1, depth, height, width) - trivial representation
                - Layer i: (batch, channels×|G|, depth/2^i, height/2^i, width/2^i) - regular representation
                - Skip connections: Store features after each layer for decoder
                
            2. Decoder Path: Feature reconstruction with group upsampling
                - Skip connections: Concatenate encoder and decoder features
                - Group upsampling: Restore group order for next level
                - Spatial upsampling: Restore spatial dimensions
                - Output: (batch, channels×|G|, depth, height, width) - upsampled features
                
            3. Final Output: Segmentation mask generation
                - Group convolution: Convert to segmentation classes
                - Group pooling: Collapse group dimension
                - Output: (batch, num_classes, depth, height, width) - segmentation mask
                
        GROUP EQUIVARIANCE:
            The entire forward pass maintains group equivariance f(g·x) = g·f(x).
            This means that rotating the input by 90° around the z-axis will
            produce a correspondingly rotated segmentation mask.
            
        SKIP CONNECTIONS:
            Skip connections concatenate encoder and decoder features at the same
            spatial resolution, helping the decoder reconstruct fine details.
            The concatenation is done at the group level to maintain equivariance.
            
        Args:
            x (torch.Tensor): Input tensor with shape (batch, 1, depth, height, width)
                - batch: Batch size
                - 1: Single input channel (grayscale medical image)
                - depth, height, width: 3D spatial dimensions
                
        Returns:
            torch.Tensor: Segmentation mask with shape (batch, num_classes, depth, height, width)
                - batch: Batch size
                - num_classes: Number of segmentation classes
                - depth, height, width: 3D spatial dimensions (same as input)
                
        Example:
            >>> model = Gcnn3DSegmentation(...)
            >>> x = torch.randn(2, 1, 32, 32, 32)  # 2 samples, 1 channel, 32×32×32
            >>> output = model(x)  # (2, 4, 32, 32, 32) for 4-class segmentation
        """
        # Initialize skip connections list for storing encoder features
        skip_connections = []
        
        # ENCODER PATH: Feature extraction with group downsampling
        # This extracts hierarchical features while reducing spatial and group dimensions
        for i in range(self.num_layers):
            # Step 1: Group convolution with ReLU activation
            # This applies the group-equivariant convolution and non-linearity
            x = self.encoder.conv_layers[i](x)  # Group convolution (maintains group order)
            x = torch.relu(x)                   # ReLU activation (preserves equivariance)
            
            # Step 2: Spatial downsampling with anti-aliasing
            # This reduces spatial dimensions while preventing aliasing artifacts
            if self.spatial_subsampling_factors[i] > 1:
                x = self.encoder.spatial_sampling_layers[i](x)  # BlurPool3d for anti-aliased downsampling
            
            # Step 3: Group downsampling (if enabled)
            # This reduces group order while maintaining equivariance
            if self.encoder.sampling_layers[i] is not None:
                x, _ = self.encoder.sampling_layers[i](x)  # Group downsampling
            
            # Step 4: Store skip connection after all downsampling operations
            # This preserves features at the current resolution for decoder concatenation
            skip_connections.append(x)
            
            # Step 5: Dropout regularization
            # This prevents overfitting by randomly zeroing some activations
            if self.dropout_rate > 0:
                x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        # DECODER PATH: Feature reconstruction with group upsampling
        # This reconstructs the segmentation mask using encoder features and skip connections
        for i in range(self.num_layers):
            # Step 1: Get corresponding skip connection
            # Skip connections are used in reverse order (deepest to shallowest)
            skip = skip_connections[self.num_layers - 1 - i]

            # Step 2: Determine group order from skip connection for proper concatenation
            # This ensures the concatenation maintains the correct group structure
            current_layer_idx = self.num_layers - 1 - i
            expected_features = self.num_channels[current_layer_idx + 1]
            skip_group_order = skip.shape[1] // expected_features

            # Step 3: Concatenate skip connection with current features
            # This combines encoder and decoder features for better reconstruction
            x = self._concatenate_skip_connection(x, skip, skip_group_order)

            # Step 4: Hybrid block: convolution + group resampling
            # This applies group convolution and changes group order for next level
            x = self.decoder_blocks[i](x)  # Hybrid convolution + group resampling
            x = torch.relu(x)              # ReLU activation

            # Step 5: Spatial upsampling for next level
            # This restores spatial dimensions to match the corresponding encoder level
            if self.spatial_subsampling_factors[self.num_layers - 1 - i] > 1:
                x = self.decoder_spatial_upsampling[i](x)  # Spatial upsampling

            # Step 6: Dropout regularization
            # This prevents overfitting during training
            if self.dropout_rate > 0:
                x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        # FINAL OUTPUT: Segmentation mask generation
        # Step 1: Group convolution to convert features to segmentation classes
        # This applies the final convolution to produce class predictions
        x = self.output_conv(x)      # (batch, num_classes×group_order, depth, height, width)
        
        # Step 2: Group pooling to collapse group dimension
        # This removes the group dimension to produce the final segmentation mask
        x = self.group_pool(x)       # (batch, num_classes, depth, height, width)
        
        # Note: Final upsampling to match target size is handled by the loss function
        # This allows for flexible handling of size mismatches during training
        
        return x

    def _concatenate_skip_connection(self, x, skip, group_order):
        """
        Concatenate skip connection with explicit group dimension handling.
        
        This method handles the concatenation of encoder and decoder features
        in the U-Net architecture, ensuring proper alignment of both spatial
        and group dimensions while maintaining group equivariance.
        
        MATHEMATICAL APPROACH:
            1. Factor out group and channel dimensions: (B, C*G, D, H, W) -> (B, C, G, D, H, W)
            2. Concatenate along channel dimension: (B, C1+C2, G, D, H, W)  
            3. Reshape back: (B, (C1+C2)*G, D, H, W)
            
        GROUP EQUIVARIANCE:
            The concatenation preserves group equivariance because both tensors
            have the same group structure (group_order). The operation is applied
            uniformly across all group elements.
            
        SPATIAL ALIGNMENT:
            When spatial dimensions don't match, trilinear interpolation is used
            to upsample the skip connection to match the current decoder features.
            This preserves the spatial structure while allowing for size differences.
            
        Args:
            x (torch.Tensor): Current decoder features
                - Shape: (B, C1*G, D, H, W) where C1 is decoder channels, G is group order
                - Current features from decoder path
                
            skip (torch.Tensor): Skip connection from encoder
                - Shape: (B, C2*G, D_skip, H_skip, W_skip) where C2 is encoder channels
                - Features from corresponding encoder layer
                
            group_order (int): Number of group elements G
                - Used to factor out group and channel dimensions
                - Must match between x and skip for proper concatenation
                
        Returns:
            torch.Tensor: Concatenated features
                - Shape: (B, (C1+C2)*G, D, H, W)
                - Combined features from encoder and decoder
                
        Example:
            >>> x = torch.randn(2, 64*24, 8, 8, 8)      # Decoder features
            >>> skip = torch.randn(2, 32*24, 16, 16, 16) # Encoder features
            >>> group_order = 24
            >>> concat = model._concatenate_skip_connection(x, skip, group_order)
            >>> print(concat.shape)  # (2, 96*24, 8, 8, 8)
        """
        # Extract tensor dimensions for processing
        B, CG_x, D, H, W = x.shape                    # Current decoder features
        B_skip, CG_skip, D_skip, H_skip, W_skip = skip.shape  # Skip connection features
        
        # Factor out group and channel dimensions
        # This separates the combined channel×group dimension into individual components
        C_x = CG_x // group_order      # Decoder channels (channels per group element)
        C_skip = CG_skip // group_order  # Encoder channels (channels per group element)
        
        # Debug information for troubleshooting concatenation issues
        print(f"Concatenating: x=({C_x}*{group_order}={CG_x}) + skip=({C_skip}*{group_order}={CG_skip})")
        print(f"Actual shapes: x={x.shape}, skip={skip.shape}")
        print(f"Expected skip shape: ({B}, {C_skip}, {group_order}, {D}, {H}, {W})")
        
        # Check if skip connection needs spatial upsampling to match current tensor
        # This is common in U-Net architectures where encoder and decoder have different spatial resolutions
        if skip.shape[2:] != (D, H, W):
            print(f"Skip connection spatial dims {skip.shape[2:]} don't match current {D, H, W}, upsampling...")
            # Upsample skip connection to match current spatial dimensions
            skip = F.interpolate(skip, size=(D, H, W), mode='trilinear', align_corners=False)
            print(f"Upsampled skip shape: {skip.shape}")
        
        # Step 1: Reshape to separate group dimension
        # This separates the combined channel×group dimension into individual components
        # From (B, C*G, D, H, W) to (B, C, G, D, H, W)
        x_reshaped = x.view(B, C_x, group_order, D, H, W)        # Decoder features
        skip_reshaped = skip.view(B, C_skip, group_order, D, H, W)  # Encoder features
        
        # Step 2: Concatenate along channel dimension (axis 1)
        # This combines features from encoder and decoder at the same group resolution
        # Result: (B, C_x+C_skip, G, D, H, W)
        concatenated = torch.cat([x_reshaped, skip_reshaped], dim=1)
        
        # Step 3: Reshape back to combined channel×group format
        # This restores the original tensor format for further processing
        # From (B, C_x+C_skip, G, D, H, W) to (B, (C_x+C_skip)*G, D, H, W)
        C_total = C_x + C_skip  # Total number of channels after concatenation
        result = concatenated.view(B, C_total * group_order, D, H, W)
        
        # Debug information for verification
        print(f"Result: ({C_total}*{group_order}={C_total * group_order})")
        
        return result

    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']

        # Move encoder
        self.encoder = self.encoder.to(*args, **kwargs)
        
        # Move decoder components
        for layer in self.decoder_upsampling:
            if layer is not None:
                layer = layer.to(*args, **kwargs)
        for layer in self.decoder_spatial_upsampling:
            if hasattr(layer, 'to'):
                layer = layer.to(*args, **kwargs)
        for layer in self.decoder_convs:
            layer = layer.to(*args, **kwargs)
        
        self.output_conv = self.output_conv.to(*args, **kwargs)
        
        super().to(*args, **kwargs)
        return self


class GroupUpsample3D(nn.Module):
    """Group upsampling layer for reversing group downsampling."""
    
    def __init__(
        self,
        from_group_type: str,
        to_group_type: str,
        from_group_order: int,
        to_group_order: int,
        num_features: int,
        apply_antialiasing: bool = False,
        anti_aliasing_kwargs: dict = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        upsampling_factor = to_group_order // from_group_order
        
        print(f"Creating group upsampling: {from_group_type}({from_group_order}) -> {to_group_type}({to_group_order}), factor={upsampling_factor}")
        
        self.downsampling_layer = SubgroupDownsample(
            group_type=to_group_type,
            order=to_group_order,
            sub_group_type=from_group_type,
            subsampling_factor=upsampling_factor,
            num_features=num_features,
            generator="r-s",
            device=device,
            dtype=dtype,
            sample_type="sample",
            apply_antialiasing=apply_antialiasing,
            anti_aliasing_kwargs=anti_aliasing_kwargs or {},
        )

    def forward(self, x):
        """Upsample from subgroup to main group."""
        return self.downsampling_layer.upsample(x)


class SpatialUpsample3D(nn.Module):
    """3D spatial upsampling layer."""
    
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        
        if scale_factor == 1:
            self.upsample = nn.Identity()
        else:
            # Use interpolation for adaptive channel handling
            self.upsample = nn.Upsample(
                scale_factor=scale_factor,
                mode='trilinear',
                align_corners=False
            )

    def forward(self, x):
        """Upsample spatial dimensions."""
        return self.upsample(x)


class GroupPooling3D(nn.Module):
    """Pooling layer to reduce group dimension for segmentation output."""
    
    def __init__(self, group_order: int, pooling_type: str = "mean"):
        super().__init__()
        self.group_order = group_order
        self.pooling_type = pooling_type
    
    def forward(self, x):
        """
        Pool over group dimension to get per-voxel predictions.
        
        Args:
            x: Input tensor (B, C*G, D, H, W)
            
        Returns:
            Output tensor (B, C, D, H, W)
        """
        # Reshape to separate group dimension
        B, CG, D, H, W = x.shape
        C = CG // self.group_order
        x = x.view(B, C, self.group_order, D, H, W)
        
        # Pool over group dimension (axis 2)
        if self.pooling_type == "mean":
            x = torch.mean(x, dim=2)
        elif self.pooling_type == "max":
            x = torch.max(x, dim=2)[0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        return x  # (B, C, D, H, W)
