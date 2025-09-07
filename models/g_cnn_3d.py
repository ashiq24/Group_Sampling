"""
3D Group Equivariant Convolutional Neural Network (GCNN) Implementation

This module implements a 3D Group Equivariant CNN that maintains equivariance
under group transformations, specifically designed for 3D medical image analysis.

MATHEMATICAL FOUNDATIONS:
- Group Theory: Implements equivariance under octahedral group O (24 elements)
- Cyclic Subgroup: Uses C4 cyclic subgroup (4 elements) for 90° rotations around z-axis
- Group Convolution: f * ψ where f ∈ L²(G), ψ ∈ L²(G)
- Channel Calculation: total_channels = base_channels × group_order
- Anti-aliasing: Prevents aliasing artifacts during group downsampling

ARCHITECTURE:
- Layer 0: Trivial → Regular representation (1 × |G| channels)
- Layer 1+: Hybrid layers with group convolution + group resampling
- Spatial Pooling: BlurPool3d for anti-aliased spatial downsampling
- Global Pooling: Collapse group and spatial dimensions for classification

GROUP PROCESSING:
- Input: (batch, 1, depth, height, width) - trivial representation
- Layer 0: (batch, 1×24, depth, height, width) - regular representation
- Group downsampling: (batch, channels×4, depth, height, width) - C4 subgroup
- Group upsampling: (batch, channels×24, depth, height, width) - back to octahedral
- Output: (batch, num_classes) - classification logits

EQUIVARIANCE PROPERTY:
The network maintains group equivariance: f(g·x) = g·f(x) for all group elements g.
This means that rotating the input by 90° around the z-axis will rotate the
output by the same amount, providing robust rotation-invariant features.

USAGE:
    model = Gcnn3D(
        num_layers=3,
        num_channels=[1, 32, 64, 128],
        kernel_sizes=[3, 3, 3],
        num_classes=10,
        dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "cycle"], ["cycle", "cycle"]],
        subsampling_factors=[1, 6, 1],
        init_group_order=24,
        spatial_subsampling_factors=[2, 2, 1],
        domain=3,
        pooling_type="max",
        apply_antialiasing=True,
        antialiasing_kwargs={"smoothness_loss_weight": 0.1}
    )
"""

import torch
import torch.nn as nn
from gsampling.layers.rnconv import *  # Group convolution layers
from gsampling.layers.downsampling import *  # Group downsampling layers
from gsampling.utils.group_utils import *  # Group utilities and creation functions
from einops import rearrange  # Tensor reshaping operations
from .hybrid import HybridConvGroupResample3D  # Hybrid convolution + group resampling layer


class Gcnn3D(nn.Module):
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
        pooling_type,
        apply_antialiasing,
        antialiasing_kwargs,
        dropout_rate,
        dtype=torch.float32,
        device="cpu",
        fully_convolutional=False,
        layer_kwargs={}
    ) -> None:
        """
        Initialize 3D Group Equivariant Convolutional Neural Network with Anti-Aliased Subsampling.

        This constructor builds a complete 3D Group Equivariant CNN architecture that maintains
        equivariance under group transformations. The network processes 3D medical images while
        preserving rotation equivariance through group theory principles.

        ARCHITECTURE OVERVIEW:
        - Layer 0: Trivial → Regular representation transition (1 × |G| channels)
        - Layers 1+: Hybrid layers combining group convolution + group resampling
        - Spatial Pooling: BlurPool3d for anti-aliased spatial downsampling
        - Global Pooling: Collapse group and spatial dimensions for classification

        GROUP PROCESSING FLOW:
        1. Input: (batch, 1, depth, height, width) - trivial representation
        2. Layer 0: (batch, 1×24, depth, height, width) - regular representation
        3. Group downsampling: (batch, channels×4, depth, height, width) - C4 subgroup
        4. Group upsampling: (batch, channels×24, depth, height, width) - back to octahedral
        5. Output: (batch, num_classes) - classification logits

        MATHEMATICAL FOUNDATIONS:
        - Group Equivariance: f(g·x) = g·f(x) for all group elements g
        - Group Convolution: (f * ψ)(g) = Σ_{h∈G} f(h)ψ(h⁻¹g)
        - Channel Calculation: total_channels = base_channels × group_order
        - Anti-aliasing: Prevents aliasing artifacts during group downsampling

        Args:
            num_layers (int): Number of convolutional layers in the network
                - Must be ≥ 1
                - Each layer processes both spatial and group dimensions
                
            num_channels (list[int]): Channel progression for each layer
                - Length must be num_layers + 1
                - Format: [input_channels, layer1_channels, ..., output_channels]
                - Example: [1, 32, 64, 128] for 3 layers with 1→32→64→128 channels
                
            kernel_sizes (list[int]): 3D convolution kernel sizes for each layer
                - Length must be num_layers
                - Common values: 3 (3×3×3 kernels), 5 (5×5×5 kernels)
                - Must be odd numbers for proper padding
                
            num_classes (int): Number of output classes for classification
                - For OrganMNIST3D: 11 organ classes
                - For binary classification: 2
                - Determines output tensor shape: (batch, num_classes)
                
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
                
            pooling_type (str): Global pooling type for classification
                - Options: "max", "mean"
                - Applied after feature extraction to collapse spatial dimensions
                
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
                
            fully_convolutional (bool): Whether to use fully convolutional mode
                - False: Global pooling + classification head (default)
                - True: No global pooling, output spatial features
                
            layer_kwargs (dict): Additional layer-specific parameters
                - Currently unused, reserved for future extensions

        Raises:
            ValueError: If parameter dimensions don't match or invalid values provided
            AssertionError: If group order calculations result in invalid values
            
        Example:
            >>> model = Gcnn3D(
            ...     num_layers=3,
            ...     num_channels=[1, 32, 64, 128],
            ...     kernel_sizes=[3, 3, 3],
            ...     num_classes=11,
            ...     dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "cycle"], ["cycle", "cycle"]],
            ...     subsampling_factors=[1, 6, 1],
            ...     init_group_order=24,
            ...     spatial_subsampling_factors=[2, 2, 1],
            ...     domain=3,
            ...     pooling_type="max",
            ...     apply_antialiasing=True,
            ...     antialiasing_kwargs={"smoothness_loss_weight": 0.1}
            ... )
            >>> x = torch.randn(2, 1, 28, 28, 28)  # (batch, channels, depth, height, width)
            >>> output = model(x)  # (batch, num_classes)
        """

        Mathematical Operations:
            1. Group Convolution:
                f * ψ(g⁻¹·) = ∫_G f(h)ψ(g⁻¹h)dh
                Implemented via rnConv layers with G-equivariant kernels

            2. Spectral Subsampling:
                S: L²(G) → L²(H) via S = Π_H∘R_G
                Where R_G is Reynolds projection and Π_H is subgroup restriction

            3. Anti-Aliasing:
                X̃ = L1_projector·X̂ before subsampling
                L1_projector removes high-frequency components beyond Nyquist limit

            4. 3D Spatial BlurPool:
                Implements [Zhang19]'s anti-aliased downsampling for 3D:
                x↓ = (x * k)↓ where k is 3D low-pass filter

        Forward Pass:
            Input shape: (B, C, D, H, W) → lifted to (B, C*|G|, D, H, W)
            For each layer:
                1. G-equivariant 3D conv + ReLU
                2. 3D spatial subsampling with blur
                3. Group subsampling with anti-aliasing
                4. Dropout
            Final pooling: Collapse (group × spatial) dimensions via max/mean
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        self.dwn_group_types = dwn_group_types
        self.spatial_subsampling_factors = spatial_subsampling_factors
        self.subsampling_factors = subsampling_factors
        self.domain = domain
        self.pooling_type = pooling_type
        self.apply_antialiasing = apply_antialiasing
        self.antialiasing_kwargs = antialiasing_kwargs
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        self.device = device
        self.fully_convolutional = fully_convolutional

        # Validate domain parameter - must be 3 for 3D data
        if domain != 3:
            raise ValueError(f"Gcnn3D requires domain=3, got {domain}")

        # Initialize module lists for storing network layers
        self.conv_layers = nn.ModuleList()          # Convolutional layers (rnConv or Hybrid)
        self.sampling_layers = nn.ModuleList()      # Group sampling layers (now handled by Hybrid)
        self.spatial_sampling_layers = nn.ModuleList()  # Spatial sampling layers (BlurPool3d)
        
        # Track current group order throughout the network
        # This is crucial for calculating channel dimensions correctly
        current_group_order = init_group_order

        # Build the network layer by layer
        for i in range(num_layers):
            # Determine input representation type for this layer
            if i == 0:
                rep = "trivial"  # First layer always takes trivial representation input
                in_features = num_channels[i]  # Input channels (typically 1 for medical images)
            else:
                rep = "regular"  # Subsequent layers use regular representation
                in_features = num_channels[i]  # Input channels for this layer

            # Group sampling is now handled by the hybrid layer
            # This simplifies the architecture by combining convolution and sampling
            sampling_layer = None

            if i == 0:
                # FIRST LAYER: Trivial → Regular representation transition
                # This is the only layer that changes representation type
                # Input: (batch, 1, depth, height, width) - trivial representation
                # Output: (batch, 1×24, depth, height, width) - regular representation
                conv = rnConv(
                    in_group_type=self.dwn_group_types[i][0],  # Input group type (octahedral)
                    in_order=current_group_order,              # Input group order (24)
                    in_num_features=in_features,               # Input features (1)
                    in_representation="trivial",               # Input representation type
                    out_group_type=self.dwn_group_types[i][0], # Output group type (octahedral)
                    out_num_features=num_channels[i + 1],      # Output features (32)
                    out_representation="regular",              # Output representation type
                    domain=domain,                             # Spatial domain (3)
                    kernel_size=kernel_sizes[i],               # Kernel size (3)
                    layer_kwargs=layer_kwargs,                 # Additional layer parameters
                )
            else:
                # SUBSEQUENT LAYERS: Use hybrid layer for group type transitions
                # This combines group convolution with group resampling
                # Handles both same-group and cross-group transitions
                
                # Calculate output group order based on target group type
                target_group_type = self.dwn_group_types[i][1]
                if target_group_type == "cycle":
                    # For cyclic groups (C4), use the subsampling factor
                    # Example: 24 / 6 = 4 (octahedral → C4)
                    out_group_order = current_group_order // subsampling_factors[i]
                else:
                    # For other groups (octahedral, etc.), use the full group order
                    # This handles transitions like C4 → octahedral
                    out_group_order = get_group(target_group_type, 1).order()
                
                # Create hybrid layer that combines convolution and group resampling
                conv = HybridConvGroupResample3D(
                    in_group_type=self.dwn_group_types[i][0],  # Input group type
                    in_group_order=current_group_order,        # Input group order
                    in_num_features=in_features,               # Input features
                    out_group_type=self.dwn_group_types[i][1],
                    out_group_order=out_group_order,
                    out_num_features=num_channels[i + 1],
                    representation="regular",
                    kernel_size=kernel_sizes[i],
                    domain=domain,
                    apply_antialiasing=self.apply_antialiasing,
                    anti_aliasing_kwargs=self.antialiasing_kwargs,
                    device=self.device,
                    dtype=self.dtype,
                )

            # Move convolution layer to correct device and dtype
            # This ensures all computations happen on the specified device (CPU/GPU)
            conv = conv.to(device=self.device, dtype=self.dtype)
            self.conv_layers.append(conv)

            # CRITICAL: Update group order based on hybrid layer output BEFORE creating spatial sampling layer
            # This is essential for calculating the correct number of channels for spatial pooling
            if subsampling_factors[i] > 1:
                # Group downsampling: reduce group order by subsampling factor
                # Example: 24 / 6 = 4 (octahedral → C4)
                current_group_order = current_group_order // subsampling_factors[i]
            else:
                # No group downsampling: keep same group order
                # This handles cases where group type changes but order stays the same
                current_group_order = get_group(self.dwn_group_types[i][1], current_group_order).order()

            # Create spatial sampling layer (3D BlurPool)
            if self.spatial_subsampling_factors[i] > 1:
                # 3D Blur Pooling for anti-aliased spatial downsampling
                # This implements the anti-aliased downsampling from [Zhang19]
                # Formula: x↓ = (x * k)↓ where k is a 3D low-pass filter
                
                # CRITICAL: Use updated group order for channel calculation
                # Total channels = base_channels × current_group_order
                spatial_sampling_layer = BlurPool3d(
                    channels=num_channels[i + 1] * current_group_order,  # Total channels after group processing
                    stride=self.spatial_subsampling_factors[i],          # Spatial downsampling factor
                )
                # Move to correct device and dtype
                spatial_sampling_layer = spatial_sampling_layer.to(device=self.device, dtype=self.dtype)
            else:
                # No spatial downsampling: use identity layer
                spatial_sampling_layer = nn.Identity()

            # Store layers in module lists
            self.sampling_layers.append(sampling_layer)          # Group sampling (handled by Hybrid)
            self.spatial_sampling_layers.append(spatial_sampling_layer)  # Spatial sampling (BlurPool3d)

        self.last_g_size = get_group(
            dwn_group_types[-1][1], current_group_order
        ).order()

        # We'll determine the actual output features dynamically after the first forward pass
        # For now, use the expected number of features
        self.expected_output_features = num_channels[-1]
        self.init_group_order = init_group_order  # Add this for compatibility with tests

        self.dropout_layer = nn.Dropout(p=0.3)
        # We'll create the linear layer after we know the actual output dimensions
        self.linear_layer = None
        self.num_classes = num_classes

    def pooling_3d(self, x):
        """
        3D pooling that collapses group and spatial dimensions for classification.
        
        This method performs global pooling over both group and spatial dimensions
        to produce a fixed-size feature vector for classification. The pooling
        operation maintains the group equivariance property.
        
        Mathematical Details:
            - Input: (batch, channels×group_order, depth, height, width)
            - Reshape: (batch, channels, group_order×depth×height×width)
            - Pool: max or mean over the flattened dimension
            - Output: (batch, channels)
            
        Group Equivariance:
            The pooling operation preserves group equivariance because it operates
            uniformly over all group elements and spatial locations.
            
        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels×group_order, depth, height, width)
            
        Returns:
            torch.Tensor: Pooled tensor with shape (batch, channels)
        """
        # Reshape to separate channel and group dimensions
        # From (batch, channels×group_order, depth, height, width)
        # To (batch, channels, group_order×depth×height×width)
        x = rearrange(x, "b (c g) d h w -> b c (g d h w)", g=self.last_g_size)
        
        # Apply global pooling over the flattened dimension
        if self.pooling_type == "max":
            # Max pooling: take maximum value across all group and spatial positions
            x = torch.max(x, dim=-1)[0]  # (batch, channels)
        elif self.pooling_type == "mean":
            # Mean pooling: take average value across all group and spatial positions
            x = torch.mean(x, dim=-1)    # (batch, channels)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        return x

    def get_feature(self, x):
        """
        Forward pass through the 3D GCNN feature extractor.
        
        This method performs the complete feature extraction pipeline including:
        - Group convolution layers with ReLU activation
        - Spatial downsampling with anti-aliasing
        - Dropout regularization
        - Optional global pooling for classification
        
        Mathematical Details:
            For each layer i:
                1. Group convolution: x = conv_i(x)
                2. ReLU activation: x = ReLU(x)
                3. Spatial downsampling: x = spatial_pool_i(x) if stride > 1
                4. Dropout: x = dropout(x) if dropout_rate > 0
                
        Group Equivariance:
            Each operation maintains group equivariance f(g·x) = g·f(x):
            - Group convolution: inherently equivariant
            - ReLU: pointwise operation preserves equivariance
            - Spatial pooling: uniform operation preserves equivariance
            - Dropout: random but equivariant when applied consistently
            
        Args:
            x (torch.Tensor): Input tensor with shape (batch, 1, depth, height, width)
            
        Returns:
            torch.Tensor: Feature tensor with shape:
                - Classification: (batch, channels) if not fully_convolutional
                - Dense prediction: (batch, channels×group_order, depth, height, width) if fully_convolutional
        """
        # Process through each layer sequentially
        for i in range(self.num_layers):
            # Step 1: Group convolution with ReLU activation
            # This applies the group-equivariant convolution and non-linearity
            x = self.conv_layers[i](x)  # Group convolution (maintains group order)
            x = torch.relu(x)           # ReLU activation (preserves equivariance)

            # Step 2: Spatial downsampling with anti-aliasing
            # This reduces spatial dimensions while preventing aliasing artifacts
            if self.spatial_subsampling_factors[i] > 1:
                x = self.spatial_sampling_layers[i](x)  # BlurPool3d for anti-aliased downsampling

            # Step 3: Dropout regularization
            # This prevents overfitting by randomly zeroing some activations
            if self.dropout_rate > 0:
                x = nn.functional.dropout(
                    x, p=self.dropout_rate, training=self.training
                )

        # Step 4: Global pooling for classification (optional)
        # Only apply pooling if not in fully convolutional mode
        if not self.fully_convolutional:
            x = self.pooling_3d(x)  # Collapse group and spatial dimensions
        
        return x

    def get_hidden_feature(self, x):
        """Extract features before and after sampling layers."""
        feature_before_sampling = []
        feature_after_sampling = []
        sampling_layers = []

        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = torch.relu(x)

            if self.spatial_subsampling_factors[i] > 1:
                x = self.spatial_sampling_layers[i](x)

            feature_before_sampling.append(x.clone().detach())

            if self.sampling_layers[i] is not None:
                x, _ = self.sampling_layers[i](x)

            sampling_layers.append(self.sampling_layers[i])
            feature_after_sampling.append(x.clone().detach())

        return feature_before_sampling, feature_after_sampling, sampling_layers

    def forward(self, x):
        """
        Complete forward pass through the 3D Group Equivariant CNN.
        
        This method performs the complete forward pass from input to output,
        including feature extraction and optional classification head.
        
        MATHEMATICAL FLOW:
            1. Feature Extraction: x → get_feature(x)
                - Group convolution layers with ReLU
                - Spatial downsampling with anti-aliasing
                - Dropout regularization
                - Global pooling (if not fully convolutional)
                
            2. Classification (if not fully convolutional):
                - Linear layer: features → logits
                - Output: (batch, num_classes)
                
            3. Dense Prediction (if fully convolutional):
                - Output: (batch, channels×group_order, depth, height, width)
                
        GROUP EQUIVARIANCE:
            The entire forward pass maintains group equivariance f(g·x) = g·f(x).
            This means that rotating the input by 90° around the z-axis will
            produce a correspondingly rotated output.
            
        Args:
            x (torch.Tensor): Input tensor with shape (batch, 1, depth, height, width)
                - batch: Batch size
                - 1: Single input channel (grayscale medical image)
                - depth, height, width: 3D spatial dimensions
                
        Returns:
            torch.Tensor: Output tensor with shape:
                - Classification: (batch, num_classes) - class logits
                - Dense prediction: (batch, channels×group_order, depth, height, width) - spatial features
                
        Example:
            >>> model = Gcnn3D(...)
            >>> x = torch.randn(2, 1, 28, 28, 28)  # 2 samples, 1 channel, 28×28×28
            >>> output = model(x)  # (2, num_classes) for classification
        """
        # Step 1: Feature extraction through the complete network
        # This applies all convolutional layers, pooling, and regularization
        x = self.get_feature(x)
        
        # Step 2: Classification head (if not in fully convolutional mode)
        if not self.fully_convolutional:
            # Create linear layer on first forward pass if not created yet
            # This is done dynamically to handle variable input sizes
            if self.linear_layer is None:
                actual_features = x.shape[1]  # Get actual number of features from feature extraction
                # Create linear layer on the same device as input tensor
                device = x.device
                self.linear_layer = nn.Linear(
                    actual_features,      # Input features from feature extraction
                    self.num_classes,    # Output classes for classification
                    dtype=self.dtype,    # Match model dtype
                    device=device        # Match input device
                )
            else:
                # Ensure linear layer is on the same device as input
                # This handles cases where input is moved to different device
                if self.linear_layer.weight.device != x.device:
                    self.linear_layer = self.linear_layer.to(x.device)
            
            # Apply linear layer for classification
            x = self.linear_layer(x)  # (batch, features) → (batch, num_classes)
        
        return x

    def to(self, *args, **kwargs):
        """Move the model to the specified device and dtype."""
        # Update device and dtype attributes
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']

        # Move all components to the specified device/dtype
        self.dropout_layer = self.dropout_layer.to(*args, **kwargs)
        if self.linear_layer is not None:
            self.linear_layer = self.linear_layer.to(*args, **kwargs)

        for layer in self.conv_layers:
            layer = layer.to(*args, **kwargs)
        for layer in self.spatial_sampling_layers:
            if hasattr(layer, 'to'):
                layer = layer.to(*args, **kwargs)
        for layer in self.sampling_layers:
            if layer is not None and hasattr(layer, 'to'):
                layer = layer.to(*args, **kwargs)

        # Also move the model itself using super().to()
        super().to(*args, **kwargs)

        return self


class BlurPool3d(nn.Module):
    """3D Blur Pooling layer for anti-aliased downsampling."""

    def __init__(self, channels, stride):
        super(BlurPool3d, self).__init__()
        self.channels = channels
        self.stride = stride

        if stride == 1:
            self.blur = nn.Identity()
        else:
            # Create 3D blur kernel
            kernel_size = 3
            padding = 1

            # 3D blur kernel (separable)
            kernel_1d = torch.tensor([1, 2, 1], dtype=torch.float32)
            kernel_1d = kernel_1d / kernel_1d.sum()

            # Create 3D kernel by outer product
            kernel_3d = torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d)
            kernel_3d = kernel_3d.expand(1, 1, *kernel_3d.shape).contiguous()

            self.blur = nn.Conv3d(
                channels, channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=channels,
                bias=False,
                padding_mode='replicate'
            )

            # Set kernel weights
            with torch.no_grad():
                kernel_3d = kernel_3d.expand(channels, 1, *kernel_3d.shape[2:])
                self.blur.weight.data = kernel_3d

            # Freeze weights
            self.blur.weight.requires_grad = False

    def forward(self, x):
        return self.blur(x)
