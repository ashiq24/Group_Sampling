import torch
import torch.nn as nn
from gsampling.layers.rnconv import *
from gsampling.thirdparty.blurpool2d import BlurPool2d
from gsampling.layers.downsampling import *
from gsampling.utils.group_utils import *
from einops import rearrange


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
        canonicalize,
        antialiasing_kwargs,
        dropout_rate,
        dtype=torch.float32,
        device="cpu",
        fully_convolutional=False,
        layer_kwargs={}
    ) -> None:
        """3D Group Equivariant Convolutional Neural Network with Anti-Aliased Subsampling.

        Implements a deep network with:
        - Group-equivariant convolutions for 3D data
        - Spectral anti-aliasing for group subsampling
        - Spatial anti-aliasing via 3D blur pooling
        - Adaptive group-space pooling
        - Support for both 2D and 3D groups on 3D data

        Parameters:
            num_layers (list[int]): Number of layers
            num_channels (list[int]): Channels per layer [input, layer1, ..., layerN]
            kernel_sizes (list[int]): Convolution kernel sizes per layer (3D kernels)
            num_classes (int): Output classes for classification
            dwn_group_types (list): Tuples of (group_type, subgroup_type) per layer
            init_group_order (int): Initial group order (|G₀|)
            spatial_subsampling_factors (list[int]): Spatial stride factors per layer (3D strides)
            subsampling_factors (list[int]): Group subsampling ratios per layer (|Gₙ|/|Gₙ₊₁|)
            domain (int): Input dimension (3 for 3D data)
            pooling_type (str): Final pooling method ('max' or 'mean')
            apply_antialiasing (bool): Enable spectral anti-aliasing in group subsampling
            canonicalize (bool): Standardize group element ordering
            antialiasing_kwargs (dict): Parameters for AntiAliasingLayer
            dropout_rate (float): Dropout probability
            fully_convolutional (bool): Skip final linear layer for dense prediction

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
        self.canonicalize = canonicalize
        self.antialiasing_kwargs = antialiasing_kwargs
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        self.device = device
        self.fully_convolutional = fully_convolutional

        # Validate domain
        if domain != 3:
            raise ValueError(f"Gcnn3D requires domain=3, got {domain}")

        self.conv_layers = nn.ModuleList()
        self.sampling_layers = nn.ModuleList()
        self.spatial_sampling_layers = nn.ModuleList()
        current_group_order = init_group_order

        for i in range(num_layers):
            if i == 0:
                rep = "trivial"
            else:
                rep = "regular"

            if subsampling_factors[i] > 1:
                print("Antialiasing", self.apply_antialiasing)
                sampling_layer = SubgroupDownsample(
                    group_type=self.dwn_group_types[i][0],
                    order=current_group_order,
                    sub_group_type=self.dwn_group_types[i][1],
                    subsampling_factor=subsampling_factors[i],
                    num_features=num_channels[i + 1],
                    generator="r-s",
                    device="cpu",
                    dtype=self.dtype,
                    sample_type="sample",
                    apply_antialiasing=self.apply_antialiasing,
                    anti_aliasing_kwargs=self.antialiasing_kwargs,
                    cannonicalize=self.canonicalize,
                )
            else:
                sampling_layer = None

            conv = rnConv(
                in_group_type=self.dwn_group_types[i][0],
                in_order=current_group_order,
                in_num_features=num_channels[i],
                in_representation=rep,
                out_group_type=self.dwn_group_types[i][0],
                out_num_features=num_channels[i + 1],
                out_representation="regular",
                domain=domain,
                kernel_size=kernel_sizes[i],
                layer_kwargs=layer_kwargs,
            )

            # Move conv layer to correct device and dtype
            conv = conv.to(device=self.device, dtype=self.dtype)
            self.conv_layers.append(conv)

            if self.spatial_subsampling_factors[i] > 1:
                # 3D Blur Pooling
                spatial_sampling_layer = BlurPool3d(
                    channels=num_channels[i + 1] * conv.G_out.order(),
                    stride=self.spatial_subsampling_factors[i],
                )
                # Move to correct device and dtype
                spatial_sampling_layer = spatial_sampling_layer.to(device=self.device, dtype=self.dtype)
            else:
                spatial_sampling_layer = nn.Identity()

            self.sampling_layers.append(sampling_layer)
            self.spatial_sampling_layers.append(spatial_sampling_layer)

            if sampling_layer is not None:
                current_group_order = sampling_layer.sub_order

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
        """3D pooling that collapses group and spatial dimensions."""
        x = rearrange(x, "b (c g) d h w -> b c (g d h w)", g=self.last_g_size)
        if self.pooling_type == "max":
            x = torch.max(x, dim=-1)[0]
        elif self.pooling_type == "mean":
            x = torch.mean(x, dim=-1)
        return x

    def get_feature(self, x):
        """Forward pass through the 3D GCNN feature extractor."""
        for i in range(self.num_layers):
            print(f"Layer {i}: input shape {x.shape}")
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            print(f"  After conv: {x.shape}")

            if self.spatial_subsampling_factors[i] > 1:
                x = self.spatial_sampling_layers[i](x)
                print(f"  After spatial subsampling: {x.shape}")

            if self.sampling_layers[i] is not None:
                x, _ = self.sampling_layers[i](x)
                print(f"  After group subsampling: {x.shape}")

            if self.dropout_rate > 0:
                x = nn.functional.dropout(
                    x, p=self.dropout_rate, training=self.training
                )

        # Only apply pooling if not fully convolutional
        if not self.fully_convolutional:
            x = self.pooling_3d(x)
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
        """Forward pass through the 3D GCNN."""
        x = self.get_feature(x)
        if not self.fully_convolutional:
            # Create linear layer on first forward pass if not created yet
            if self.linear_layer is None:
                actual_features = x.shape[1]
                print(f"Creating linear layer: {actual_features} -> {self.num_classes} (dtype: {self.dtype}, device: {self.device})")
                self.linear_layer = nn.Linear(actual_features, self.num_classes, dtype=self.dtype, device=self.device)
            x = self.linear_layer(x)
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
