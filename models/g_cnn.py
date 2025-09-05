import torch
import torch.nn as nn
from gsampling.layers.rnconv import *
from gsampling.thirdparty.blurpool2d import BlurPool2d
from gsampling.layers.downsampling import *
from gsampling.utils.group_utils import *
from einops import rearrange


class Gcnn(nn.Module):
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
        fully_convolutional=False,
        layer_kwargs={}
    ) -> None:
        """Group Equivariant Convolutional Neural Network with Anti-Aliased Subsampling.

        Implements a deep network with:
        - Group-equivariant convolutions
        - Spectral anti-aliasing for group subsampling
        - Spatial anti-aliasing via blur pooling
        - Adaptive group-space pooling

        Parameters:
            num_layers (list[int]): Number of layers
            num_channels (list[int]): Channels per layer [input, layer1, ..., layerN]
            kernel_sizes (list[int]): Convolution kernel sizes per layer
            num_classes (int): Output classes for classification
            dwn_group_types (list): Tuples of (group_type, subgroup_type) per layer
            init_group_order (int): Initial group order (|G₀|)
            spatial_subsampling_factors (list[int]): Spatial stride factors per layer
            subsampling_factors (list[int]): Group subsampling ratios per layer (|Gₙ|/|Gₙ₊₁|)
            domain (int): Input dimension (2 for images)
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

            4. Spatial BlurPool:
                Implements [Zhang19]'s anti-aliased downsampling:
                x↓ = (x * k)↓ where k is low-pass filter

        Forward Pass:
            Input shape: (B, C, H, W) → lifted to (B, C*|G|, H, W)
            For each layer:
                1. G-equivariant conv + ReLU
                2. Spatial subsampling with blur
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
        self.fully_convolutional = fully_convolutional

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
                print("Antialiasing Condition at layer ", i, ": ", self.apply_antialiasing)
                sampling_layer = SubgroupDownsample(
                    group_type=self.dwn_group_types[i][0],
                    order=current_group_order,
                    sub_group_type=self.dwn_group_types[i][1],
                    subsampling_factor=subsampling_factors[i],
                    num_features=num_channels[i + 1],
                    generator="r-s",
                    device="cpu",
                    dtype=torch.float32,
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

            self.conv_layers.append(conv)

            if self.spatial_subsampling_factors[i] > 1:
                spatial_sampling_layer = BlurPool2d(
                    channels=num_channels[i + 1] * conv.G_out.order(),
                    stride=self.spatial_subsampling_factors[i],
                )
            else:
                spatial_sampling_layer = nn.Identity()

            self.sampling_layers.append(sampling_layer)
            self.spatial_sampling_layers.append(spatial_sampling_layer)

            if sampling_layer is not None:
                current_group_order = sampling_layer.sub_order

        self.last_g_size = get_group(
            dwn_group_types[-1][1], current_group_order
        ).order()

        self.dropout_layer = nn.Dropout(p=0.3)
        self.linear_layer = nn.Linear(num_channels[-1], num_classes)

    def pooling(self, x):
        x = rearrange(x, "b (c g) h w -> b c (g h w)", g=self.last_g_size)
        if self.pooling_type == "max":
            x = torch.max(x, dim=-1)[0]
        elif self.pooling_type == "mean":
            x = torch.mean(x, dim=-1)
        return x

    def get_feature(self, x):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            if self.spatial_subsampling_factors[i] > 1:
                x = self.spatial_sampling_layers[i](x)

            if self.sampling_layers[i] is not None:
                x, _ = self.sampling_layers[i](x)

            if self.dropout_rate > 0:
                x = nn.functional.dropout(
                    x, p=self.dropout_rate, training=self.training
                )
        x = self.pooling(x)

        return x

    def get_hidden_feature(self, x):
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
        x = self.get_feature(x)
        if not self.fully_convolutional:
            x = self.linear_layer(x)
        return x
