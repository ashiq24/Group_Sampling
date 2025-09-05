import torch
import torch.nn as nn
from gsampling.layers.rnconv import rnConv
from gsampling.layers.downsampling import SubgroupDownsample
from gsampling.utils.group_utils import get_group
from einops import rearrange
from .g_cnn_3d import Gcnn3D
from .hybrid import HybridConvGroupResample3D


class Gcnn3DSegmentation(nn.Module):
    """3D Group Equivariant Segmentation model extending Gcnn3D."""
    
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
        3D Group Equivariant Segmentation model.
        
        Uses the existing Gcnn3D encoder and adds a decoder with group upsampling.
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
        """Forward pass through encoder-decoder architecture."""
        # Get encoder features with skip connections
        skip_connections = []
        
        # Run through encoder and collect skip connections
        for i in range(self.num_layers):
            x = self.encoder.conv_layers[i](x)
            x = torch.relu(x)
            
            # Apply spatial downsampling
            if self.spatial_subsampling_factors[i] > 1:
                x = self.encoder.spatial_sampling_layers[i](x)
            
            # Apply group downsampling and store skip connection AFTER downsampling
            if self.encoder.sampling_layers[i] is not None:
                x, _ = self.encoder.sampling_layers[i](x)
            
            # Store skip connection after all downsampling operations
            skip_connections.append(x)
            
            if self.dropout_rate > 0:
                x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        # Decoder path
        for i in range(self.num_layers):
            # Get corresponding skip connection
            skip = skip_connections[self.num_layers - 1 - i]

            # Determine group order from skip connection for proper concat
            current_layer_idx = self.num_layers - 1 - i
            expected_features = self.num_channels[current_layer_idx + 1]
            skip_group_order = skip.shape[1] // expected_features

            # Concatenate at current group order
            x = self._concatenate_skip_connection(x, skip, skip_group_order)

            # Hybrid block: conv at current order then resample group order for next level
            x = self.decoder_blocks[i](x)
            x = torch.relu(x)

            # Spatial upsampling for next level
            if self.spatial_subsampling_factors[self.num_layers - 1 - i] > 1:
                x = self.decoder_spatial_upsampling[i](x)

            if self.dropout_rate > 0:
                x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        # Final output
        x = self.output_conv(x)      # (B, num_classes*G, D, H, W)
        x = self.group_pool(x)       # (B, num_classes, D, H, W)
        
        return x

    def _concatenate_skip_connection(self, x, skip, group_order):
        """
        Concatenate skip connection with explicit group dimension handling.
        
        Your suggested approach:
        1. Factor out group and channel dimensions: (B, C*G, D, H, W) -> (B, C, G, D, H, W)
        2. Concatenate along channel dimension: (B, C1+C2, G, D, H, W)  
        3. Reshape back: (B, (C1+C2)*G, D, H, W)
        
        Args:
            x: Current features (B, C1*G, D, H, W)
            skip: Skip connection (B, C2*G, D, H, W)
            group_order: Number of group elements G
            
        Returns:
            Concatenated features (B, (C1+C2)*G, D, H, W)
        """
        B, CG_x, D, H, W = x.shape
        B_skip, CG_skip, D_skip, H_skip, W_skip = skip.shape
        
        # Factor out group and channel dimensions
        C_x = CG_x // group_order
        C_skip = CG_skip // group_order
        
        print(f"Concatenating: x=({C_x}*{group_order}={CG_x}) + skip=({C_skip}*{group_order}={CG_skip})")
        
        # Reshape to separate group dimension: (B, C, G, D, H, W)
        x_reshaped = x.view(B, C_x, group_order, D, H, W)
        skip_reshaped = skip.view(B, C_skip, group_order, D, H, W)
        
        # Concatenate along channel dimension (axis 1)
        concatenated = torch.cat([x_reshaped, skip_reshaped], dim=1)  # (B, C_x+C_skip, G, D, H, W)
        
        # Reshape back to (B, (C_x+C_skip)*G, D, H, W)
        C_total = C_x + C_skip
        result = concatenated.view(B, C_total * group_order, D, H, W)
        
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
