import pytest
import torch
import numpy as np
from tests.conftest import device_real_dtype_parametrize, tolerance_config

# Import the modules under test
try:
    from escnn.group import *
    from gsampling.models.g_cnn_3d import Gcnn3D
    from gsampling.models.model_handler import get_3d_model
    from gsampling.utils.group_utils import *
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class Test3DGCNN:
    """Test 3D GCNN model functionality."""

    @pytest.mark.parametrize("group_config", [
        # Octahedral group configurations
        {
            "dwn_group_types": [["octahedral", "cycle"], ["cycle", "cycle"]],
            "init_group_order": 24,
            "subsampling_factors": [6, 1],
            "expected_shapes": [(1, 32*24, 8, 8, 8), (1, 64*4, 4, 4, 4)]  # After conv and subsampling
        },
        {
            "dwn_group_types": [["full_octahedral", "dihedral"], ["dihedral", "dihedral"]],
            "init_group_order": 48,
            "subsampling_factors": [6, 1],
            "expected_shapes": [(1, 32*48, 8, 8, 8), (1, 64*8, 4, 4, 4)]
        },

        # Complex group transition: full_octahedral → octahedral → cycle
        {
            "dwn_group_types": [["full_octahedral", "octahedral"], ["octahedral", "cycle"]],
            "init_group_order": 48,
            "subsampling_factors": [2, 6],  # 48/24=2, 24/4=6
            "expected_shapes": [(1, 32*48, 8, 8, 8), (1, 64*4, 4, 4, 4)]
        }
    ])
    @device_real_dtype_parametrize
    def test_3d_gcnn_forward_pass(self, group_config, device, dtype):
        """Test forward pass with different 3D group configurations."""
        # Skip double precision tests due to ESCNN compatibility issues
        if dtype == torch.float64:
            pytest.skip("Skipping double precision test due to ESCNN 3D convolution dtype limitations")

        print(f"*****Testing 3D GCNN Forward Pass: {group_config['dwn_group_types']}******")

        # Test with 3D input tensor
        batch_size = 2
        input_channels = 1
        depth, height, width = 8, 8, 8

        model = Gcnn3D(
            num_layers=2,
            num_channels=[input_channels, 32, 64],  # input + 2 layer outputs
            kernel_sizes=[3, 3],
            num_classes=10,
            dwn_group_types=group_config["dwn_group_types"],
            init_group_order=group_config["init_group_order"],
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=group_config["subsampling_factors"],
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,  # Disable for faster testing
            canonicalize=False,
            antialiasing_kwargs=None,  # Not needed since apply_antialiasing=False
            dropout_rate=0.0,
            device=device,
            dtype=dtype,
        )

        # Move model to correct device and dtype
        model = model.to(device=device, dtype=dtype)

        x = torch.randn(batch_size, input_channels, depth, height, width).to(device, dtype)

        print(f"Input shape: {x.shape}")

        # Forward pass
        output = model(x)

        # Check output shape for classification
        expected_output_shape = (batch_size, 10)  # num_classes=10
        assert output.shape == expected_output_shape, \
            f"Expected output shape {expected_output_shape}, got {output.shape}"

        print(f"✅ Forward pass successful, output shape: {output.shape}")

    def test_3d_gcnn_tensor_shapes(self):
        """Test that 3D tensor shapes are handled correctly throughout the network."""
        print("*****Testing 3D Tensor Shape Handling******")

        model = Gcnn3D(
            num_layers=2,
            num_channels=[1, 16, 32],
            kernel_sizes=[3, 3],
            num_classes=5,
            dwn_group_types=[["octahedral", "cycle"], ["cycle", "cycle"]],
            init_group_order=24,
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=[6, 1],
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,
            canonicalize=False,
            antialiasing_kwargs=None,
            dropout_rate=0.0,
        )

        # Test different input spatial dimensions
        test_dims = [(4, 4, 4), (6, 6, 6), (8, 8, 8), (16, 16, 16)]

        for depth, height, width in test_dims:
            print(f"Testing input dimensions: {depth}x{height}x{width}")

            x = torch.randn(1, 1, depth, height, width)
            output = model(x)

            # Output should always be (batch_size, num_classes)
            assert output.shape == (1, 5), \
                f"Expected (1, 5), got {output.shape} for input {depth}x{height}x{width}"

            print(f"  ✅ Input {depth}x{height}x{width} -> Output {output.shape}")

    def test_3d_gcnn_anti_aliasing(self):
        """Test that anti-aliasing works with 3D groups."""
        print("*****Testing 3D Anti-Aliasing******")

        model = Gcnn3D(
            num_layers=1,
            num_channels=[1, 16],
            kernel_sizes=[3],
            num_classes=3,
            dwn_group_types=[["octahedral", "cycle"]],
            init_group_order=24,
            spatial_subsampling_factors=[1],  # No spatial subsampling
            subsampling_factors=[6],
            domain=3,
            pooling_type="max",
            apply_antialiasing=True,
            canonicalize=False,
            dropout_rate=0.0,
            antialiasing_kwargs={
                "smooth_operator": "adjacency",
                "mode": "analytical",
                "iterations": 30,
                "smoothness_loss_weight": 1.0,
                "threshold": 0.0,
                "equi_constraint": True,
                "equi_correction": False,
            }
        )

        x = torch.randn(1, 1, 6, 6, 6)
        output = model(x)

        # Should complete without errors
        assert output.shape == (1, 3)
        print("✅ Anti-aliasing with 3D groups successful")

    def test_3d_gcnn_pooling_types(self):
        """Test different pooling types for 3D GCNN."""
        print("*****Testing 3D Pooling Types******")

        for pooling_type in ["max", "mean"]:
            print(f"Testing pooling type: {pooling_type}")

            model = Gcnn3D(
                num_layers=1,
                num_channels=[1, 8],
                kernel_sizes=[3],
                num_classes=2,
                dwn_group_types=[["octahedral", "cycle"]],
                init_group_order=24,
                spatial_subsampling_factors=[1],
                subsampling_factors=[6],
                domain=3,
                pooling_type=pooling_type,
                apply_antialiasing=False,
                canonicalize=False,
                antialiasing_kwargs=None,
                dropout_rate=0.0,
            )

            x = torch.randn(2, 1, 4, 4, 4)
            output = model(x)

            assert output.shape == (2, 2)
            print(f"  ✅ {pooling_type} pooling successful")

    def test_3d_gcnn_fully_convolutional(self):
        """Test fully convolutional 3D GCNN mode."""
        print("*****Testing Fully Convolutional 3D GCNN******")

        model = Gcnn3D(
            num_layers=2,
            num_channels=[1, 16, 32],
            kernel_sizes=[3, 3],
            num_classes=5,  # Not used in fully convolutional mode
            dwn_group_types=[["octahedral", "cycle"], ["cycle", "cycle"]],
            init_group_order=24,
            spatial_subsampling_factors=[2, 2],
            subsampling_factors=[6, 1],
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,
            canonicalize=False,
            antialiasing_kwargs=None,
            dropout_rate=0.0,
            fully_convolutional=True,
        )

        # Test with different input sizes
        test_sizes = [(8, 8, 8), (12, 12, 12), (16, 16, 16)]

        for depth, height, width in test_sizes:
            x = torch.randn(1, 1, depth, height, width)

            # Calculate expected output size after 2 layers of 2x subsampling
            expected_depth = depth // 4
            expected_height = height // 4
            expected_width = width // 4

            output = model(x)

            # For fully convolutional, output should preserve spatial dimensions after subsampling
            expected_shape = (1, 32 * 4, expected_depth, expected_height, expected_width)  # 32 channels * 4 group elements
            assert output.shape == expected_shape, \
                f"Expected {expected_shape}, got {output.shape} for input {depth}x{height}x{width}"

            print(f"  ✅ Input {depth}x{height}x{width} -> Output {output.shape}")

    def test_3d_gcnn_hidden_features(self):
        """Test extraction of hidden features from 3D GCNN."""
        print("*****Testing 3D Hidden Features******")

        model = Gcnn3D(
            num_layers=2,
            num_channels=[1, 16, 32],
            kernel_sizes=[3, 3],
            num_classes=5,
            dwn_group_types=[["octahedral", "cycle"], ["cycle", "cycle"]],
            init_group_order=24,
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=[6, 1],
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,
            canonicalize=False,
            antialiasing_kwargs=None,
            dropout_rate=0.0,
        )

        x = torch.randn(1, 1, 8, 8, 8)

        # Get hidden features
        feature_before, feature_after, sampling_layers = model.get_hidden_feature(x)

        # Should have features for each layer
        assert len(feature_before) == 2, f"Expected 2 layers, got {len(feature_before)}"
        assert len(feature_after) == 2, f"Expected 2 layers, got {len(feature_after)}"
        assert len(sampling_layers) == 2, f"Expected 2 sampling layers, got {len(sampling_layers)}"

        # Check shapes
        print(f"Before sampling shapes: {[f.shape for f in feature_before]}")
        print(f"After sampling shapes: {[f.shape for f in feature_after]}")

        print("✅ Hidden features extraction successful")


class Test3DModelHandler:
    """Test 3D model handler functionality."""

    def test_get_3d_model_basic(self):
        """Test basic 3D model creation through model handler."""
        print("*****Testing 3D Model Handler Basic******")

        model = get_3d_model(
            input_channel=1,
            num_channels=[1, 32, 64],
            num_layers=2,
            dwn_group_types=[["octahedral", "cycle"], ["cycle", "cycle"]],
            init_group_order=24,
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=[6, 1],
        )

        # Should create a Gcnn3D instance
        assert isinstance(model, Gcnn3D)
        assert model.domain == 3
        assert model.num_layers == 2
        # The model handler adds input_channel to num_channels, so [1, 32, 64] becomes [1, 1, 32, 64]
        assert model.num_channels == [1, 1, 32, 64]

        print("✅ Basic 3D model creation successful")

    def test_get_3d_model_defaults(self):
        """Test 3D model creation with default parameters."""
        print("*****Testing 3D Model Handler Defaults******")

        model = get_3d_model(
            input_channel=2,
            num_layers=1,
        )

        # Check defaults
        assert model.domain == 3
        assert model.num_layers == 1
        assert model.dwn_group_types == [["octahedral", "octahedral"]]  # No group downsampling by default
        assert model.init_group_order == 24
        assert model.spatial_subsampling_factors == [1]  # No spatial downsampling by default
        assert model.subsampling_factors == [1]  # No group downsampling (same group type)

        print("✅ 3D model with defaults successful")

    def test_get_3d_model_complex_transitions(self):
        """Test 3D model with complex group transitions."""
        print("*****Testing Complex 3D Group Transitions******")

        model = get_3d_model(
            input_channel=1,
            num_layers=3,
            dwn_group_types=[
                ["full_octahedral", "octahedral"],
                ["octahedral", "dihedral"],
                ["dihedral", "cycle"]
            ],
            init_group_order=48,  # Full octahedral
        )

        assert isinstance(model, Gcnn3D)
        assert model.domain == 3
        assert model.num_layers == 3
        assert model.dwn_group_types == [
            ["full_octahedral", "octahedral"],
            ["octahedral", "dihedral"],
            ["dihedral", "cycle"]
        ]

        print("✅ Complex 3D group transitions successful")

    def test_get_3d_model_domain_validation(self):
        """Test that domain validation works for 3D models."""
        print("*****Testing 3D Domain Validation******")

        with pytest.raises(ValueError, match="get_3d_model requires domain=3"):
            get_3d_model(
                input_channel=1,
                domain=2,  # Should fail
            )

        print("✅ 3D domain validation successful")

    def test_3d_model_forward_pass_handler(self):
        """Test end-to-end forward pass through model handler."""
        print("*****Testing 3D Model Handler Forward Pass******")

        model = get_3d_model(
            input_channel=1,
            num_layers=1,
            num_channels=[1, 16],
            fully_convolutional=False,
        )

        # Test with 3D input
        x = torch.randn(2, 1, 8, 8, 8)
        output = model(x)

        # Should be classification output
        assert output.shape == (2, 10)  # Default num_classes=10

        print("✅ 3D model handler forward pass successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
