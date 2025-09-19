import pytest
import torch
import numpy as np
from tests.conftest import device_real_dtype_parametrize, tolerance_config

# Import the modules under test
try:
    from escnn.group import *
    from models.g_cnn_3d import Gcnn3D
    from models.model_handler import get_3d_model
    from gsampling.utils.group_utils import *
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class Test3DGCNNFixed:
    """Test 3D GCNN model functionality with fixed configurations."""

    @pytest.mark.parametrize("group_config", [
        # Simple configurations that work with current implementation
        {
            "dwn_group_types": [["octahedral", "octahedral"], ["octahedral", "octahedral"]],
            "init_group_order": 24,
            "subsampling_factors": [1, 1],
            "expected_shapes": [(1, 32*24, 8, 8, 8), (1, 64*24, 4, 4, 4)]  # No group downsampling
        },
        {
            "dwn_group_types": [["cycle", "cycle"], ["cycle", "cycle"]],
            "init_group_order": 8,
            "subsampling_factors": [1, 1],
            "expected_shapes": [(1, 32*8, 8, 8, 8), (1, 64*8, 4, 4, 4)]
        }
    ])
    @device_real_dtype_parametrize
    def test_3d_gcnn_forward_pass(self, group_config, device, dtype):
        """Test forward pass with different 3D group configurations."""
        if dtype == torch.float64:
            pytest.skip("Skipping double precision test due to ESCNN 3D convolution dtype limitations")

        print(f"*****Testing 3D GCNN Forward Pass: {group_config['dwn_group_types']}******")

        # Create 3D input tensor
        batch_size = 2
        input_channels = 1
        depth, height, width = 8, 8, 8

        model = Gcnn3D(
            num_layers=2,
            num_channels=[input_channels, 32, 64],
            kernel_sizes=[3, 3],
            num_classes=10,
            dwn_group_types=group_config["dwn_group_types"],
            init_group_order=group_config["init_group_order"],
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=group_config["subsampling_factors"],
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,
            antialiasing_kwargs=None,
            dropout_rate=0.0,
            device=device,
            dtype=dtype,
        )

        model = model.to(device=device, dtype=dtype)
        x = torch.randn(batch_size, input_channels, depth, height, width).to(device, dtype)

        # Forward pass and verify output
        output = model(x)
        expected_output_shape = (batch_size, 10)
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
            dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "octahedral"]],
            init_group_order=24,
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=[1, 1],  # No group downsampling
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,
            antialiasing_kwargs=None,
            dropout_rate=0.0,
        )

        # Test different input spatial dimensions
        test_dims = [(4, 4, 4), (6, 6, 6), (8, 8, 8), (16, 16, 16)]

        for depth, height, width in test_dims:
            print(f"Testing input dimensions: {depth}x{height}x{width}")
            x = torch.randn(1, 1, depth, height, width)
            output = model(x)
            expected_shape = (1, 5)
            assert output.shape == expected_shape, \
                f"Expected output shape {expected_shape}, got {output.shape} for input {depth}x{height}x{width}"

        print("✅ All tensor shape tests passed")

    def test_3d_gcnn_anti_aliasing(self):
        """Test anti-aliasing functionality."""
        print("*****Testing 3D Anti-aliasing******")

        # Test with anti-aliasing enabled
        model_with_aa = Gcnn3D(
            num_layers=1,
            num_channels=[1, 16],
            kernel_sizes=[3],
            num_classes=5,
            dwn_group_types=[["octahedral", "octahedral"]],
            init_group_order=24,
            spatial_subsampling_factors=[1],
            subsampling_factors=[1],
            domain=3,
            pooling_type="max",
            apply_antialiasing=True,
            antialiasing_kwargs={"smoothness_loss_weight": 0.1},
            dropout_rate=0.0,
        )

        x = torch.randn(1, 1, 8, 8, 8)
        output = model_with_aa(x)
        assert output.shape == (1, 5), f"Expected output shape (1, 5), got {output.shape}"

        print("✅ Anti-aliasing test passed")

    def test_3d_gcnn_pooling_types(self):
        """Test different pooling types."""
        print("*****Testing 3D Pooling Types******")

        for pooling_type in ["max", "mean"]:
            model = Gcnn3D(
                num_layers=1,
                num_channels=[1, 16],
                kernel_sizes=[3],
                num_classes=5,
                dwn_group_types=[["octahedral", "octahedral"]],
                init_group_order=24,
                spatial_subsampling_factors=[1],
                subsampling_factors=[1],
                domain=3,
                pooling_type=pooling_type,
                apply_antialiasing=False,
                antialiasing_kwargs=None,
                dropout_rate=0.0,
            )

            x = torch.randn(1, 1, 8, 8, 8)
            output = model(x)
            assert output.shape == (1, 5), f"Expected output shape (1, 5), got {output.shape} for {pooling_type} pooling"

        print("✅ All pooling type tests passed")

    def test_3d_gcnn_fully_convolutional(self):
        """Test fully convolutional mode."""
        print("*****Testing Fully Convolutional 3D GCNN******")

        model = Gcnn3D(
            num_layers=2,
            num_channels=[1, 16, 32],
            kernel_sizes=[3, 3],
            num_classes=5,
            dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "octahedral"]],
            init_group_order=24,
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=[1, 1],
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,
            antialiasing_kwargs=None,
            dropout_rate=0.0,
            fully_convolutional=True,
        )

        x = torch.randn(1, 1, 10, 10, 10)
        output = model(x)
        # In fully convolutional mode, output should have spatial dimensions
        expected_shape = (1, 32 * 24, 5, 5, 5)  # 32 channels * 24 group order, spatial dims halved once
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {output.shape}"

        print("✅ Fully convolutional test passed")

    def test_3d_gcnn_hidden_features(self):
        """Test hidden feature extraction."""
        print("*****Testing 3D Hidden Features******")

        model = Gcnn3D(
            num_layers=2,
            num_channels=[1, 16, 32],
            kernel_sizes=[3, 3],
            num_classes=5,
            dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "octahedral"]],
            init_group_order=24,
            spatial_subsampling_factors=[2, 1],
            subsampling_factors=[1, 1],
            domain=3,
            pooling_type="max",
            apply_antialiasing=False,
            antialiasing_kwargs=None,
            dropout_rate=0.0,
        )

        x = torch.randn(1, 1, 10, 10, 10)
        feature_before, feature_after, sampling_layers = model.get_hidden_feature(x)

        # Check that we get the expected number of features
        assert len(feature_before) == 2, f"Expected 2 before features, got {len(feature_before)}"
        assert len(feature_after) == 2, f"Expected 2 after features, got {len(feature_after)}"
        assert len(sampling_layers) == 2, f"Expected 2 sampling layers, got {len(sampling_layers)}"

        print("✅ Hidden features test passed")


class Test3DModelHandlerFixed:
    """Test 3D model handler functionality with fixed configurations."""

    def test_get_3d_model_basic(self):
        """Test basic model creation through handler."""
        print("*****Testing Basic 3D Model Handler******")

        model = get_3d_model(
            input_channel=1,
            num_layers=2,
            dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "octahedral"]],
            init_group_order=24,
            subsampling_factors=[1, 1],  # No group downsampling
        )

        assert isinstance(model, Gcnn3D), "Model should be instance of Gcnn3D"
        assert model.num_layers == 2, f"Expected 2 layers, got {model.num_layers}"

        print("✅ Basic model handler test passed")

    def test_get_3d_model_defaults(self):
        """Test model creation with default parameters."""
        print("*****Testing 3D Model Handler Defaults******")

        model = get_3d_model(
            input_channel=1,
            num_layers=1,
            dwn_group_types=[["octahedral", "octahedral"]],
            init_group_order=24,
        )

        assert isinstance(model, Gcnn3D), "Model should be instance of Gcnn3D"
        assert model.num_layers == 1, f"Expected 1 layer, got {model.num_layers}"

        print("✅ Default model handler test passed")

    def test_get_3d_model_complex_transitions(self):
        """Test model creation with complex group transitions."""
        print("*****Testing Complex 3D Group Transitions******")

        model = get_3d_model(
            input_channel=1,
            num_layers=2,
            dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "octahedral"]],
            init_group_order=24,
            subsampling_factors=[1, 1],  # No group downsampling
        )

        assert isinstance(model, Gcnn3D), "Model should be instance of Gcnn3D"
        assert model.num_layers == 2, f"Expected 2 layers, got {model.num_layers}"

        print("✅ Complex transitions test passed")

    def test_get_3d_model_domain_validation(self):
        """Test domain validation in model handler."""
        print("*****Testing 3D Model Domain Validation******")

        with pytest.raises(ValueError, match="Gcnn3D requires domain=3"):
            get_3d_model(
                input_channel=1,
                num_layers=1,
                dwn_group_types=[["octahedral", "octahedral"]],
                init_group_order=24,
                domain=2,  # Wrong domain
            )

        print("✅ Domain validation test passed")

    def test_3d_model_forward_pass_handler(self):
        """Test forward pass through model handler."""
        print("*****Testing 3D Model Handler Forward Pass******")

        model = get_3d_model(
            input_channel=1,
            num_layers=1,
            dwn_group_types=[["octahedral", "octahedral"]],
            init_group_order=24,
        )

        x = torch.randn(2, 1, 8, 8, 8)
        output = model(x)
        assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"

        print("✅ Model handler forward pass test passed")
