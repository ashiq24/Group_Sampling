import pytest
import torch
import numpy as np
from tests.conftest import device_real_dtype_parametrize, tolerance_config
from tests.common_test_utils import (
    STANDARD_3D_GROUP_CONFIGS,
    FAST_ANTI_ALIASING_CONFIG,
    create_test_tensor,
    verify_tensor_shapes,
    test_group_equivariance_basic
)

# Import the modules under test
try:
    from escnn.group import *
    from escnn import gspaces
    from gsampling.layers.downsampling import SubgroupDownsample
    from gsampling.utils.group_utils import *
    from gsampling.core.subsampling import SubsamplingRegistry
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class Test3DDownsamplingLayer:
    """Test 3D group downsampling layer functionality."""

    @pytest.mark.parametrize("group_config", [
        # Standard 3D configurations with antialiasing False
        {"group_type": "octahedral", "order": 24, "sub_group_type": "cycle", "subsampling_factor": 6, "antialiasing": False},
        {"group_type": "full_octahedral", "order": 48, "sub_group_type": "octahedral", "subsampling_factor": 2, "antialiasing": False},
        # Standard 3D configurations with antialiasing True
        {"group_type": "octahedral", "order": 24, "sub_group_type": "cycle", "subsampling_factor": 6, "antialiasing": True},
        {"group_type": "full_octahedral", "order": 48, "sub_group_type": "octahedral", "subsampling_factor": 2, "antialiasing": True},
    ])
    @device_real_dtype_parametrize
    def test_3d_downsampling_layer_functionality(
        self, group_config, device, dtype
    ):
        """Test 3D downsampling layer with different group configurations."""
        group_type = group_config["group_type"]
        order = group_config["order"]
        sub_group_type = group_config["sub_group_type"]
        subsampling_factor = group_config["subsampling_factor"]
        antialiasing = group_config["antialiasing"]
        
        print(f"*****Testing {group_type}->{sub_group_type} 3D Downsampling Layer******")
        
        G = get_group(group_type, order)
        num_features = 8
        d_layer = SubgroupDownsample(
            group_type=group_type,
            order=order,
            sub_group_type=sub_group_type,
            subsampling_factor=subsampling_factor,
            num_features=num_features,
            generator="r-s",  # Will be ignored for 3D groups
            device=device,
            dtype=dtype,
            sample_type="sample",
            apply_antialiasing=antialiasing,
            anti_aliasing_kwargs=FAST_ANTI_ALIASING_CONFIG if antialiasing else None,
            cannonicalize=False,
        )
        
        # Test basic functionality
        x = create_test_tensor(
            batch_size=2,
            channels=num_features,
            spatial_dims=(6, 6, 6),
            device=device,
            dtype=dtype,
            group_order=G.order()
        )
        
        # Test downsampling and upsampling
        x_sub, _ = d_layer(x)
        x_sub_up = d_layer.upsample(x_sub)
        
        # Verify tensor shapes
        assert x_sub.shape[1] < x.shape[1], \
            f"Downsampling should reduce channels: {x_sub.shape[1]} should be < {x.shape[1]}"
        assert x_sub_up.shape[1] == x.shape[1], \
            f"Upsampling should restore original channels: {x_sub_up.shape[1]} should equal {x.shape[1]}"
        
        verify_tensor_shapes(x, x_sub, x.shape[2:], "Downsampling spatial preservation")
        verify_tensor_shapes(x, x_sub_up, x.shape[2:], "Upsampling spatial preservation")
        
        print("✅ Basic functionality test passed!")
        
        # Test group equivariance
        
        def test_transform(x_input):
            """Test function for group equivariance testing."""
            x_sub, _ = d_layer(x_input)
            x_sub_up = d_layer.upsample(x_sub)
            return x_sub_up
        
        equivariance_tests_passed = test_group_equivariance_basic(
            input_tensor=x,
            group_type=group_type,
            order=order,
            transform_func=test_transform,
            num_test_elements=3,
            test_name="3D downsampling layer equivariance"
        )
        
        print(f"✅ Group equivariance tests passed: {equivariance_tests_passed}/3")




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
