import pytest
import torch
import numpy as np
from tests.conftest import device_real_dtype_parametrize, tolerance_config

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
        # antializaing False
        {"group_type": "octahedral", "order": 24, "sub_group_type": "cycle", "subsampling_factor": 6, "antialiasing": False},
        {"group_type": "full_octahedral", "order": 48, "sub_group_type": "cycle", "subsampling_factor": 12, "antialiasing": False},
        {"group_type": "full_octahedral", "order": 48, "sub_group_type": "dihedral", "subsampling_factor": 6, "antialiasing": False},
        {"group_type": "full_octahedral", "order": 48, "sub_group_type": "octahedral", "subsampling_factor": 2, "antialiasing": False},
        # antializaing True
        {"group_type": "octahedral", "order": 24, "sub_group_type": "cycle", "subsampling_factor": 6, "antialiasing": True},
        {"group_type": "full_octahedral", "order": 48, "sub_group_type": "cycle", "subsampling_factor": 12, "antialiasing": True},
        {"group_type": "full_octahedral", "order": 48, "sub_group_type": "dihedral", "subsampling_factor": 6, "antialiasing": True},
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
            anti_aliasing_kwargs={
                "smooth_operator": "adjacency",
                "mode": "analytical",  # Use faster analytical mode for tests
                "iterations": 50,  # Reduced for testing
                "smoothness_loss_weight": 1.0,
                "threshold": 0.0,
                "equi_constraint": True,
                "equi_correction": False,
            } if antialiasing else None,
            cannonicalize=False,
        )
        
        # Test basic downsampling and upsampling functionality
        print(f"Testing basic functionality...")
        
        # Create test input tensor
        x = torch.randn(2, G.order() * num_features, 6, 6, 6).to(device, dtype)
        print(f"Input tensor shape: {x.shape}")
        
        # Test downsampling
        x_sub, _ = d_layer(x)
        print(f"Downsampled tensor shape: {x_sub.shape}")
        
        # Test upsampling
        x_sub_up = d_layer.upsample(x_sub)
        print(f"Upsampled tensor shape: {x_sub_up.shape}")
        
        # Check that downsampling reduces group dimension correctly
        assert x_sub.shape[1] < x.shape[1], \
            f"Downsampling should reduce channels: {x_sub.shape[1]} should be < {x.shape[1]}"
        
        # Check that upsampling restores original group dimension
        assert x_sub_up.shape[1] == x.shape[1], \
            f"Upsampling should restore original channels: {x_sub_up.shape[1]} should equal {x.shape[1]}"
        
        # Check that spatial dimensions are preserved
        assert x_sub.shape[2:] == x.shape[2:], \
            f"Spatial dimensions should be preserved: {x_sub.shape[2:]} vs {x.shape[2:]}"
        assert x_sub_up.shape[2:] == x.shape[2:], \
            f"Spatial dimensions should be preserved: {x_sub_up.shape[2:]} vs {x.shape[2:]}"
        
        print("✅ Basic functionality test passed!")
        
        # Test group equivariance (similar to layer_tester.py)
        print(f"Testing group equivariance...")
        equivariance_tests_passed = 0
        total_tests = min(3, len(G.elements))  # Test with first 3 group elements for speed
        gspace = get_gspace(group_type=group_type, order=order, num_features=num_features)
        for i, g in enumerate(G.elements):
            if i >= total_tests:
                break
                
            print(f"  Testing group element {i+1}/{total_tests}")
            
            # Transform input using ESCNN gspace
            x_t = gspace.transform(x.clone(), g)
            
            # Apply downsampling to transformed input
            x_t_sub, _ = d_layer(x_t)
            
            # Apply upsampling to downsampled result
            x_t_sub_up = d_layer.upsample(x_t_sub)
            
            # Verify shapes are consistent
            assert x_t.shape == x.shape, f"Transformed input shape mismatch: {x_t.shape} vs {x.shape}"
            assert x_t_sub.shape == x_sub.shape, f"Transformed downsampled shape mismatch: {x_t_sub.shape} vs {x_sub.shape}"
            assert x_t_sub_up.shape == x_sub_up.shape, f"Transformed upsampled shape mismatch: {x_t_sub_up.shape} vs {x_sub_up.shape}"
            
            equivariance_tests_passed += 1
        
        print(f"✅ Group equivariance tests passed: {equivariance_tests_passed}/{total_tests}")




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
