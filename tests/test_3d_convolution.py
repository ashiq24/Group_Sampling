"""
Test suite for 3D Group Equivariant Convolution layers.

This module tests the rnConv layer with domain=3 for:
- 2D groups acting on 3D data (dihedralOnR3, rot2dOnR3)
- 3D groups acting on 3D data (octahedral, full_octahedral)
- Tensor shape validation and group equivariance
- Forward pass with realistic 3D input tensors
"""

import pytest
import torch
import torch.nn as nn
from gsampling.layers.rnconv import rnConv
from gsampling.utils.group_utils import get_group, get_gspace


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def device_dtype():
    """Provide device and dtype combinations for testing."""
    return [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
    ]


@pytest.fixture
def group_configs():
    """Provide different group configurations for testing."""
    return [
        # 2D groups on 3D data
        {"type": "dihedral", "order": 8, "description": "Dihedral D8 on 3D data"},
        {"type": "cycle", "order": 4, "description": "Cyclic C4 on 3D data"},
        {"type": "cycle", "order": 8, "description": "Cyclic C8 on 3D data"},
        
        # 3D groups on 3D data
        {"type": "octahedral", "order": 24, "description": "Octahedral group O"},
        {"type": "full_octahedral", "order": 48, "description": "Full octahedral group O_h"},
    ]


# ============================================================================
# Test 3D Convolution Layer Creation
# ============================================================================

class Test3DConvolutionLayerCreation:
    """Test creation of 3D convolution layers with different group types."""
    
    @pytest.mark.parametrize("group_config", [
        {"type": "dihedral", "order": 8},
        {"type": "cycle", "order": 4},
        {"type": "octahedral", "order": 24},
    ])
    def test_layer_creation(self, group_config):
        """Test that 3D convolution layers can be created with different group types."""
        group_type = group_config["type"]
        order = group_config["order"]
        
        conv = rnConv(
            in_group_type=group_type,
            in_order=order,
            in_num_features=16,
            in_representation='regular',
            out_group_type=group_type,
            out_num_features=32,
            out_representation='regular',
            domain=3,
            kernel_size=3
        )
        
        assert conv is not None
        assert conv.domain == 3
        assert conv.in_group_type == group_type
        assert conv.out_group_type == group_type
        assert conv.in_order == order
        assert conv.out_order == order
    
    def test_domain_parameter(self):
        """Test that domain parameter correctly sets the convolution type."""
        # 2D convolution
        conv_2d = rnConv(
            in_group_type='cycle',
            in_order=4,
            in_num_features=16,
            out_group_type='cycle',
            out_num_features=32,
            domain=2,
            in_representation='regular',
            out_representation='regular',
            kernel_size=3
        )
        
        # 3D convolution
        conv_3d = rnConv(
            in_group_type='cycle', in_order=4, in_num_features=16,
            out_group_type='cycle', out_num_features=32,
            domain=3,
            in_representation='regular',
            out_representation='regular',
            kernel_size=3
        )
        
        assert conv_2d.domain == 2
        assert conv_3d.domain == 3
        assert conv_2d.get_domain() == 2
        assert conv_3d.get_domain() == 3
    


# ============================================================================
# Test Tensor Shape Handling
# ============================================================================

class Test3DTensorShapeHandling:
    """Test tensor shape validation and handling for 3D convolutions."""
    
    def test_expected_shapes(self):
        """Test that expected input/output shapes are correctly calculated."""
        conv = rnConv(
            in_group_type='octahedral', in_order=24, in_num_features=16,
            in_representation='regular',
            out_group_type='octahedral', out_num_features=32,
            out_representation='regular',
            domain=3
        )
        
        # Check expected shapes
        input_shape = conv.get_expected_input_shape(batch_size=2)
        output_shape = conv.get_expected_output_shape(batch_size=2)
        
        assert input_shape == (2, 384, None, None, None)  # 16 * 24 = 384
        assert output_shape == (2, 768, None, None, None)  # 32 * 24 = 768
    
    def test_2d_groups_on_3d_data_shapes(self):
        """Test shape handling for 2D groups acting on 3D data."""
        conv = rnConv(
            in_group_type='dihedral', in_order=8, in_num_features=16,
            in_representation='regular',
            out_group_type='dihedral', out_num_features=32,
            out_representation='regular',
            domain=3
        )
        
        input_shape = conv.get_expected_input_shape(batch_size=1)
        output_shape = conv.get_expected_output_shape(batch_size=1)
        
        assert input_shape == (1, 128, None, None, None)  # 16 * 8 = 128
        assert output_shape == (1, 256, None, None, None)  # 32 * 8 = 256


# ============================================================================
# Test Forward Pass with 3D Tensors
# ============================================================================

class Test3DForwardPass:
    """Test forward pass functionality with realistic 3D input tensors."""
    
    @pytest.mark.parametrize("group_config", [
        {"type": "dihedral", "order": 8, "channels": 16, "expected_in": 128, "expected_out": 256},
        {"type": "cycle", "order": 4, "channels": 16, "expected_in": 64, "expected_out": 128},
        {"type": "octahedral", "order": 24, "channels": 8, "expected_in": 192, "expected_out": 384},
    ])
    def test_forward_pass_3d_tensors(self, group_config):
        """Test forward pass with 3D input tensors."""
        group_type = group_config["type"]
        order = group_config["order"]
        in_channels = group_config["channels"]
        expected_in = group_config["expected_in"]
        expected_out = group_config["expected_out"]
        
        conv = rnConv(
            in_group_type=group_type,
            in_order=order,
            in_num_features=in_channels,
            in_representation='regular',
            out_group_type=group_type,
            out_num_features=in_channels * 2,  # Double the output channels
            out_representation='regular',
            domain=3,
            kernel_size=3
        )
        
        # Create 3D input tensor (B, C*|G|, H, W, D)
        batch_size = 2
        height, width, depth = 8, 8, 8
        x = torch.randn(batch_size, expected_in, height, width, depth)
        
        # Forward pass
        output = conv(x)
        
        # Check output shape
        assert output.shape == (batch_size, expected_out, height, width, depth)
        assert output.dtype == x.dtype
        assert output.device == x.device
    
    def test_invalid_tensor_dimensions(self):
        """Test that invalid tensor dimensions raise appropriate errors."""
        conv = rnConv(
            in_group_type='cycle', in_order=4, in_num_features=16,
            out_group_type='cycle', out_num_features=32,
            domain=3,
            in_representation='regular',
            out_representation='regular',
            kernel_size=3
        )
        
        # Test with 4D tensor (should fail)
        x_4d = torch.randn(2, 64, 8, 8)  # Missing depth dimension
        with pytest.raises(ValueError, match="Expected 5D tensor for 3D convolution"):
            conv(x_4d)
        
        # Test with 6D tensor (should fail)
        x_6d = torch.randn(2, 16, 4, 8, 8, 8)  # Extra group dimension
        with pytest.raises(ValueError, match="Expected 5D tensor for 3D convolution"):
            conv(x_6d)



# ============================================================================
# Test Integration with Group Utils
# ============================================================================

class Test3DIntegrationWithGroupUtils:
    """Test integration between 3D convolution and group utilities."""
    
    def test_gspace_consistency(self):
        """Test that gspace types are consistent between group_utils and rnConv."""
        # Test 2D group on 3D data
        conv_dihedral = rnConv(
            in_group_type='dihedral', in_order=8, in_num_features=16,
            in_representation='regular',
            out_group_type='dihedral', out_num_features=32,
            out_representation='regular',
            domain=3
        )
        
        # Get gspace directly from group_utils
        gspace_dihedral = get_gspace(
            group_type='dihedral', order=8, num_features=16, domain=3
        )
        
        # Check that they use the same gspace type
        assert str(conv_dihedral.gspace_in.gspace) == str(gspace_dihedral.gspace)
    
    def test_3d_group_gspace_consistency(self):
        """Test that 3D group gspaces are consistent."""
        conv_octa = rnConv(
            in_group_type='octahedral', in_order=24, in_num_features=16,
            in_representation='regular',
            out_group_type='octahedral', out_num_features=32,
            out_representation='regular',
            domain=3
        )
        
        gspace_octa = get_gspace(
            group_type='octahedral', order=24, num_features=16
        )
        
        assert str(conv_octa.gspace_in.gspace) == str(gspace_octa.gspace)


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class Test3DEdgeCases:
    """Test edge cases and error handling for 3D convolutions."""
    
    def test_small_kernel_size(self):
        """Test convolution with small kernel size."""
        conv = rnConv(
            in_group_type='cycle', in_order=4, in_num_features=8,
            in_representation='regular',
            out_group_type='cycle', out_num_features=16,
            out_representation='regular',
            domain=3, kernel_size=1
        )
        
        x = torch.randn(1, 32, 4, 4, 4)
        output = conv(x)
        
        assert output.shape == (1, 64, 4, 4, 4)
    
    def test_large_kernel_size(self):
        """Test convolution with large kernel size."""
        conv = rnConv(
            in_group_type='cycle', in_order=4, in_num_features=8,
            in_representation='regular',
            out_group_type='cycle', out_num_features=16,
            out_representation='regular',
            domain=3, kernel_size=5
        )
        
        x = torch.randn(1, 32, 8, 8, 8)
        output = conv(x)
        
        assert output.shape == (1, 64, 8, 8, 8)
    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
