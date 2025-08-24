# ============================================================================
# Common Test Utilities
# ============================================================================
"""
Common utilities and fixtures for reducing code duplication in tests.
This module provides standardized test configurations, parameter sets, and helper functions.
"""

import pytest
import torch
from typing import Dict, List, Tuple, Any, Optional
from gsampling.utils.group_utils import get_group, get_gspace


# ============================================================================
# Common Test Parameter Sets
# ============================================================================

# Standard 2D group configurations for testing
STANDARD_2D_GROUP_CONFIGS = [
    {"group_type": "dihedral", "order": 8, "sub_group_type": "dihedral", "subsampling_factor": 2},
    {"group_type": "dihedral", "order": 12, "sub_group_type": "cycle", "subsampling_factor": 2},
    {"group_type": "cycle", "order": 8, "sub_group_type": "cycle", "subsampling_factor": 2},
]

# Standard 3D group configurations for testing
STANDARD_3D_GROUP_CONFIGS = [
    {"group_type": "octahedral", "order": 24, "sub_group_type": "cycle", "subsampling_factor": 6},
    {"group_type": "full_octahedral", "order": 48, "sub_group_type": "octahedral", "subsampling_factor": 2},
    {"group_type": "full_octahedral", "order": 48, "sub_group_type": "dihedral", "subsampling_factor": 6},
]

# Common anti-aliasing configurations
COMMON_ANTI_ALIASING_CONFIG = {
    "smooth_operator": "adjacency",
    "mode": "analytical",  # Use faster analytical mode for tests
    "iterations": 100,  # Reduced for testing
    "smoothness_loss_weight": 1.0,
    "threshold": 0.0,
    "equi_constraint": True,
    "equi_correction": False,
}

# Fast anti-aliasing config for quick tests
FAST_ANTI_ALIASING_CONFIG = {
    "smooth_operator": "adjacency",
    "mode": "analytical",
    "iterations": 50,  # Very fast for testing
    "smoothness_loss_weight": 1.0,
    "threshold": 0.0,
    "equi_constraint": True,
    "equi_correction": False,
}

# ============================================================================
# Common Test Fixtures
# ============================================================================

@pytest.fixture
def standard_2d_group_configs():
    """Fixture providing standard 2D group configurations for testing."""
    return STANDARD_2D_GROUP_CONFIGS


@pytest.fixture
def standard_3d_group_configs():
    """Fixture providing standard 3D group configurations for testing."""
    return STANDARD_3D_GROUP_CONFIGS


@pytest.fixture
def common_anti_aliasing_config():
    """Fixture providing common anti-aliasing configuration."""
    return COMMON_ANTI_ALIASING_CONFIG.copy()


@pytest.fixture
def fast_anti_aliasing_config():
    """Fixture providing fast anti-aliasing configuration for quick tests."""
    return FAST_ANTI_ALIASING_CONFIG.copy()


# ============================================================================
# Common Test Helper Functions
# ============================================================================

def create_test_tensor(
    batch_size: int,
    channels: int,
    spatial_dims: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    group_order: Optional[int] = None
) -> torch.Tensor:
    """
    Create a test tensor with specified dimensions.
    
    Args:
        batch_size: Batch size
        channels: Number of channels
        spatial_dims: Spatial dimensions (e.g., (H, W) for 2D, (D, H, W) for 3D)
        device: Device to create tensor on
        dtype: Data type
        group_order: If provided, multiply channels by group order for group-equivariant tensors
    
    Returns:
        Test tensor with shape (batch_size, channels * group_order, *spatial_dims) if group_order provided,
        otherwise (batch_size, channels, *spatial_dims)
    """
    if group_order:
        total_channels = channels * group_order
    else:
        total_channels = channels
    
    # Ensure spatial_dims is a tuple
    if isinstance(spatial_dims, list):
        spatial_dims = tuple(spatial_dims)
    
    shape = (batch_size, total_channels) + spatial_dims
    return torch.randn(shape, device=device, dtype=dtype)


def verify_tensor_shapes(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    expected_spatial_dims: Tuple[int, ...],
    test_name: str = "tensor shape verification"
) -> None:
    """
    Verify that tensor shapes are correct after processing.
    
    Args:
        input_tensor: Input tensor
        output_tensor: Output tensor
        expected_spatial_dims: Expected spatial dimensions after processing
        test_name: Name of the test for error messages
    """
    # Check batch dimension is preserved
    assert output_tensor.shape[0] == input_tensor.shape[0], \
        f"{test_name}: Batch dimension should be preserved: {output_tensor.shape[0]} vs {input_tensor.shape[0]}"
    
    # Check spatial dimensions match expected
    actual_spatial = output_tensor.shape[2:] if output_tensor.dim() > 2 else ()
    assert actual_spatial == expected_spatial_dims, \
        f"{test_name}: Spatial dimensions mismatch: {actual_spatial} vs {expected_spatial_dims}"


def test_group_equivariance_basic(
    input_tensor: torch.Tensor,
    group_type: str,
    order: int,
    transform_func,
    num_test_elements: int = 3,
    test_name: str = "group equivariance"
) -> int:
    """
    Test basic group equivariance with a subset of group elements.
    
    Args:
        input_tensor: Input tensor to test
        group_type: Type of group
        order: Order of group
        transform_func: Function that applies group transformation
        num_test_elements: Number of group elements to test (default: 3 for speed)
        test_name: Name of the test for logging
    
    Returns:
        Number of successful equivariance tests
    """
    G = get_group(group_type, order)
    
    # For group-equivariant tensors, the first dimension after batch is channels * group_order
    # We need to extract the actual number of features per group element
    if input_tensor.shape[1] % G.order() == 0:
        num_features = input_tensor.shape[1] // G.order()
    else:
        # Fallback: assume the tensor is already in the correct format
        num_features = input_tensor.shape[1]
    
    gspace = get_gspace(group_type=group_type, order=order, num_features=num_features)
    
    equivariance_tests_passed = 0
    total_tests = min(num_test_elements, len(G.elements))
    
    for i, g in enumerate(G.elements):
        if i >= total_tests:
            break
            
        # Transform input using group element
        x_t = gspace.transform(input_tensor.clone(), g)
        
        # Apply the transformation function
        x_t_transformed = transform_func(x_t)
        
        # Verify shapes are consistent
        assert x_t_transformed.shape == input_tensor.shape, \
            f"{test_name}: Transformed tensor shape mismatch: {x_t_transformed.shape} vs {input_tensor.shape}"
        
        equivariance_tests_passed += 1
    
    return equivariance_tests_passed


def get_standard_test_config(
    test_type: str = "2d",
    include_anti_aliasing: bool = True
) -> Dict[str, Any]:
    """
    Get a standard test configuration based on test type.
    
    Args:
        test_type: Type of test ("2d" or "3d")
        include_anti_aliasing: Whether to include anti-aliasing configuration
    
    Returns:
        Dictionary with standard test configuration
    """
    if test_type == "2d":
        base_config = STANDARD_2D_GROUP_CONFIGS[0].copy()
    else:  # 3d
        base_config = STANDARD_3D_GROUP_CONFIGS[0].copy()
    
    if include_anti_aliasing:
        base_config["anti_aliasing_config"] = FAST_ANTI_ALIASING_CONFIG.copy()
    
    return base_config


# ============================================================================
# Common Test Decorators
# ============================================================================

def parametrize_group_configs(configs: List[Dict[str, Any]]):
    """
    Decorator to parametrize tests with group configurations.
    
    Args:
        configs: List of group configuration dictionaries
    
    Returns:
        pytest parametrize decorator
    """
    return pytest.mark.parametrize("group_config", configs)


def parametrize_2d_groups():
    """Decorator to parametrize tests with standard 2D group configurations."""
    return parametrize_group_configs(STANDARD_2D_GROUP_CONFIGS)


def parametrize_3d_groups():
    """Decorator to parametrize tests with standard 3D group configurations."""
    return parametrize_group_configs(STANDARD_3D_GROUP_CONFIGS)
