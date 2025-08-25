# ============================================================================
# Standardized Test Configurations
# ============================================================================
"""
Standardized test configurations to reduce repetition and ensure consistency
across different test files.
"""

from typing import Dict, List, Any

# ============================================================================
# Common Test Parameters
# ============================================================================

# Standard batch sizes for different test types
BATCH_SIZES = {
    "small": 1,
    "medium": 2,
    "large": 4,
    "default": 2
}

# Standard spatial dimensions for different domains
SPATIAL_DIMS = {
    "2d": {
        "small": (8, 8),
        "medium": (16, 16),
        "large": (32, 32),
        "default": (16, 16)
    },
    "3d": {
        "small": (6, 6, 6),
        "medium": (8, 8, 8),
        "large": (16, 16, 16),
        "default": (8, 8, 8)
    }
}

# Standard channel counts for different test types
CHANNEL_COUNTS = {
    "minimal": 1,
    "small": 8,
    "medium": 16,
    "large": 32,
    "default": 16
}

# Standard kernel sizes
KERNEL_SIZES = {
    "small": 3,
    "medium": 5,
    "large": 7,
    "default": 3
}

# Standard number of classes for classification tests
NUM_CLASSES = {
    "binary": 2,
    "small": 5,
    "medium": 10,
    "large": 100,
    "default": 10
}

# ============================================================================
# Model Configuration Templates
# ============================================================================

def get_2d_gcnn_config(
    num_layers: int = 2,
    num_channels: List[int] = None,
    kernel_sizes: List[int] = None,
    num_classes: int = None,
    dwn_group_types: List[List[str]] = None,
    init_group_order: int = 8,
    spatial_subsampling_factors: List[int] = None,
    subsampling_factors: List[int] = None,
    pooling_type: str = "max",
    apply_antialiasing: bool = False,
    canonicalize: bool = False,
    dropout_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Get a standard 2D GCNN configuration.
    
    Args:
        num_layers: Number of layers
        num_channels: List of channel counts
        kernel_sizes: List of kernel sizes
        num_classes: Number of output classes
        dwn_group_types: Group downsampling configuration
        init_group_order: Initial group order
        spatial_subsampling_factors: Spatial subsampling factors
        subsampling_factors: Group subsampling factors
        pooling_type: Type of pooling
        apply_antialiasing: Whether to apply anti-aliasing
        canonicalize: Whether to canonicalize
        dropout_rate: Dropout rate
    
    Returns:
        Dictionary with 2D GCNN configuration
    """
    if num_channels is None:
        num_channels = [CHANNEL_COUNTS["minimal"], CHANNEL_COUNTS["small"], CHANNEL_COUNTS["medium"]]
    
    if kernel_sizes is None:
        kernel_sizes = [KERNEL_SIZES["default"]] * num_layers
    
    if num_classes is None:
        num_classes = NUM_CLASSES["default"]
    
    if dwn_group_types is None:
        dwn_group_types = [["dihedral", "cycle"]]
    
    if spatial_subsampling_factors is None:
        spatial_subsampling_factors = [2] * num_layers
    
    if subsampling_factors is None:
        subsampling_factors = [2] * num_layers
    
    return {
        "num_layers": num_layers,
        "num_channels": num_channels,
        "kernel_sizes": kernel_sizes,
        "num_classes": num_classes,
        "dwn_group_types": dwn_group_types,
        "init_group_order": init_group_order,
        "spatial_subsampling_factors": spatial_subsampling_factors,
        "subsampling_factors": subsampling_factors,
        "domain": 2,
        "pooling_type": pooling_type,
        "apply_antialiasing": apply_antialiasing,
        "canonicalize": canonicalize,
        "dropout_rate": dropout_rate
    }


def get_3d_gcnn_config(
    num_layers: int = 2,
    num_channels: List[int] = None,
    kernel_sizes: List[int] = None,
    num_classes: int = None,
    dwn_group_types: List[List[str]] = None,
    init_group_order: int = 24,
    spatial_subsampling_factors: List[int] = None,
    subsampling_factors: List[int] = None,
    pooling_type: str = "max",
    apply_antialiasing: bool = False,
    canonicalize: bool = False,
    dropout_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Get a standard 3D GCNN configuration.
    
    Args:
        num_layers: Number of layers
        num_channels: List of channel counts
        kernel_sizes: List of kernel sizes
        num_classes: Number of output classes
        dwn_group_types: Group downsampling configuration
        init_group_order: Initial group order
        spatial_subsampling_factors: Spatial subsampling factors
        subsampling_factors: Group subsampling factors
        pooling_type: Type of pooling
        apply_antialiasing: Whether to apply anti-aliasing
        canonicalize: Whether to canonicalize
        dropout_rate: Dropout rate
    
    Returns:
        Dictionary with 3D GCNN configuration
    """
    if num_channels is None:
        num_channels = [CHANNEL_COUNTS["minimal"], CHANNEL_COUNTS["small"], CHANNEL_COUNTS["medium"]]
    
    if kernel_sizes is None:
        kernel_sizes = [KERNEL_SIZES["default"]] * num_layers
    
    if num_classes is None:
        num_classes = NUM_CLASSES["default"]
    
    if dwn_group_types is None:
        dwn_group_types = [["octahedral", "cycle"]]
    
    if spatial_subsampling_factors is None:
        spatial_subsampling_factors = [2] * num_layers
    
    if subsampling_factors is None:
        subsampling_factors = [6] * num_layers
    
    return {
        "num_layers": num_layers,
        "num_channels": num_channels,
        "kernel_sizes": kernel_sizes,
        "num_classes": num_classes,
        "dwn_group_types": dwn_group_types,
        "init_group_order": init_group_order,
        "spatial_subsampling_factors": spatial_subsampling_factors,
        "subsampling_factors": subsampling_factors,
        "domain": 3,
        "pooling_type": pooling_type,
        "apply_antialiasing": apply_antialiasing,
        "canonicalize": canonicalize,
        "dropout_rate": dropout_rate
    }


def get_downsampling_layer_config(
    group_type: str,
    order: int,
    sub_group_type: str,
    subsampling_factor: int,
    num_features: int = None,
    generator: str = "r-s",
    sample_type: str = "sample",
    apply_antialiasing: bool = True,
    cannonicalize: bool = False
) -> Dict[str, Any]:
    """
    Get a standard downsampling layer configuration.
    
    Args:
        group_type: Type of input group
        order: Order of input group
        sub_group_type: Type of output subgroup
        subsampling_factor: Subsampling factor
        num_features: Number of features
        generator: Generator type
        sample_type: Sampling type
        apply_antialiasing: Whether to apply anti-aliasing
        cannonicalize: Whether to canonicalize
    
    Returns:
        Dictionary with downsampling layer configuration
    """
    if num_features is None:
        num_features = CHANNEL_COUNTS["small"]
    
    return {
        "group_type": group_type,
        "order": order,
        "sub_group_type": sub_group_type,
        "subsampling_factor": subsampling_factor,
        "num_features": num_features,
        "generator": generator,
        "sample_type": sample_type,
        "apply_antialiasing": apply_antialiasing,
        "cannonicalize": cannonicalize
    }


# ============================================================================
# Test Data Generators
# ============================================================================

def get_test_tensor_config(
    batch_size: str = "default",
    channels: str = "default",
    spatial_dims: str = "default",
    domain: str = "2d",
    group_order: int = None
) -> Dict[str, Any]:
    """
    Get a standard test tensor configuration.
    
    Args:
        batch_size: Batch size category
        channels: Channel count category
        spatial_dims: Spatial dimensions category
        domain: Domain (2d or 3d)
        group_order: Group order for group-equivariant tensors
    
    Returns:
        Dictionary with test tensor configuration
    """
    return {
        "batch_size": BATCH_SIZES.get(batch_size, BATCH_SIZES["default"]),
        "channels": CHANNEL_COUNTS.get(channels, CHANNEL_COUNTS["default"]),
        "spatial_dims": SPATIAL_DIMS[domain].get(spatial_dims, SPATIAL_DIMS[domain]["default"]),
        "domain": domain,
        "group_order": group_order
    }


# ============================================================================
# Validation Functions
# ============================================================================

def validate_model_config(config: Dict[str, Any], expected_keys: List[str]) -> bool:
    """
    Validate that a model configuration contains all required keys.
    
    Args:
        config: Configuration dictionary
        expected_keys: List of expected keys
    
    Returns:
        True if all keys are present, False otherwise
    """
    return all(key in config for key in expected_keys)


def validate_tensor_shapes(
    input_shape: tuple,
    output_shape: tuple,
    expected_batch_size: int = None,
    expected_spatial_dims: tuple = None
) -> Dict[str, bool]:
    """
    Validate tensor shapes after processing.
    
    Args:
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        expected_batch_size: Expected batch size
        expected_spatial_dims: Expected spatial dimensions
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check batch dimension
    if expected_batch_size is not None:
        results["batch_size"] = output_shape[0] == expected_batch_size
    
    # Check spatial dimensions
    if expected_spatial_dims is not None:
        results["spatial_dims"] = output_shape[2:] == expected_spatial_dims
    
    # Check that output has at least 2 dimensions
    results["min_dims"] = len(output_shape) >= 2
    
    return results
