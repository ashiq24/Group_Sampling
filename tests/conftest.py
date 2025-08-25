"""
Test configuration and fixtures for Group_Sampling test suite.

This module provides:
- Device/dtype parametrization helpers (cpu/cuda-if-available; float32/float64/cfloat/cdouble)
- Random seed fixture for deterministic tests
- Tolerance constants for numerical comparisons
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple, Union


# ============================================================================
# Device and dtype parametrization
# ============================================================================

def get_available_devices() -> List[str]:
    """Get list of available devices for testing."""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


def get_test_dtypes() -> List[torch.dtype]:
    """Get list of dtypes to test."""
    return [
        torch.float32,
        torch.float64,
        torch.cfloat,   # complex64
        torch.cdouble,  # complex128
    ]


def get_real_dtypes() -> List[torch.dtype]:
    """Get list of real dtypes for tests that don't support complex."""
    return [torch.float32, torch.float64]


def get_complex_dtypes() -> List[torch.dtype]:
    """Get list of complex dtypes."""
    return [torch.cfloat, torch.cdouble]


# Pytest parametrize decorators for common combinations
device_parametrize = pytest.mark.parametrize("device", get_available_devices())
dtype_parametrize = pytest.mark.parametrize("dtype", get_test_dtypes())
real_dtype_parametrize = pytest.mark.parametrize("dtype", get_real_dtypes())
complex_dtype_parametrize = pytest.mark.parametrize("dtype", get_complex_dtypes())

# Combined device and dtype parametrization
device_dtype_combinations = [
    (device, dtype) 
    for device in get_available_devices() 
    for dtype in get_test_dtypes()
]
device_dtype_parametrize = pytest.mark.parametrize(
    "device,dtype", device_dtype_combinations
)

# Real-only device/dtype combinations
device_real_dtype_combinations = [
    (device, dtype) 
    for device in get_available_devices() 
    for dtype in get_real_dtypes()
]
device_real_dtype_parametrize = pytest.mark.parametrize(
    "device,dtype", device_real_dtype_combinations
)


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture(scope="function")
def random_seed():
    """Set deterministic random seeds for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    yield seed
    
    # Reset to non-deterministic after test
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@pytest.fixture(scope="session")
def tolerance_config():
    """Tolerance constants for numerical comparisons."""
    return {
        # Relative tolerance
        'rtol': {
            torch.float32: 1e-5,
            torch.float64: 1e-12,
            torch.cfloat: 1e-5,
            torch.cdouble: 1e-12,
        },
        # Absolute tolerance  
        'atol': {
            torch.float32: 1e-6,
            torch.float64: 1e-15,
            torch.cfloat: 1e-6,
            torch.cdouble: 1e-15,
        },
        # Eigenvalue tolerance for projector tests
        'eigen_tol': 1e-4,
        # Reconstruction error tolerance
        'reconstruction_tol': 1e-4,
        # Matrix orthogonality tolerance
        'orthogonality_tol': 1e-6,
    }


@pytest.fixture
def device_manager():
    """Device management utilities for tests."""
    class DeviceManager:
        @staticmethod
        def to_device(tensor_or_tensors, device):
            """Move tensor(s) to specified device."""
            if isinstance(tensor_or_tensors, (list, tuple)):
                return [t.to(device) if hasattr(t, 'to') else t for t in tensor_or_tensors]
            elif hasattr(tensor_or_tensors, 'to'):
                return tensor_or_tensors.to(device)
            else:
                return tensor_or_tensors
        
        @staticmethod
        def get_tolerance(dtype, tolerance_cfg, key='rtol'):
            """Get appropriate tolerance for dtype."""
            return tolerance_cfg[key].get(dtype, tolerance_cfg[key][torch.float32])
        
        @staticmethod
        def skip_if_no_cuda():
            """Skip test if CUDA not available."""
            return pytest.mark.skipif(
                not torch.cuda.is_available(), 
                reason="CUDA not available"
            )
    
    return DeviceManager()


# ============================================================================
# Group theory test parameters
# ============================================================================

# Common group configurations for testing
GROUP_TEST_CONFIGS = [
    # (group_type, order, generator)
    ('cycle', 4, None),
    ('cycle', 8, None),
    ('dihedral', 8, 'r-s'),    # D_4 with 8 elements
    ('dihedral', 12, 'r-s'),   # D_6 with 12 elements
    ('dihedral', 8, 's-sr'),   # Alternative generator
]

# Subsampling test configurations
# (group_type, order, subgroup_type, subsampling_factor, generator)
SUBSAMPLING_TEST_CONFIGS = [
    ('dihedral', 8, 'dihedral', 2, 'r-s'),
    ('dihedral', 12, 'dihedral', 3, 'r-s'),
    ('dihedral', 8, 'cycle', 2, 'r-s'),
    ('cycle', 8, 'cycle', 2, None),
    ('cycle', 12, 'cycle', 3, None),
]

group_config_parametrize = pytest.mark.parametrize(
    "group_type,order,generator", GROUP_TEST_CONFIGS
)

subsampling_config_parametrize = pytest.mark.parametrize(
    "group_type,order,subgroup_type,subsampling_factor,generator", 
    SUBSAMPLING_TEST_CONFIGS
)


# ============================================================================
# Utility functions for tests
# ============================================================================

def assert_tensors_close(
    actual: torch.Tensor, 
    expected: torch.Tensor, 
    rtol: float = None, 
    atol: float = None,
    tolerance_cfg: dict = None,
    msg: str = ""
):
    """Assert two tensors are close with appropriate tolerances."""
    if tolerance_cfg is not None and rtol is None:
        rtol = tolerance_cfg['rtol'].get(actual.dtype, 1e-5)
    if tolerance_cfg is not None and atol is None:
        atol = tolerance_cfg['atol'].get(actual.dtype, 1e-6)
    
    rtol = rtol or 1e-5
    atol = atol or 1e-6
    
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{msg}\nExpected:\n{expected}\nActual:\n{actual}\n"
        f"Max diff: {torch.max(torch.abs(actual - expected))}\n"
        f"Tolerances: rtol={rtol}, atol={atol}"
    )


def assert_matrix_properties(
    matrix: torch.Tensor,
    properties: List[str],
    tolerance_cfg: dict,
    msg: str = ""
):
    """Assert matrix satisfies given properties (unitary, symmetric, etc.)."""
    rtol = tolerance_cfg['rtol'].get(matrix.dtype, 1e-5)
    atol = tolerance_cfg['atol'].get(matrix.dtype, 1e-6)
    
    if 'unitary' in properties:
        # Check if matrix is unitary: A @ A.H ≈ I
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        product = matrix @ matrix.conj().transpose(-2, -1)
        assert_tensors_close(
            product, identity, rtol, atol, 
            msg=f"{msg}: Matrix not unitary"
        )
    
    if 'symmetric' in properties:
        # Check if matrix is symmetric: A ≈ A.T
        assert_tensors_close(
            matrix, matrix.transpose(-2, -1), rtol, atol,
            msg=f"{msg}: Matrix not symmetric"
        )
    
    if 'hermitian' in properties:
        # Check if matrix is Hermitian: A ≈ A.H
        assert_tensors_close(
            matrix, matrix.conj().transpose(-2, -1), rtol, atol,
            msg=f"{msg}: Matrix not Hermitian"
        )
    
    if 'idempotent' in properties:
        # Check if matrix is idempotent: A @ A ≈ A
        product = matrix @ matrix
        assert_tensors_close(
            product, matrix, rtol, atol,
            msg=f"{msg}: Matrix not idempotent"
        )


def check_eigenvalue_property(
    matrix: torch.Tensor,
    expected_eigenvalue: Union[float, complex],
    tolerance_cfg: dict,
    msg: str = ""
):
    """Check if matrix has expected eigenvalue(s)."""
    eigenvals = torch.linalg.eigvals(matrix)
    eigen_tol = tolerance_cfg['eigen_tol']
    
    # Check if any eigenvalue is close to expected
    close_mask = torch.abs(eigenvals - expected_eigenvalue) < eigen_tol
    assert torch.any(close_mask), (
        f"{msg}: Expected eigenvalue {expected_eigenvalue} not found.\n"
        f"Found eigenvalues: {eigenvals}\n"
        f"Tolerance: {eigen_tol}"
    )


# ============================================================================
# Skip conditions
# ============================================================================

def skip_if_no_escnn():
    """Skip test if ESCNN not available."""
    try:
        import escnn  # noqa: F401
        return False
    except ImportError:
        return pytest.mark.skip(reason="ESCNN not available")


def skip_if_no_scipy():
    """Skip test if SciPy not available."""
    try:
        import scipy  # noqa: F401
        return False
    except ImportError:
        return pytest.mark.skip(reason="SciPy not available")


# ============================================================================
# Test data generators
# ============================================================================

def generate_random_signal(
    shape: Tuple[int, ...], 
    dtype: torch.dtype, 
    device: str,
    seed: int = None
) -> torch.Tensor:
    """Generate random signal for testing."""
    if seed is not None:
        torch.manual_seed(seed)
    
    if dtype.is_complex:
        real_part = torch.randn(shape, dtype=torch.float32, device=device)
        imag_part = torch.randn(shape, dtype=torch.float32, device=device)
        signal = torch.complex(real_part, imag_part).to(dtype)
    else:
        signal = torch.randn(shape, dtype=dtype, device=device)
    
    return signal


def generate_batch_signal(
    batch_size: int,
    channels: int, 
    group_size: int,
    spatial_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    seed: int = None
) -> torch.Tensor:
    """Generate batch of group-structured signals for testing."""
    shape = (batch_size, channels * group_size, *spatial_shape)
    return generate_random_signal(shape, dtype, device, seed)
