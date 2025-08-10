# Test Infrastructure for Group_Sampling

This directory contains the test infrastructure for the Group_Sampling project, implementing Phase 1 of the refactoring plan.

## Test Scaffolding (`conftest.py`)

Provides comprehensive test utilities:

### Device/Dtype Parametrization
- **Device support**: Automatic CPU/CUDA detection
- **Dtype support**: float32, float64, cfloat (complex64), cdouble (complex128)
- **Parametrize decorators**: Ready-to-use pytest parametrizers

```python
@device_dtype_parametrize
def test_my_function(device, dtype):
    # Test will run on all device/dtype combinations
    pass
```

### Test Fixtures
- **`random_seed`**: Ensures deterministic tests across PyTorch/NumPy/CUDA
- **`tolerance_config`**: Dtype-specific numerical tolerances
- **`device_manager`**: Device management utilities

### Assertion Utilities
- **`assert_tensors_close`**: Tensor comparison with appropriate tolerances
- **`assert_matrix_properties`**: Check unitary/symmetric/hermitian/idempotent properties
- **`check_eigenvalue_property`**: Verify expected eigenvalues

### Group Theory Test Data
- **`GROUP_TEST_CONFIGS`**: Standard group configurations (cycle, dihedral)
- **`SUBSAMPLING_TEST_CONFIGS`**: Standard subsampling scenarios
- **Parametrized decorators**: `@group_config_parametrize`, `@subsampling_config_parametrize`

## Test Helpers (`helpers.py`)

Provides utilities for component testing:

### Tensor Layout Management
```python
helper = TensorLayoutHelper()
# Convert (B, C*|G|, H, W) ↔ (B, C, |G|, H, W)
x_unflat = helper.unflatten_group_channels(x, group_size=4, num_channels=3)
x_flat = helper.flatten_group_channels(x_unflat, group_size=4)
```

### Mock Graph Builders
```python
# Create test graphs for isolated testing
cycle_graph = MockGraphBuilder.build_cycle_graph(n=4)
dihedral_graph = MockGraphBuilder.build_dihedral_graph(n=4)  # D_4 with 8 elements
```

### Mock Components
- **`MockSamplingMatrix`**: Test sampling operations
- **`MockSmoothnessOperator`**: Test different smoothness operators
- **Test data generators**: Perfect reconstruction data, bandlimited signals

### Equivariance Testing
```python
# Test group actions for equivariance validation
transformed = apply_group_action(x, group_element=2, group_type='cycle', group_size=8)
```

## Usage Examples

### Basic Test Structure
```python
import pytest
from tests.conftest import device_dtype_parametrize, tolerance_config
from tests.helpers import TensorLayoutHelper, MockGraphBuilder

@device_dtype_parametrize
def test_my_layer(device, dtype, tolerance_config):
    # Setup
    layer = MyLayer()
    helper = TensorLayoutHelper()
    
    # Test data
    x = helper.generate_random_signal((2, 12, 8, 8), dtype, device)
    
    # Test
    y = layer(x)
    
    # Assertions
    assert_tensors_close(y, expected, tolerance_cfg=tolerance_config)
```

### Graph Property Testing
```python
@pytest.mark.parametrize("graph_type,order", [('cycle', 4), ('dihedral', 8)])
def test_graph_properties(graph_type, order, tolerance_config):
    if graph_type == 'cycle':
        graph = MockGraphBuilder.build_cycle_graph(order)
    else:
        graph = MockGraphBuilder.build_dihedral_graph(order // 2)
    
    # Test unitary property of Fourier basis
    assert_matrix_properties(
        graph.fourier_basis, 
        ['unitary'], 
        tolerance_config,
        msg="Fourier basis should be unitary"
    )
```

## What's Next

This test infrastructure enables safe implementation of Phase 1 tests:

1. **Step 2**: Graph and Fourier properties tests (`test_graph_factory.py`, etc.)
2. **Step 3**: Anti-aliasing solver tests
3. **Step 4**: Sampling layer tests
4. **Step 5**: Canonicalizer tests
5. **Steps 6-10**: Layer integration and validation tests

Each test module will use these utilities to ensure comprehensive, reliable testing throughout the refactoring process.

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=gsampling tests/

# Run specific device/dtype combinations
pytest -k "cpu and float32" tests/
```

The test infrastructure is designed to be:
- ✅ **Comprehensive**: Covers all major components and properties
- ✅ **Reliable**: Deterministic with appropriate tolerances
- ✅ **Flexible**: Supports all device/dtype combinations
- ✅ **Extensible**: Easy to add new test utilities and configurations

