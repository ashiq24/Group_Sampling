"""
Tests for 3D Group Subsampling Strategies.

This module tests the 3D group subsampling strategies implemented in gsampling/core/subsampling.py:

1. **OctahedralToCycleStrategy**: Subsamples octahedral group (24 elements) to cyclic subgroup (4 elements)
2. **FullOctahedralToCycleStrategy**: Subsamples full octahedral group (48 elements) to cyclic subgroup (4 elements)
3. **FullOctahedralToDihedralStrategy**: Subsamples full octahedral group (48 elements) to dihedral subgroup (8 elements)
4. **FullOctahedralToOctahedralStrategy**: Subsamples full octahedral group (48 elements) to octahedral subgroup (24 elements)

**Mathematical Background:**
- Octahedral group O: 24 rotational symmetries of regular octahedron
- Full octahedral group O_h: 48 symmetries including reflections
- Cyclic subgroup C4: 4-fold rotations around z-axis
- Dihedral subgroup D4: 8 symmetries (4 rotations + 4 reflections) around z-axis
- Subsampling preserves group structure and equivariance properties
"""

import pytest
import numpy as np
# Import the strategies under test
try:
    from gsampling.core.subsampling import (
        OctahedralToCycleStrategy,
        FullOctahedralToCycleStrategy,
        FullOctahedralToDihedralStrategy,
        FullOctahedralToOctahedralStrategy,
        SubsamplingRegistry
    )
    from escnn.group import octa_group, full_octa_group
    ESCNN_AVAILABLE = True
    ESCNN_IMPORT_ERROR = False
except ImportError as e:
    ESCNN_AVAILABLE = False
    ESCNN_IMPORT_ERROR = str(e)


# ============================================================================
# Test Configuration
# ============================================================================

# 3D group test configurations
OCTAHEDRAL_TEST_CONFIGS = [
    # (group_size, expected_subgroup_size, description)
    (24, 4, "Octahedral O → Cyclic C4"),
]

FULL_OCTAHEDRAL_CYCLE_TEST_CONFIGS = [
    # (group_size, expected_subgroup_size, description)
    (48, 4, "Full Octahedral O_h → Cyclic C4"),
]

FULL_OCTAHEDRAL_DIHEDRAL_TEST_CONFIGS = [
    # (group_size, expected_subgroup_size, description)
    (48, 8, "Full Octahedral O_h → Dihedral D4"),
]

FULL_OCTAHEDRAL_OCTAHEDRAL_TEST_CONFIGS = [
    # (group_size, expected_subgroup_size, description)
    (48, 24, "Full Octahedral O_h → Octahedral O"),
]


# ============================================================================
# Test Classes
# ============================================================================

@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class TestOctahedralToCycleStrategy:
    """Test OctahedralToCycleStrategy subsampling from O (24) to C4 (4)."""
    
    def test_strategy_creation(self):
        """Test that strategy can be created successfully."""
        strategy = OctahedralToCycleStrategy()
        assert strategy is not None
        assert hasattr(strategy, 'subsample')
        assert hasattr(strategy, 'validate_parameters')
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid inputs."""
        strategy = OctahedralToCycleStrategy()
        
        # Valid parameters
        assert strategy.validate_parameters(24, 6)  # 24 → 4 elements
        
        # Should not raise any exceptions
        try:
            strategy.validate_parameters(24, 6)
        except Exception as e:
            pytest.fail(f"Validation should pass but raised: {e}")
    
    def test_validate_parameters_invalid_group_size(self):
        """Test parameter validation with invalid group size."""
        strategy = OctahedralToCycleStrategy()
        
        # Invalid group size
        with pytest.raises(ValueError, match="Octahedral group must have 24 elements"):
            strategy.validate_parameters(48, 6)
        
        with pytest.raises(ValueError, match="Octahedral group must have 24 elements"):
            strategy.validate_parameters(12, 6)
    
    def test_subsample_basic_functionality(self):
        """Test basic subsampling functionality."""
        strategy = OctahedralToCycleStrategy()
        
        # Create test nodes
        nodes = list(range(24))
        
        # Perform subsampling
        result = strategy.subsample(nodes, 6)  # 24 → 4 elements
        
        # Check basic properties
        assert isinstance(result, list)
        assert len(result) == 4  # Should return 4 elements
        
        # All indices should be valid
        for idx in result:
            assert 0 <= idx < 24
            assert idx in nodes
    
    def test_subsample_z_axis_preservation(self):
        """Test that subsampled elements preserve the z-axis."""
        strategy = OctahedralToCycleStrategy()
        
        # Get ESCNN octahedral group
        G = octa_group()
        elements = list(G.elements)
        
        # Perform subsampling
        nodes = list(range(24))
        result_indices = strategy.subsample(nodes, 6)
        
        # Check that all subsampled elements preserve z-axis
        for idx in result_indices:
            g = elements[idx]
            rot_mat = g.to('MAT')  # This is fine for octahedral group (24 elements)
            
            # Check z-axis preservation: rot_mat @ [0,0,1] = [0,0,1]
            z_axis = np.array([0, 0, 1])
            transformed_z = rot_mat @ z_axis
            
            assert np.allclose(transformed_z, z_axis, atol=1e-10), \
                f"Element {idx} does not preserve z-axis"
    
    def test_subsample_cyclic_structure(self):
        """Test that subsampled elements form a cyclic group."""
        strategy = OctahedralToCycleStrategy()
        
        # Get ESCNN octahedral group
        G = octa_group()
        elements = list(G.elements)
        
        # Perform subsampling
        nodes = list(range(24))
        result_indices = strategy.subsample(nodes, 6)
        
        # Check that we have exactly 4 elements
        assert len(result_indices) == 4, "Should have exactly 4 elements"
        
        # Check that identity is included (octahedral group test)
        identity_found = False
        for idx in result_indices:
            g = elements[idx]
            if np.allclose(g.to('MAT'), np.eye(3), atol=1e-10):
                identity_found = True
                break
        
        assert identity_found, "Identity element should be in subsampled group"
    
    def test_subsample_consistency(self):
        """Test that subsampling is consistent across multiple calls."""
        strategy = OctahedralToCycleStrategy()
        nodes = list(range(24))
        
        # Perform subsampling multiple times
        result1 = strategy.subsample(nodes, 6)
        result2 = strategy.subsample(nodes, 6)
        result3 = strategy.subsample(nodes, 6)
        
        # All results should be identical
        assert result1 == result2, "Subsampling should be consistent"
        assert result2 == result3, "Subsampling should be consistent"
        assert result1 == result3, "Subsampling should be consistent"


@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class TestFullOctahedralToCycleStrategy:
    """Test FullOctahedralToCycleStrategy subsampling from O_h (48) to C4 (4)."""
    
    def test_strategy_creation(self):
        """Test that strategy can be created successfully."""
        strategy = FullOctahedralToCycleStrategy()
        assert strategy is not None
        assert hasattr(strategy, 'subsample')
        assert hasattr(strategy, 'validate_parameters')
        assert hasattr(strategy, '_z_axis_subgroup_indices')
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid inputs."""
        strategy = FullOctahedralToCycleStrategy()
        
        # Valid parameters
        assert strategy.validate_parameters(48, 12)  # 48 → 4 elements
        
        # Should not raise any exceptions
        try:
            strategy.validate_parameters(48, 12)
        except Exception as e:
            pytest.fail(f"Validation should pass but raised: {e}")
    
    def test_validate_parameters_invalid_group_size(self):
        """Test parameter validation with invalid group size."""
        strategy = FullOctahedralToCycleStrategy()
        
        # Invalid group size
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.validate_parameters(24, 12)
        
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.validate_parameters(96, 12)
    
    def test_subsample_basic_functionality(self):
        """Test basic subsampling functionality."""
        strategy = FullOctahedralToCycleStrategy()
        
        # Create test nodes
        nodes = list(range(48))
        
        # Perform subsampling
        result = strategy.subsample(nodes, 12)  # 48 → 4 elements
        
        # Check basic properties
        assert isinstance(result, list)
        assert len(result) == 4  # Should return 4 elements
        
        # All indices should be valid
        for idx in result:
            assert 0 <= idx < 48
            assert idx in nodes
    
    def test_subsample_z_axis_preservation(self):
        """Test that subsampled elements preserve the z-axis."""
        strategy = FullOctahedralToCycleStrategy()
        
        # Get ESCNN full octahedral group
        G = full_octa_group()
        elements = list(G.elements)
        
        # Perform subsampling
        nodes = list(range(48))
        result_indices = strategy.subsample(nodes, 12)
        
        # Check that all subsampled elements preserve z-axis
        for idx in result_indices:
            g = elements[idx]
            rot_mat = g.to('[int | MAT]')[1]
            
            # Check z-axis preservation: rot_mat @ [0,0,1] = [0,0,1]
            z_axis = np.array([0, 0, 1])
            transformed_z = rot_mat @ z_axis
            
            assert np.allclose(transformed_z, z_axis, atol=1e-10), \
                f"Element {idx} does not preserve z-axis"
    
    def test_precomputed_subgroup_indices(self):
        """Test that subgroup indices are precomputed correctly."""
        strategy = FullOctahedralToCycleStrategy()
        
        # Check that indices are precomputed
        assert hasattr(strategy, '_z_axis_subgroup_indices')
        assert isinstance(strategy._z_axis_subgroup_indices, list)
        assert len(strategy._z_axis_subgroup_indices) == 4
        
        # All indices should be valid
        for idx in strategy._z_axis_subgroup_indices:
            assert 0 <= idx < 48


@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class TestFullOctahedralToDihedralStrategy:
    """Test FullOctahedralToDihedralStrategy subsampling from O_h (48) to D4 (8)."""
    
    def test_strategy_creation(self):
        """Test that strategy can be created successfully."""
        strategy = FullOctahedralToDihedralStrategy()
        assert strategy is not None
        assert hasattr(strategy, 'subsample')
        assert hasattr(strategy, 'validate_parameters')
        assert hasattr(strategy, '_z_axis_subgroup_indices')
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid inputs."""
        strategy = FullOctahedralToDihedralStrategy()
        
        # Valid parameters
        assert strategy.validate_parameters(48, 6)  # 48 → 8 elements
        
        # Should not raise any exceptions
        try:
            strategy.validate_parameters(48, 6)
        except Exception as e:
            pytest.fail(f"Validation should pass but raised: {e}")
    
    def test_validate_parameters_invalid_group_size(self):
        """Test parameter validation with invalid group size."""
        strategy = FullOctahedralToDihedralStrategy()
        
        # Invalid group size
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.validate_parameters(24, 6)
        
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.validate_parameters(96, 6)
    
    def test_subsample_basic_functionality(self):
        """Test basic subsampling functionality."""
        strategy = FullOctahedralToDihedralStrategy()
        
        # Create test nodes
        nodes = list(range(48))
        
        # Perform subsampling
        result = strategy.subsample(nodes, 6)  # 48 → 8 elements
        
        # Check basic properties
        assert isinstance(result, list)
        assert len(result) == 8  # Should return 8 elements
        
        # All indices should be valid
        for idx in result:
            assert 0 <= idx < 48
            assert idx in nodes
    
    def test_subsample_factor_validation(self):
        """Test that subsampling factor must be 6."""
        strategy = FullOctahedralToDihedralStrategy()
        nodes = list(range(48))
        
        # Valid factor
        result = strategy.subsample(nodes, 6)
        print(result)
        assert len(result) == 8
        
    
    def test_subsample_dihedral_structure(self):
        """Test that subsampled elements form a dihedral group."""
        strategy = FullOctahedralToDihedralStrategy()
        
        # Get ESCNN full octahedral group
        G = full_octa_group()
        elements = list(G.elements)
        
        # Perform subsampling
        nodes = list(range(48))
        result_indices = strategy.subsample(nodes, 6)
        
        # Check that we have exactly 8 elements
        assert len(result_indices) == 8, "Should have exactly 8 elements"
        
        # Check that identity is included (full octahedral group test)
        identity_found = False
        for idx in result_indices:
            g = elements[idx]
            
            # Convert to matrix using correct format for full octahedral group
            result = g.to('[int | MAT]')
            mat = result[1]  # Extract the 3x3 matrix from tuple
            if np.allclose(mat, np.eye(3), atol=1e-10):
                    identity_found = True
                    break
        assert identity_found, "Identity element should be in subsampled group"
        
        # Check that we have both rotations and reflections
        rotation_count = 0
        reflection_count = 0
        
        for idx in result_indices:
            g = elements[idx]
            
            # Convert to matrix using correct format for full octahedral group
            result = g.to('[int | MAT]')
            mat = result[1]  # Extract the 3x3 matrix from tuple
            det = 1 if result[0] == 0 else -1
            
            if np.allclose(det, 1.0, atol=1e-10):
                rotation_count += 1
            elif np.allclose(det, -1.0, atol=1e-10):
                reflection_count += 1
            
        
        assert rotation_count > 0, "Should have rotations"
        assert reflection_count > 0, "Should have reflections"
        assert rotation_count + reflection_count == 8, f"Total should be 8 elements, got {rotation_count + reflection_count}"


@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class TestFullOctahedralToOctahedralStrategy:
    """Test FullOctahedralToOctahedralStrategy subsampling from O_h (48) to O (24)."""
    
    def test_strategy_creation(self):
        """Test that strategy can be created successfully."""
        strategy = FullOctahedralToOctahedralStrategy()
        assert strategy is not None
        assert hasattr(strategy, 'subsample')
        assert hasattr(strategy, 'validate_parameters')
        assert hasattr(strategy, '_octahedral_subgroup_indices')
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid inputs."""
        strategy = FullOctahedralToOctahedralStrategy()
        
        # Valid parameters
        assert strategy.validate_parameters(48, 2)  # 48 → 24 elements
        
        # Should not raise any exceptions
        try:
            strategy.validate_parameters(48, 2)
        except Exception as e:
            pytest.fail(f"Validation should pass but raised: {e}")
    
    def test_validate_parameters_invalid_group_size(self):
        """Test parameter validation with invalid group size."""
        strategy = FullOctahedralToOctahedralStrategy()
        
        # Invalid group size
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.validate_parameters(24, 2)
        
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.validate_parameters(96, 2)
    
    def test_subsample_basic_functionality(self):
        """Test basic subsampling functionality."""
        strategy = FullOctahedralToOctahedralStrategy()
        
        # Create test nodes
        nodes = list(range(48))
        
        # Perform subsampling
        result = strategy.subsample(nodes, 2)  # 48 → 24 elements
        
        # Check basic properties
        assert isinstance(result, list)
        assert len(result) == 24  # Should return 24 elements
        
        # All indices should be valid
        for idx in result:
            assert 0 <= idx < 48
            assert idx in nodes
    
    def test_subsample_proper_rotations_only(self):
        """Test that subsampled elements are all proper rotations."""
        strategy = FullOctahedralToOctahedralStrategy()
        
        # Get ESCNN full octahedral group
        G = full_octa_group()
        elements = list(G.elements)
        
        # Perform subsampling
        nodes = list(range(48))
        result_indices = strategy.subsample(nodes, 2)
        
        # Check that we have exactly 24 elements
        assert len(result_indices) == 24, "Should have exactly 24 elements"
        
        # Check that all elements are proper rotations (determinant = +1)
        for idx in result_indices:
            g = elements[idx]

            # Convert to matrix using correct format for full octahedral group
            result = g.to('[int | MAT]')
            mat = result[1]  # Extract the 3x3 matrix from tuple
            det = np.linalg.det(mat)
            
            assert np.allclose(det, 1.0, atol=1e-10), \
                f"Element {idx} is not a proper rotation (det = {det})"
    
    def test_subsample_octahedral_subgroup(self):
        """Test that subsampled elements form the octahedral subgroup."""
        strategy = FullOctahedralToOctahedralStrategy()
        
        # Get ESCNN groups
        G_full = full_octa_group()
        G_octa = octa_group()
        
        # Perform subsampling
        nodes = list(range(48))
        result_indices = strategy.subsample(nodes, 2)
        
        # Check that we have exactly 24 elements
        assert len(result_indices) == 24, "Should have exactly 24 elements"
        
        # Check that identity is included (full octahedral group test)
        identity_found = False
        for idx in result_indices:
            g = G_full.elements[idx]
            
            # Convert to matrix using correct format for full octahedral group
            result = g.to('[int | MAT]')
            mat = result[1]  # Extract the 3x3 matrix from tuple
            if np.allclose(mat, np.eye(3), atol=1e-10):
                identity_found = True
                break
        assert identity_found, "Identity element should be in subsampled group"


@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class Test3DSubsamplingRegistry:
    """Test 3D subsampling strategy registration and retrieval."""
    
    def test_3d_strategies_registered(self):
        """Test that all 3D strategies are properly registered."""
        # Check that strategies are registered
        assert ("octahedral", "cycle") in SubsamplingRegistry.get_supported_transitions()
        assert ("full_octahedral", "cycle") in SubsamplingRegistry.get_supported_transitions()
        assert ("full_octahedral", "dihedral") in SubsamplingRegistry.get_supported_transitions()
        assert ("full_octahedral", "octahedral") in SubsamplingRegistry.get_supported_transitions()
    
    def test_strategy_retrieval(self):
        """Test that strategies can be retrieved from registry."""
        # Retrieve strategies
        octa_cycle = SubsamplingRegistry.get_strategy("octahedral", "cycle")
        full_octa_cycle = SubsamplingRegistry.get_strategy("full_octahedral", "cycle")
        full_octa_dihedral = SubsamplingRegistry.get_strategy("full_octahedral", "dihedral")
        full_octa_octahedral = SubsamplingRegistry.get_strategy("full_octahedral", "octahedral")
        
        # Check types
        assert isinstance(octa_cycle, OctahedralToCycleStrategy)
        assert isinstance(full_octa_cycle, FullOctahedralToCycleStrategy)
        assert isinstance(full_octa_dihedral, FullOctahedralToDihedralStrategy)
        assert isinstance(full_octa_octahedral, FullOctahedralToOctahedralStrategy)
    
    def test_invalid_transition(self):
        """Test that invalid transitions raise appropriate errors."""
        with pytest.raises(NotImplementedError):
            SubsamplingRegistry.get_strategy("octahedral", "dihedral")
        
        with pytest.raises(NotImplementedError):
            SubsamplingRegistry.get_strategy("full_octahedral", "tetrahedral")


@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class Test3DSubsamplingIntegration:
    """Integration tests for 3D subsampling strategies."""
    
    def test_octahedral_to_cycle_integration(self):
        """Test complete octahedral → cycle subsampling workflow."""
        strategy = OctahedralToCycleStrategy()
        
        # Test with actual ESCNN group
        G = octa_group()
        assert len(G.elements) == 24, "ESCNN octahedral group should have 24 elements"
        
        # Perform subsampling
        nodes = list(range(24))
        result = strategy.subsample(nodes, 6)
        
        # Validate result
        assert len(result) == 4
        assert all(0 <= idx < 24 for idx in result)
        
        # Check that result contains identity
        identity_idx = result[0]  # First element should be identity
        identity_element = G.elements[identity_idx]
        assert np.allclose(identity_element.to('MAT'), np.eye(3), atol=1e-10)
    
    def test_full_octahedral_to_octahedral_integration(self):
        """Test complete full octahedral → octahedral subsampling workflow."""
        strategy = FullOctahedralToOctahedralStrategy()
        
        # Test with actual ESCNN group
        G = full_octa_group()
        assert len(G.elements) == 48, "ESCNN full octahedral group should have 48 elements"
        
        # Perform subsampling
        nodes = list(range(48))
        result = strategy.subsample(nodes, 2)
        
        # Validate result
        assert len(result) == 24
        assert all(0 <= idx < 48 for idx in result)
        
        # Check that all elements are proper rotations
        for idx in result:
            element = G.elements[idx]
            result = element.to('[int | MAT]')
            det = 1 if result[0] == 0 else -1
            assert np.allclose(det, 1.0, atol=1e-10)
    
    def test_cross_strategy_consistency(self):
        """Test consistency between different 3D subsampling strategies."""
        # Test that octahedral → cycle and full_octahedral → cycle give consistent results
        octa_strategy = OctahedralToCycleStrategy()
        full_octa_strategy = FullOctahedralToCycleStrategy()
        
        # Get results
        octa_result = octa_strategy.subsample(list(range(24)), 6)
        full_octa_result = full_octa_strategy.subsample(list(range(48)), 12)
        
        # Both should return 4 elements
        assert len(octa_result) == 4
        assert len(full_octa_result) == 4


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================

@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class Test3DSubsamplingPerformance:
    """Test performance characteristics of 3D subsampling strategies."""
    
    def test_strategy_initialization_performance(self):
        """Test that strategy initialization is reasonably fast."""
        import time
        
        # Time strategy creation
        start_time = time.time()
        strategy = FullOctahedralToDihedralStrategy()
        init_time = time.time() - start_time
        
        # Initialization should be fast (< 1 second)
        assert init_time < 1.0, f"Strategy initialization took {init_time:.3f}s, should be < 1s"
    
    def test_subsampling_performance(self):
        """Test that subsampling operations are reasonably fast."""
        import time
        
        strategy = FullOctahedralToOctahedralStrategy()
        nodes = list(range(48))
        
        # Time subsampling operation
        start_time = time.time()
        result = strategy.subsample(nodes, 2)
        subsampling_time = time.time() - start_time
        
        # Subsampling should be very fast (< 0.1 second)
        assert subsampling_time < 0.1, f"Subsampling took {subsampling_time:.3f}s, should be < 0.1s"
        assert len(result) == 24


@pytest.mark.skipif(not ESCNN_AVAILABLE, reason=f"ESCNN not available: {ESCNN_IMPORT_ERROR}")
class Test3DSubsamplingEdgeCases:
    """Test edge cases and error conditions for 3D subsampling strategies."""
            
    def test_malformed_nodes(self):
        """Test behavior with malformed node lists."""
        strategy = FullOctahedralToOctahedralStrategy()
        
        # Test with wrong number of nodes
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.subsample(list(range(24)), 2)
        
        with pytest.raises(ValueError, match="Full octahedral group must have 48 elements"):
            strategy.subsample(list(range(96)), 2)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
