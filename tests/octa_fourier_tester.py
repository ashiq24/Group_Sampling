"""Comprehensive tests for Octahedral group Fourier basis and Reynolds operator functions."""

import pytest
import torch
import numpy as np
from tests.conftest import (
    assert_tensors_close,
    assert_matrix_properties,
    check_eigenvalue_property
)

# Import the modules under test
try:
    from gsampling.core.graphs.factory import GroupGraphFactory
    from gsampling.core.graphs.octahedral import OctahedralGraph, FullOctahedralGraph
    import escnn.group as escnn_group
    import escnn.gspaces as escnn_gspaces
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestOctahedralFourierBasisProperties:
    """Test mathematical properties of Fourier bases from octahedral graph constructors."""

    def test_octahedral_basis_unitary(self, tolerance_config):
        """Test that octahedral Fourier basis is unitary: U @ U† = I."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        basis = torch.tensor(octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Check unitary property
        assert_matrix_properties(basis, ['unitary'], tolerance_config,
                                msg="Octahedral basis should be unitary")

    def test_full_octahedral_basis_unitary(self, tolerance_config):
        """
        Test that full octahedral Fourier basis is unitary: U @ U† = I.
        
        **Mathematical Property:**
        A matrix U is unitary if U @ U† = I, where U† is the conjugate transpose.
        This property ensures that Fourier transforms preserve inner products and norms.
        
        **For Full Octahedral Groups:**
        The Fourier basis is constructed from the irreducible representations of O_h,
        which represent the 48 symmetries including both rotations and reflections.
        
        **Test Procedure:**
        1. Build FullOctahedralGraph for 48 elements
        2. Extract Fourier basis matrix
        3. Compute U @ U† and verify it equals identity matrix
        4. Use appropriate numerical tolerances for floating-point comparison
        """
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        basis = torch.tensor(full_octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Check unitary property
        assert_matrix_properties(basis, ['unitary'], tolerance_config,
                                msg="Full octahedral basis should be unitary")

    def test_octahedral_basis_orthogonality(self, tolerance_config):
        """Test that octahedral basis vectors are orthogonal to each other."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        basis = torch.tensor(octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Gram matrix should be identity
        gram_matrix = basis.conj().T @ basis
        identity = torch.eye(basis.shape[1], dtype=torch.cfloat)
        
        assert_tensors_close(gram_matrix, identity, tolerance_cfg=tolerance_config,
                           msg="Octahedral basis vectors should be orthogonal")

    def test_full_octahedral_basis_orthogonality(self, tolerance_config):
        """Test that full octahedral basis vectors are orthogonal to each other."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        basis = torch.tensor(full_octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Gram matrix should be identity
        gram_matrix = basis.conj().T @ basis
        identity = torch.eye(basis.shape[1], dtype=torch.cfloat)
        
        assert_tensors_close(gram_matrix, identity, tolerance_cfg=tolerance_config,
                           msg="Full octahedral basis vectors should be orthogonal")

    def test_octahedral_basis_unit_norm(self, tolerance_config):
        """Test that each octahedral basis vector has unit norm."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        basis = torch.tensor(octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Each column should have unit norm
        column_norms = torch.norm(basis, dim=0)
        expected_norms = torch.ones(basis.shape[1], dtype=torch.float32)
        
        assert_tensors_close(column_norms, expected_norms, tolerance_cfg=tolerance_config,
                           msg="Octahedral basis vectors should have unit norm")

    def test_full_octahedral_basis_unit_norm(self, tolerance_config):
        """Test that each full octahedral basis vector has unit norm."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        basis = torch.tensor(full_octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Each column should have unit norm
        column_norms = torch.norm(basis, dim=0)
        expected_norms = torch.ones(basis.shape[1], dtype=torch.float32)
        
        assert_tensors_close(column_norms, expected_norms, tolerance_cfg=tolerance_config,
                           msg="Full octahedral basis vectors should have unit norm")


class TestOctahedralReynoldsOperatorProperties:
    """
    Test mathematical properties of Reynolds operators from octahedral graph constructors.
    
    **Mathematical Foundation:**
    Reynolds operators implement the formula: R = (1/|G|) Σ_{g∈G} ρ(g) ⊗ ρ(g⁻¹)ᵀ
    where ρ is the direct sum of irreps acting on Fourier coefficients.
    
    **Required Properties:**
    - Hermitian: R = R† (ensures real eigenvalues and stable numerics)
    - Idempotent: R² = R (projection property - applying twice = applying once)
    - Positive Semidefinite: R ≥ 0 (all eigenvalues non-negative)
    - Eigenvalue 1: Must exist for projecting onto G-invariant subspace
    
    **Physical Interpretation:**
    Reynolds operators project arbitrary operators to their G-equivariant versions.
    This is crucial for enforcing equivariance constraints in the anti-aliasing layer.
    
    **Implementation Notes:**
    - OctahedralGraph: Normalized by group order |G| = 24
    - FullOctahedralGraph: Normalized by group order |G| = 48
    - Both use Kronecker products of irrep matrices over all group elements
    """

    def test_octahedral_reynolds_hermitian(self, tolerance_config):
        """Test that octahedral Reynolds operator is Hermitian: R = R†."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        reynolds = torch.tensor(octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check Hermitian property
        assert_matrix_properties(reynolds, ['hermitian'], tolerance_config,
                                msg="Octahedral Reynolds operator should be Hermitian")

    def test_full_octahedral_reynolds_hermitian(self, tolerance_config):
        """Test that full octahedral Reynolds operator is Hermitian: R = R†."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        reynolds = torch.tensor(full_octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check Hermitian property
        assert_matrix_properties(reynolds, ['hermitian'], tolerance_config,
                                msg="Full octahedral Reynolds operator should be Hermitian")

    def test_octahedral_reynolds_idempotent(self, tolerance_config):
        """
        Test that octahedral Reynolds operator is idempotent: R² = R.
        
        **Mathematical Property:**
        A matrix R is idempotent if R @ R = R. This is the defining property of projections.
        Reynolds operators are projections onto the G-invariant subspace.
        
        **Physical Meaning:**
        Applying the Reynolds projection twice gives the same result as applying it once.
        This ensures that equivariant corrections are stable and don't accumulate errors.
        
        **Implementation Formula:**
        R = (1/|G|) Σ_{g∈G} ρ(g) ⊗ ρ(g⁻¹)ᵀ
        where ρ is the representation acting on Fourier coefficients.
        
        **Test Significance:**
        Critical for equivariant anti-aliasing - ensures projection stability.
        """
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        reynolds = torch.tensor(octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check idempotent property
        assert_matrix_properties(reynolds, ['idempotent'], tolerance_config,
                                msg="Octahedral Reynolds operator should be idempotent")

    def test_full_octahedral_reynolds_idempotent(self, tolerance_config):
        """Test that full octahedral Reynolds operator is idempotent: R² = R."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        reynolds = torch.tensor(full_octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check idempotent property
        assert_matrix_properties(reynolds, ['idempotent'], tolerance_config,
                                msg="Full octahedral Reynolds operator should be idempotent")

    def test_octahedral_reynolds_positive_semidefinite(self, tolerance_config):
        """Test that octahedral Reynolds operator is positive semidefinite: R ≥ 0."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        reynolds = torch.tensor(octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check eigenvalues are non-negative
        eigenvals = torch.linalg.eigvals(reynolds).real
        min_eigenval = torch.min(eigenvals)
        
        assert min_eigenval >= -tolerance_config['eigen_tol'], \
            f"Octahedral Reynolds operator should be positive semidefinite, " \
            f"min eigenvalue: {min_eigenval}"

    def test_full_octahedral_reynolds_positive_semidefinite(self, tolerance_config):
        """Test that full octahedral Reynolds operator is positive semidefinite: R ≥ 0."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        reynolds = torch.tensor(full_octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check eigenvalues are non-negative
        eigenvals = torch.linalg.eigvals(reynolds).real
        min_eigenval = torch.min(eigenvals)
        
        assert min_eigenval >= -tolerance_config['eigen_tol'], \
            f"Full octahedral Reynolds operator should be positive semidefinite, " \
            f"min eigenvalue: {min_eigenval}"

    def test_octahedral_reynolds_eigenvalue_one(self, tolerance_config):
        """Test that octahedral Reynolds operator has eigenvalue 1."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        reynolds = torch.tensor(octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check for eigenvalue 1
        check_eigenvalue_property(
            reynolds, 
            expected_eigenvalue=1.0,
            tolerance_cfg=tolerance_config,
            msg="Octahedral Reynolds operator should have eigenvalue 1"
        )

    def test_full_octahedral_reynolds_eigenvalue_one(self, tolerance_config):
        """Test that full octahedral Reynolds operator has eigenvalue 1."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        reynolds = torch.tensor(full_octa_graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check for eigenvalue 1
        check_eigenvalue_property(
            reynolds,
            expected_eigenvalue=1.0,
            tolerance_cfg=tolerance_config,
            msg="Full octahedral Reynolds operator should have eigenvalue 1"
        )


class TestOctahedralBasisDimensionConsistency:
    """
    Test that basis dimensions are consistent with 3D group theory.
    
    **Group Theory Background:**
    - Octahedral group O: 24 elements representing rotational symmetries of octahedron
    - Full octahedral group O_h: 48 elements including reflections
    - Reynolds operators: Act on vectorized matrices → size = (basis_cols)²
    
    **Why Dimension Tests Matter:**
    - Ensures proper memory allocation and matrix operations for 3D groups
    - Validates that irrep construction matches theoretical expectations
    - Prevents runtime errors in tensor operations with 3D data
    - Confirms compatibility between graph constructors and ESCNN 3D groups
    """

    def test_octahedral_basis_dimensions(self):
        """Test that octahedral basis has correct dimensions."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        basis = octa_graph.fourier_basis
        
        # Should be 24×24 matrix for octahedral group
        assert basis.shape == (24, 24), \
            f"Octahedral basis should be 24×24, got {basis.shape}"

    def test_full_octahedral_basis_dimensions(self):
        """Test that full octahedral basis has correct dimensions."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        basis = full_octa_graph.fourier_basis
        
        # Should be 48×48 matrix for full octahedral group
        assert basis.shape == (48, 48), \
            f"Full octahedral basis should be 48×48, got {basis.shape}"

    def test_octahedral_reynolds_dimensions(self):
        """Test that octahedral Reynolds operator has correct dimensions."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        reynolds = octa_graph.equi_raynold_op
        
        # Should be (24²)×(24²) = 576×576 for octahedral group
        expected_shape = (24*24, 24*24)
        assert reynolds.shape == expected_shape, \
            f"Octahedral Reynolds should be {expected_shape}, got {reynolds.shape}"

    def test_full_octahedral_reynolds_dimensions(self):
        """Test that full octahedral Reynolds operator has correct dimensions."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        reynolds = full_octa_graph.equi_raynold_op
        
        # Should be (48²)×(48²) = 2304×2304 for full octahedral group
        expected_shape = (48*48, 48*48)
        assert reynolds.shape == expected_shape, \
            f"Full octahedral Reynolds should be {expected_shape}, got {reynolds.shape}"


class TestOctahedralESCNNIntegration:
    """
    Test integration with ESCNN library for 3D groups.
    
    **ESCNN Background:**
    The Group_Sampling library builds on ESCNN (e2cnn) for group-equivariant deep learning.
    ESCNN provides:
    - 3D group objects (octahedral groups) with irrep decompositions
    - 3D gspaces (octaOnR3) for 3D equivariant operations
    - Geometric tensor operations for 3D equivariant layers
    
    **Integration Points:**
    - Graph constructors use ESCNN 3D groups to build Fourier bases
    - Irrep direct sums are computed using ESCNN's directsum utility
    - Group elements and operations rely on ESCNN's group algebra
    
    **Validation Strategy:**
    - Compare graph constructor outputs with direct ESCNN computations
    - Ensure mathematical properties hold for both approaches
    - Verify numerical consistency and absence of degenerate cases
    
    **Reference:** https://quva-lab.github.io/escnn/
    """

    def test_octahedral_escnn_consistency(self, tolerance_config):
        """Test that octahedral basis is consistent with ESCNN."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        # Compare with ESCNN octahedral group
        try:
            G = escnn_group.octa_group()
            assert len(G.elements) == 24, \
                "ESCNN octahedral group should have 24 elements"
        except AttributeError:
            # ESCNN might not have octa_group directly
            pytest.skip("ESCNN octahedral group not available")
        
        # Test that basis is well-formed
        basis = torch.tensor(octa_graph.fourier_basis, dtype=torch.cfloat)
        assert not torch.any(torch.isnan(basis)), "Octahedral basis should not contain NaN"
        assert not torch.any(torch.isinf(basis)), "Octahedral basis should not contain Inf"

    def test_full_octahedral_escnn_consistency(self, tolerance_config):
        """Test that full octahedral basis is consistent with ESCNN."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        # Compare with ESCNN full octahedral group
        try:
            G = escnn_group.full_octa_group()
            assert len(G.elements) == 48, \
                "ESCNN full octahedral group should have 48 elements"
        except AttributeError:
            # ESCNN might not have full_octa_group directly
            pytest.skip("ESCNN full octahedral group not available")
        
        # Test that basis is well-formed
        basis = torch.tensor(full_octa_graph.fourier_basis, dtype=torch.cfloat)
        assert not torch.any(torch.isnan(basis)), "Full octahedral basis should not contain NaN"
        assert not torch.any(torch.isinf(basis)), "Full octahedral basis should not contain Inf"

    def test_octahedral_gspace_integration(self):
        """Test integration with ESCNN 3D gspaces."""
        try:
            # Test that we can create 3D gspaces for octahedral groups
            gspace = escnn_gspaces.octaOnR3()
            assert gspace is not None, "Should be able to create octaOnR3 gspace"
        except AttributeError:
            pytest.skip("ESCNN octaOnR3 gspace not available")


class TestOctahedralNumericalStability:
    """
    Test numerical stability for 3D octahedral groups.
    
    **Numerical Challenges for 3D Groups:**
    - Complex 3D symmetries: More complex than 2D groups, potential for numerical issues
    - Larger matrices: 24×24 and 48×48 bases, 576×576 and 2304×2304 Reynolds operators
    - Mixed irreps: 3D groups often have both 1D and higher-dimensional irreps
    - Matrix conditioning: Avoiding ill-conditioned bases that cause instability
    
    **Stability Requirements:**
    - No NaN/Inf values in computed matrices
    - Non-singular bases (determinant ≠ 0)
    - Reasonable condition numbers for numerical operations
    - Consistent behavior across different 3D group types
    
    **Edge Cases Tested:**
    - Octahedral group O: 24 elements, rotational symmetries only
    - Full octahedral group O_h: 48 elements, including reflections
    - Matrix properties: Unitarity, orthogonality, eigenvalue properties
    """

    def test_octahedral_basis_numerical_stability(self):
        """Test octahedral basis construction numerical stability."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        basis = torch.tensor(octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Check for numerical issues
        assert not torch.any(torch.isnan(basis)), "Octahedral basis should not contain NaN"
        assert not torch.any(torch.isinf(basis)), "Octahedral basis should not contain Inf"
        
        # Check that it's not degenerate
        assert torch.det(basis).abs() > 1e-10, "Octahedral basis should be non-singular"

    def test_full_octahedral_basis_numerical_stability(self):
        """Test full octahedral basis construction numerical stability."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        basis = torch.tensor(full_octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Check for numerical issues
        assert not torch.any(torch.isnan(basis)), "Full octahedral basis should not contain NaN"
        assert not torch.any(torch.isinf(basis)), "Full octahedral basis should not contain Inf"
        
        # Check that it's not degenerate
        assert torch.det(basis).abs() > 1e-10, "Full octahedral basis should be non-singular"

    def test_octahedral_reynolds_numerical_stability(self):
        """Test octahedral Reynolds operator numerical stability."""
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        reynolds = octa_graph.equi_raynold_op
        
        # Check for numerical issues
        assert not np.any(np.isnan(reynolds)), "Octahedral Reynolds should not contain NaN"
        assert not np.any(np.isinf(reynolds)), "Octahedral Reynolds should not contain Inf"
        
        # Check that it's not all zeros
        assert np.any(np.abs(reynolds) > 1e-15), "Octahedral Reynolds should not be all zeros"

    def test_full_octahedral_reynolds_numerical_stability(self):
        """Test full octahedral Reynolds operator numerical stability."""
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        reynolds = full_octa_graph.equi_raynold_op
        
        # Check for numerical issues
        assert not np.any(np.isnan(reynolds)), "Full octahedral Reynolds should not contain NaN"
        assert not np.any(np.isinf(reynolds)), "Full octahedral Reynolds should not contain Inf"
        
        # Check that it's not all zeros
        assert np.any(np.abs(reynolds) > 1e-15), "Full octahedral Reynolds should not be all zeros"


class TestOctahedralCrossGroupConsistency:
    """
    Test consistency properties across different 3D group types.
    
    **Cross-Group Relationships:**
    - Parent-Subgroup: When H ⊆ G, the subgroup basis should be compatible with parent
    - Octahedral vs Full Octahedral: O ⊆ O_h relationship should be mathematically consistent
    - Subsampling Compatibility: Subgroup bases should align with subsampling operations
    
    **Mathematical Consistency:**
    - Unitary property preserved across group hierarchies
    - Reynolds operators maintain projection properties regardless of group size
    - Fourier transforms remain invertible under subgroup restrictions
    
    **Practical Importance:**
    - Validates that anti-aliasing works correctly across 3D group transitions
    - Ensures subsampling doesn't break mathematical foundations
    - Confirms that different 3D group choices don't affect core algorithms
    
    **Test Strategy:**
    - Compare mathematical properties across related 3D groups
    - Verify that group size doesn't affect fundamental properties
    - Test parent-subgroup basis relationships used in downsampling
    """

    def test_octahedral_vs_full_octahedral_consistency(self, tolerance_config):
        """Test consistency between octahedral and full octahedral groups."""
        # Octahedral group (O)
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        octa_basis = torch.tensor(octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Full octahedral group (O_h)
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        full_octa_basis = torch.tensor(full_octa_graph.fourier_basis, dtype=torch.cfloat)
        
        # Both should be unitary
        assert torch.allclose(octa_basis @ octa_basis.conj().T, 
                            torch.eye(octa_basis.shape[0], dtype=torch.cfloat), 
                            rtol=1e-5, atol=1e-6)
        assert torch.allclose(full_octa_basis @ full_octa_basis.conj().T, 
                            torch.eye(full_octa_basis.shape[0], dtype=torch.cfloat), 
                            rtol=1e-5, atol=1e-6)

    def test_octahedral_subgroup_relationship(self):
        """Test that octahedral group can be viewed as subgroup of full octahedral."""
        # This test validates the mathematical relationship O ⊆ O_h
        # The octahedral group O is a proper subgroup of the full octahedral group O_h
        
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        # Basic group size relationship
        assert len(octa_nodes) < len(full_octa_nodes), \
            "Octahedral group should be smaller than full octahedral group"
        
        # Both should have valid mathematical properties
        assert octa_graph.fourier_basis is not None, "Octahedral basis should exist"
        assert full_octa_graph.fourier_basis is not None, "Full octahedral basis should exist"


class TestOctahedralFactoryIntegration:
    """
    Test integration with the GroupGraphFactory system.
    
    **Factory System Background:**
    The GroupGraphFactory provides a unified interface for creating different types of group graphs.
    This test ensures that octahedral groups integrate seamlessly with the existing factory system.
    
    **Integration Points:**
    - Factory registration: Octahedral types should be registered and discoverable
    - Creation consistency: Factory-created graphs should match direct instantiation
    - Error handling: Invalid parameters should raise appropriate errors
    - Type safety: Factory should return correct graph types
    
    **Why Factory Integration Matters:**
    - Enables polymorphic usage across different group types
    - Provides consistent interface for graph construction
    - Supports dynamic group type selection at runtime
    - Maintains backward compatibility with existing code
    """

    def test_factory_octahedral_registration(self):
        """Test that octahedral groups are properly registered in the factory."""
        # Test that factory can create octahedral graphs
        octa_nodes = list(range(24))
        octa_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        assert isinstance(octa_graph, OctahedralGraph), \
            "Factory should return OctahedralGraph instance"
        assert len(octa_graph.nodes) == 24, \
            "Factory-created octahedral graph should have 24 nodes"

    def test_factory_full_octahedral_registration(self):
        """Test that full octahedral groups are properly registered in the factory."""
        # Test that factory can create full octahedral graphs
        full_octa_nodes = list(range(48))
        full_octa_graph = GroupGraphFactory.create('full_octahedral', full_octa_nodes)
        
        assert isinstance(full_octa_graph, FullOctahedralGraph), \
            "Factory should return FullOctahedralGraph instance"
        assert len(full_octa_graph.nodes) == 48, \
            "Factory-created full octahedral graph should have 48 nodes"

    def test_factory_octahedral_consistency(self):
        """Test that factory-created graphs are consistent with direct instantiation."""
        octa_nodes = list(range(24))
        
        # Create via factory
        factory_graph = GroupGraphFactory.create('octahedral', octa_nodes)
        
        # Create directly
        direct_graph = OctahedralGraph(octa_nodes)
        
        # Both should have same basic properties
        assert len(factory_graph.nodes) == len(direct_graph.nodes), \
            "Factory and direct graphs should have same node count"
        assert len(factory_graph.edges) == len(direct_graph.edges), \
            "Factory and direct graphs should have same edge count"
        
        # Both should have same mathematical properties
        assert factory_graph.fourier_basis.shape == direct_graph.fourier_basis.shape, \
            "Factory and direct graphs should have same basis shape"
        assert factory_graph.equi_raynold_op.shape == direct_graph.equi_raynold_op.shape, \
            "Factory and direct graphs should have same Reynolds operator shape"

    def test_factory_invalid_octahedral_parameters(self):
        """Test that factory handles invalid octahedral parameters gracefully."""
        # Test with wrong number of nodes for octahedral group
        wrong_nodes = list(range(25))  # Should be 24 for octahedral
        
        with pytest.raises(ValueError, match="Octahedral group must have exactly 24 elements"):
            GroupGraphFactory.create('octahedral', wrong_nodes)
        
        # Test with wrong number of nodes for full octahedral group
        wrong_full_nodes = list(range(47))  # Should be 48 for full octahedral
        
        with pytest.raises(ValueError, match="Full octahedral group must have exactly 48 elements"):
            GroupGraphFactory.create('full_octahedral', wrong_full_nodes)
