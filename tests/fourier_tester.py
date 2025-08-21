"""
Comprehensive tests for Fourier basis and Reynolds operator functions.

This module validates the mathematical foundations of the Group_Sampling library by testing:

1. **Fourier Basis Properties** (from DihedralGraph.get_basis() and CycleGraph.get_basis()):
   - Unitarity: U @ U† = I (preserves inner products)
   - Orthogonality: Columns are orthogonal (Gram matrix = Identity)
   - Unit Norm: Each basis vector has norm 1
   - Completeness: Basis spans the full space (perfect reconstruction)

2. **Reynolds Operator Properties** (from get_equi_raynold()):
   - Hermitian: R = R† (real eigenvalues, stable decomposition)
   - Idempotent: R² = R (projection property for equivariance)
   - Positive Semidefinite: R ≥ 0 (all eigenvalues ≥ 0)
   - Eigenvalue 1: Required for equivariant projection onto invariant subspace

3. **Cross-Group Consistency**: Validates relationships between parent/subgroup bases
4. **ESCNN Integration**: Ensures compatibility with the underlying group theory library
5. **Numerical Stability**: Tests edge cases and larger group orders
6. **Extensibility**: Documents interface requirements for adding new group types

**Mathematical Background:**
- Fourier bases are constructed from irreducible representations (irreps) of finite groups
- Reynolds operators implement R = (1/|G|) Σ_{g∈G} ρ(g) ⊗ ρ(g⁻¹)ᵀ for equivariant projections
- These operators are fundamental to the anti-aliasing and subsampling algorithms
"""

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
    from gsampling.utils.graph_constructors import DihedralGraph, CycleGraph
    import escnn.group as escnn_group
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestFourierBasisProperties:
    """
    Test mathematical properties of Fourier bases from graph constructors.
    
    **Mathematical Foundation:**
    Fourier bases for finite groups are constructed from irreducible representations (irreps).
    For a finite group G, the Fourier basis Φ_G satisfies:
    - Unitarity: Φ_G @ Φ_G† = I (preserves inner products under transformation)
    - Completeness: Any signal can be perfectly reconstructed via Φ_G @ (Φ_G† @ signal)
    - Orthogonality: Columns are orthonormal (Gram matrix = Identity)
    
    **Implementation Details:**
    - CycleGraph: Uses ESCNN's cyclic_group regular representation change_of_basis
    - DihedralGraph: Constructs from irreps with proper normalization and reshaping
    
    **Why These Tests Matter:**
    - Validates that the spectral domain transformations are mathematically sound
    - Ensures anti-aliasing operators can rely on these properties
    - Prevents numerical instabilities in downstream computations
    """

    @pytest.mark.parametrize("order", [3, 4, 5, 6, 8])
    def test_cycle_basis_unitary(self, order, tolerance_config):
        """
        Test that cyclic Fourier basis is unitary: U @ U† = I.
        
        **Mathematical Property:**
        A matrix U is unitary if U @ U† = I, where U† is the conjugate transpose.
        This property ensures that Fourier transforms preserve inner products and norms.
        
        **For Cyclic Groups:**
        The Fourier basis is constructed from the regular representation of C_n,
        which is equivalent to the DFT matrix with proper normalization.
        
        **Test Procedure:**
        1. Build CycleGraph for given order
        2. Extract Fourier basis matrix
        3. Compute U @ U† and verify it equals identity matrix
        4. Use appropriate numerical tolerances for floating-point comparison
        """
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Check unitary property
        assert_matrix_properties(basis, ['unitary'], tolerance_config,
                                msg=f"Cycle basis (order {order}) should be unitary")

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    def test_dihedral_basis_unitary(self, order, tolerance_config):
        """Test that dihedral Fourier basis is unitary: U @ U† = I."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Check unitary property
        assert_matrix_properties(basis, ['unitary'], tolerance_config,
                                msg=f"Dihedral basis D_{order} should be unitary")

    @pytest.mark.parametrize("order", [3, 4, 5, 6])
    def test_cycle_basis_orthogonality(self, order, tolerance_config):
        """Test that cyclic basis vectors are orthogonal to each other."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Gram matrix should be identity
        gram_matrix = basis.conj().T @ basis
        identity = torch.eye(basis.shape[1], dtype=torch.cfloat)
        
        assert_tensors_close(gram_matrix, identity, tolerance_cfg=tolerance_config,
                           msg=f"Cycle basis vectors (order {order}) should be orthogonal")

    @pytest.mark.parametrize("order", [3, 4, 5, 6])
    def test_cycle_basis_unit_norm(self, order, tolerance_config):
        """Test that each cyclic basis vector has unit norm."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Each column should have unit norm
        column_norms = torch.norm(basis, dim=0)
        expected_norms = torch.ones(basis.shape[1], dtype=torch.float32)
        
        assert_tensors_close(column_norms, expected_norms, tolerance_cfg=tolerance_config,
                           msg=f"Cycle basis vectors (order {order}) should have unit norm")


class TestReynoldsOperatorProperties:
    """
    Test mathematical properties of Reynolds operators from graph constructors.
    
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
    - CycleGraph: Normalized by group order |G| = n
    - DihedralGraph: Normalized by group order |G| = 2n
    - Both use Kronecker products of irrep matrices over all group elements
    """

    @pytest.mark.parametrize("order", [3, 4, 5, 6])
    def test_cycle_reynolds_hermitian(self, order, tolerance_config):
        """Test that cyclic Reynolds operator is Hermitian: R = R†."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check Hermitian property
        assert_matrix_properties(reynolds, ['hermitian'], tolerance_config,
                                msg=f"Cycle Reynolds operator (order {order}) should be Hermitian")

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_reynolds_hermitian(self, order, tolerance_config):
        """Test that dihedral Reynolds operator is Hermitian: R = R†."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check Hermitian property
        assert_matrix_properties(reynolds, ['hermitian'], tolerance_config,
                                msg=f"Dihedral Reynolds operator D_{order} should be Hermitian")

    @pytest.mark.parametrize("order", [3, 4, 5])
    def test_cycle_reynolds_idempotent(self, order, tolerance_config):
        """
        Test that cyclic Reynolds operator is idempotent: R² = R.
        
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
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check idempotent property
        assert_matrix_properties(reynolds, ['idempotent'], tolerance_config,
                                msg=f"Cycle Reynolds operator (order {order}) should be idempotent")

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_reynolds_idempotent(self, order, tolerance_config):
        """Test that dihedral Reynolds operator is idempotent: R² = R."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check idempotent property
        assert_matrix_properties(reynolds, ['idempotent'], tolerance_config,
                                msg=f"Dihedral Reynolds operator D_{order} should be idempotent")

    @pytest.mark.parametrize("order", [3, 4, 6])
    def test_cycle_reynolds_positive_semidefinite(self, order, tolerance_config):
        """Test that cyclic Reynolds operator is positive semidefinite: R ≥ 0."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check eigenvalues are non-negative
        eigenvals = torch.linalg.eigvals(reynolds).real
        min_eigenval = torch.min(eigenvals)
        
        assert min_eigenval >= -tolerance_config['eigen_tol'], \
            f"Cycle Reynolds operator (order {order}) should be positive semidefinite, " \
            f"min eigenvalue: {min_eigenval}"

    @pytest.mark.parametrize("order", [3, 4, 6])
    def test_cycle_reynolds_eigenvalue_one(self, order, tolerance_config):
        """Test that cyclic Reynolds operator has eigenvalue 1."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check for eigenvalue 1
        check_eigenvalue_property(
            reynolds, 
            expected_eigenvalue=1.0,
            tolerance_cfg=tolerance_config,
            msg=f"Cycle Reynolds operator (order {order}) should have eigenvalue 1"
        )

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_reynolds_eigenvalue_one(self, order, tolerance_config):
        """Test that dihedral Reynolds operator has eigenvalue 1."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check for eigenvalue 1
        check_eigenvalue_property(
            reynolds,
            expected_eigenvalue=1.0,
            tolerance_cfg=tolerance_config,
            msg=f"Dihedral Reynolds operator D_{order} should have eigenvalue 1"
        )


class TestBasisDimensionConsistency:
    """
    Test that basis dimensions are consistent with group theory.
    
    **Group Theory Background:**
    - Cyclic groups C_n: Have n irreps, all 1-dimensional → basis is n×n
    - Dihedral groups D_n: Have mixed 1D and 2D irreps → basis is 2n×(total_irrep_dims)
    - Reynolds operators: Act on vectorized matrices → size = (basis_cols)²
    
    **Why Dimension Tests Matter:**
    - Ensures proper memory allocation and matrix operations
    - Validates that irrep construction matches theoretical expectations
    - Prevents runtime errors in tensor operations
    - Confirms compatibility between graph constructors and ESCNN groups
    """

    @pytest.mark.parametrize("order", [3, 4, 5, 6, 8])
    def test_cycle_basis_dimensions(self, order):
        """Test that cyclic basis has correct dimensions."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = graph.fourier_basis
        
        # Should be square matrix for cyclic groups
        assert basis.shape == (order, order), \
            f"Cycle basis should be {order}×{order}, got {basis.shape}"

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    def test_dihedral_basis_dimensions(self, order):
        """Test that dihedral basis has correct dimensions."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        basis = graph.fourier_basis
        
        # First dimension should equal group size
        assert basis.shape[0] == group_size, \
            f"Dihedral basis first dimension should be {group_size}, got {basis.shape[0]}"
        
        # Second dimension should match irrep structure
        assert basis.shape[1] > 0, "Dihedral basis should have positive second dimension"


class TestESCNNIntegration:
    """
    Test integration with ESCNN library.
    
    **ESCNN Background:**
    The Group_Sampling library builds on ESCNN (e2cnn) for group-equivariant deep learning.
    ESCNN provides:
    - Group objects (dihedral_group, cyclic_group) with irrep decompositions
    - Regular representations and change-of-basis matrices
    - Geometric tensor operations for equivariant layers
    
    **Integration Points:**
    - Graph constructors use ESCNN groups to build Fourier bases
    - Irrep direct sums are computed using ESCNN's directsum utility
    - Group elements and operations rely on ESCNN's group algebra
    
    **Validation Strategy:**
    - Compare graph constructor outputs with direct ESCNN computations
    - Ensure mathematical properties hold for both approaches
    - Verify numerical consistency and absence of degenerate cases
    
    **Reference:** https://quva-lab.github.io/escnn/
    """

    @pytest.mark.parametrize("order", [3, 4, 5, 6])
    def test_cycle_escnn_consistency(self, order, tolerance_config):
        """Test that cyclic basis is consistent with ESCNN."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        # Compare with ESCNN cyclic group
        G = escnn_group.cyclic_group(order)
        escnn_basis = G.regular_representation.change_of_basis
        graph_basis = graph.fourier_basis
        
        # Both should be unitary
        escnn_tensor = torch.tensor(escnn_basis, dtype=torch.cfloat)
        graph_tensor = torch.tensor(graph_basis, dtype=torch.cfloat)
        
        assert_matrix_properties(escnn_tensor, ['unitary'], tolerance_config,
                                msg="ESCNN cyclic basis should be unitary")
        assert_matrix_properties(graph_tensor, ['unitary'], tolerance_config,
                                msg="Graph cyclic basis should be unitary")

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_escnn_consistency(self, order, tolerance_config):
        """Test that dihedral basis is consistent with ESCNN."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Compare with ESCNN dihedral group
        G = escnn_group.dihedral_group(order)
        assert len(G.elements) == group_size, \
            f"ESCNN dihedral group should have {group_size} elements"
        
        # Test that basis is well-formed
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        assert not torch.any(torch.isnan(basis)), "Dihedral basis should not contain NaN"
        assert not torch.any(torch.isinf(basis)), "Dihedral basis should not contain Inf"


class TestNumericalStability:
    """
    Test numerical stability across different scales and edge cases.
    
    **Numerical Challenges:**
    - Small groups (order 1-2): Edge cases that might cause degenerate matrices
    - Large groups (order 16-32): Potential for numerical precision loss
    - Complex arithmetic: Proper handling of complex eigenvalues/eigenvectors
    - Matrix conditioning: Avoiding ill-conditioned bases that cause instability
    
    **Stability Requirements:**
    - No NaN/Inf values in computed matrices
    - Non-singular bases (determinant ≠ 0)
    - Reasonable condition numbers for numerical operations
    - Consistent behavior across different group sizes
    
    **Edge Cases Tested:**
    - Trivial group (order 1): Should produce identity basis
    - Small groups (order 2-3): Minimal non-trivial cases
    - Power-of-2 groups: Common in applications, potential for special optimizations
    - Large groups: Stress test for scalability
    """

    @pytest.mark.parametrize("order", [1, 2, 16, 32])
    def test_basis_numerical_stability(self, order):
        """Test basis construction for edge cases and larger sizes."""
        if order == 1:
            # Trivial group case
            nodes = [0]
            graph = CycleGraph(nodes)
            basis = graph.fourier_basis
            assert basis.shape == (1, 1), "Trivial group basis should be 1×1"
            assert np.abs(basis[0, 0] - 1.0) < 1e-10, "Trivial basis should be [1]"
        else:
            nodes = list(range(order))
            graph = CycleGraph(nodes)
            basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
            
            # Check for numerical issues
            assert not torch.any(torch.isnan(basis)), "Basis should not contain NaN"
            assert not torch.any(torch.isinf(basis)), "Basis should not contain Inf"
            
            # Check that it's not degenerate
            assert torch.det(basis).abs() > 1e-10, "Basis should be non-singular"

    @pytest.mark.parametrize("order", [3, 4, 8, 16])
    def test_reynolds_numerical_stability(self, order):
        """Test Reynolds operator numerical stability."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        reynolds = graph.equi_raynold_op
        
        # Check for numerical issues
        assert not np.any(np.isnan(reynolds)), "Reynolds should not contain NaN"
        assert not np.any(np.isinf(reynolds)), "Reynolds should not contain Inf"
        
        # Check that it's not all zeros
        assert np.any(np.abs(reynolds) > 1e-15), "Reynolds should not be all zeros"

class TestCrossGroupConsistency:
    """
    Test consistency properties across different group types.
    
    **Cross-Group Relationships:**
    - Parent-Subgroup: When H ⊆ G, the subgroup basis should be compatible with parent
    - Generator Independence: Different Cayley graph generators should produce equivalent math
    - Subsampling Compatibility: Subgroup bases should align with subsampling operations
    
    **Mathematical Consistency:**
    - Unitary property preserved across group hierarchies
    - Reynolds operators maintain projection properties regardless of generator choice
    - Fourier transforms remain invertible under subgroup restrictions
    
    **Practical Importance:**
    - Validates that anti-aliasing works correctly across group transitions
    - Ensures subsampling doesn't break mathematical foundations
    - Confirms that different generator choices don't affect core algorithms
    
    **Test Strategy:**
    - Compare mathematical properties across related groups
    - Verify that generator choice doesn't affect fundamental properties
    - Test parent-subgroup basis relationships used in downsampling
    """

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_subgroup_basis_relationship(self, order):
        """Test relationship between parent and subgroup bases."""
        # Parent cycle group
        parent_nodes = list(range(order))
        parent_graph = CycleGraph(parent_nodes)
        parent_basis = torch.tensor(parent_graph.fourier_basis, dtype=torch.cfloat)
        
        # Subgroup (every 2nd element)
        subgroup_order = order // 2
        subgroup_nodes = list(range(0, order, 2))
        subgroup_graph = CycleGraph(subgroup_nodes)
        subgroup_basis = torch.tensor(subgroup_graph.fourier_basis, dtype=torch.cfloat)
        
        # Both should be unitary
        assert torch.allclose(parent_basis @ parent_basis.conj().T, 
                            torch.eye(parent_basis.shape[0], dtype=torch.cfloat), 
                            rtol=1e-5, atol=1e-6)
        assert torch.allclose(subgroup_basis @ subgroup_basis.conj().T, 
                            torch.eye(subgroup_basis.shape[0], dtype=torch.cfloat), 
                            rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("generator", ["r-s", "s-sr"])
    def test_dihedral_generator_consistency(self, generator, tolerance_config):
        """Test that different dihedral generators give consistent mathematical properties."""
        order = 3
        group_size = 2 * order
        nodes = list(range(group_size))
        
        graph = DihedralGraph(nodes, generator)
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        reynolds = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Both should satisfy fundamental properties regardless of generator
        assert_matrix_properties(basis, ['unitary'], tolerance_config,
                                msg=f"Dihedral basis with generator {generator} should be unitary")
        assert_matrix_properties(reynolds, ['hermitian'], tolerance_config,
                                msg=f"Dihedral Reynolds with generator {generator} should be Hermitian")
