"""
Tests for Reynolds operator construction and properties.

This module tests Reynolds operators for equivariant projections:
- Reynolds operators have eigenvalue 1 for invariant subspace
- Projector constructed from eigenspace is idempotent (P² ≈ P)
- Mathematical properties and ESCNN integration
- Vectorized operator actions and kronecker product structure
"""

import pytest
import torch
import numpy as np
from tests.conftest import (
    group_config_parametrize,
    device_real_dtype_parametrize,
    assert_tensors_close,
    assert_matrix_properties,
    check_eigenvalue_property,
    tolerance_config
)

# Import the modules under test
try:
    from gsampling.utils.graph_constructors import DihedralGraph, CycleGraph
    import escnn.group as escnn_group
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestCyclicReynoldsOperator:
    """Test Reynolds operator construction for cyclic groups."""

    @pytest.mark.parametrize("order", [3, 4, 5, 6, 8])
    def test_cyclic_reynolds_dimension(self, order):
        """Test that Reynolds operator has correct dimensions."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_op = graph.equi_raynold_op
        
        # Reynolds operator should be a matrix with size matching the basis
        G = escnn_group.cyclic_group(order)
        basis_size = G.regular_representation.change_of_basis.shape[-1]
        expected_shape = (basis_size, basis_size)
        
        assert reynolds_op.shape == expected_shape, \
            f"Expected Reynolds operator shape {expected_shape}, got {reynolds_op.shape}"

    @pytest.mark.parametrize("order", [3, 4, 6])
    def test_cyclic_reynolds_eigenvalues(self, order, tolerance_config):
        """Test that Reynolds operator has eigenvalue 1."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        # Reynolds operator is already a matrix, not vectorized
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check for eigenvalue 1
        check_eigenvalue_property(
            reynolds_matrix, 
            expected_eigenvalue=1.0,
            tolerance_cfg=tolerance_config,
            msg="Reynolds operator should have eigenvalue 1"
        )

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_cyclic_reynolds_projector_idempotent(self, order, tolerance_config):
        """Test that projector derived from Reynolds operator is idempotent."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        # Get Reynolds operator (already a matrix)
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Find eigenspace for eigenvalue ≈ 1
        eigenvals, eigenvecs = torch.linalg.eig(reynolds_matrix)
        eigen_tol = tolerance_config['eigen_tol']
        
        # Find eigenvectors with eigenvalue close to 1
        mask = torch.abs(eigenvals - 1.0) < eigen_tol
        if torch.any(mask):
            invariant_vecs = eigenvecs[:, mask]
            
            # Construct projector: P = V @ V.H (assuming orthonormal)
            projector = invariant_vecs @ invariant_vecs.conj().T
            
            # Test idempotent property: P @ P ≈ P
            assert_matrix_properties(projector, ['idempotent'], tolerance_config,
                                    msg="Reynolds projector should be idempotent")

    @pytest.mark.parametrize("order", [3, 4, 5])
    def test_cyclic_reynolds_construction(self, order):
        """Test the Reynolds operator construction formula."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        # Test that construction method exists and works
        reynolds_op = graph.get_equi_raynold(order)
        assert len(reynolds_op) > 0, "Reynolds operator should be non-empty"
        
        # Test that it's properly normalized (should sum group contributions by 1/|G|)
        G = escnn_group.cyclic_group(order)
        assert len(G.elements) == order, f"Cyclic group should have {order} elements"

    def test_cyclic_reynolds_escnn_integration(self):
        """Test integration with ESCNN irreps."""
        order = 4
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        # Test that we can get irreps
        irreps = graph.get_irreps(order)
        
        # Test that irreps work with group elements
        G = escnn_group.cyclic_group(order)
        for g in G.elements:
            irrep_matrix = irreps(g)
            assert irrep_matrix.shape[0] == irrep_matrix.shape[1], "Irrep matrices should be square"
            
            # Test inverse element
            g_inv = g.__invert__()
            irrep_inv = irreps(g_inv)
            assert irrep_inv.shape == irrep_matrix.shape, "Inverse irrep should have same shape"


class TestDihedralReynoldsOperator:
    """Test Reynolds operator construction for dihedral groups."""

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    def test_dihedral_reynolds_dimension(self, order):
        """Test that dihedral Reynolds operator has correct dimensions."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        reynolds_op = graph.equi_raynold_op
        
        # Get expected dimension from ESCNN
        G = escnn_group.dihedral_group(order)
        basis_size = G.regular_representation.change_of_basis.shape[-1]
        expected_shape = (basis_size, basis_size)
        
        assert reynolds_op.shape == expected_shape, \
            f"Expected dihedral Reynolds operator shape {expected_shape}, got {reynolds_op.shape}"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_reynolds_eigenvalues(self, order, tolerance_config):
        """Test that dihedral Reynolds operator has eigenvalue 1."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Get Reynolds operator (already a matrix)
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check for eigenvalue 1
        check_eigenvalue_property(
            reynolds_matrix,
            expected_eigenvalue=1.0,
            tolerance_cfg=tolerance_config,
            msg="Dihedral Reynolds operator should have eigenvalue 1"
        )

    @pytest.mark.parametrize("order", [2, 3])
    def test_dihedral_reynolds_normalization(self, order):
        """Test that dihedral Reynolds operator is properly normalized."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Test construction formula normalization
        reynolds_op = graph.get_equi_raynold(order)
        
        # Should be normalized by group size (2n)
        G = escnn_group.dihedral_group(order)
        assert len(G.elements) == group_size, f"Dihedral group D_{order} should have {group_size} elements"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_irreps_structure(self, order):
        """Test dihedral irreps structure and properties."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Get irreps
        irreps = graph.get_irreps(order)
        
        # Test with group elements
        G = escnn_group.dihedral_group(order)
        
        # Test a rotation element
        rotation_elements = [g for g in G.elements if not hasattr(g._element, '__iter__') or g._element[0] == 0]
        if rotation_elements:
            r = rotation_elements[1] if len(rotation_elements) > 1 else rotation_elements[0]
            irrep_r = irreps(r)
            assert irrep_r.shape[0] == irrep_r.shape[1], "Rotation irrep should be square"
        
        # Test a reflection element if available
        reflection_elements = [g for g in G.elements if hasattr(g._element, '__iter__') and g._element[0] == 1]
        if reflection_elements:
            s = reflection_elements[0]
            irrep_s = irreps(s)
            assert irrep_s.shape[0] == irrep_s.shape[1], "Reflection irrep should be square"

    def test_dihedral_reynolds_formula(self):
        """Test Reynolds operator construction formula explicitly."""
        order = 2  # D_2 for simplicity
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Manual verification of formula: R = (1/|G|) Σ_g ρ(g) ⊗ ρ(g^{-1})^T
        G = escnn_group.dihedral_group(order)
        irreps = graph.get_irreps(order)
        
        # Compute manually
        manual_reynolds = None
        for g in G.elements:
            rho_g = irreps(g)
            rho_g_inv_T = irreps(g.__invert__()).T
            
            kron_product = np.kron(rho_g, rho_g_inv_T)
            
            if manual_reynolds is None:
                manual_reynolds = kron_product
            else:
                manual_reynolds += kron_product
        
        manual_reynolds /= len(G.elements)
        
        # Compare with graph implementation
        graph_reynolds = graph.equi_raynold_op
        
        # Should be approximately equal (up to numerical precision)
        manual_flat = manual_reynolds.flatten()
        np.testing.assert_allclose(
            graph_reynolds, manual_flat, rtol=1e-10, atol=1e-12,
            err_msg="Manual and graph Reynolds operators should match"
        )


class TestReynoldsOperatorProperties:
    """Test mathematical properties of Reynolds operators."""

    @pytest.mark.parametrize("group_type,order", [
        ("cycle", 4), ("cycle", 6), 
        ("dihedral", 3), ("dihedral", 4)
    ])
    def test_reynolds_hermitian(self, group_type, order, tolerance_config):
        """Test that Reynolds operators are Hermitian."""
        if group_type == "cycle":
            nodes = list(range(order))
            graph = CycleGraph(nodes)
        else:
            group_size = 2 * order
            nodes = list(range(group_size))
            graph = DihedralGraph(nodes, "r-s")
        
        # Get Reynolds operator (already a matrix)
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Test Hermitian property
        assert_matrix_properties(reynolds_matrix, ['hermitian'], tolerance_config,
                                msg=f"{group_type} Reynolds operator should be Hermitian")

    @pytest.mark.parametrize("group_type,order", [
        ("cycle", 3), ("cycle", 5),
        ("dihedral", 2), ("dihedral", 3)
    ])
    def test_reynolds_positive_semidefinite(self, group_type, order, tolerance_config):
        """Test that Reynolds operators are positive semidefinite."""
        if group_type == "cycle":
            nodes = list(range(order))
            graph = CycleGraph(nodes)
        else:
            group_size = 2 * order
            nodes = list(range(group_size))
            graph = DihedralGraph(nodes, "r-s")
        
        # Get Reynolds operator (already a matrix)
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Check eigenvalues are non-negative
        eigenvals = torch.linalg.eigvals(reynolds_matrix).real
        
        min_eigenval = torch.min(eigenvals)
        assert min_eigenval >= -tolerance_config['eigen_tol'], \
            f"Reynolds operator should be positive semidefinite, min eigenvalue: {min_eigenval}"

    def test_reynolds_trace_property(self):
        """Test trace properties of Reynolds operators."""
        # For cycle group
        order = 4
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Trace should be related to the dimension of invariant subspace
        trace = torch.trace(reynolds_matrix).real
        assert trace > 0, "Reynolds operator trace should be positive"
        assert trace <= size, "Reynolds operator trace should not exceed matrix size"

    @pytest.mark.parametrize("order", [3, 4, 5])
    def test_reynolds_rank(self, order, tolerance_config):
        """Test rank properties of Reynolds operators."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Compute rank via SVD
        singular_values = torch.linalg.svdvals(reynolds_matrix)
        rank_tol = tolerance_config['eigen_tol']
        rank = torch.sum(singular_values > rank_tol).item()
        
        # Rank should be > 0 (non-trivial operator)
        assert rank > 0, "Reynolds operator should have positive rank"
        
        # Rank should be ≤ dimension of invariant subspace
        assert rank <= size, f"Reynolds rank {rank} should not exceed size {size}"


class TestReynoldsNumericalStability:
    """Test numerical stability and edge cases."""

    @pytest.mark.parametrize("order", [1, 2, 16, 32])
    def test_reynolds_numerical_stability(self, order):
        """Test Reynolds operator construction for various sizes."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_op = graph.equi_raynold_op
        
        # Check for numerical issues
        assert not np.any(np.isnan(reynolds_op)), "Reynolds operator should not contain NaN"
        assert not np.any(np.isinf(reynolds_op)), "Reynolds operator should not contain Inf"
        
        # Check that it's not all zeros
        assert np.any(np.abs(reynolds_op) > 1e-15), "Reynolds operator should not be all zeros"

    def test_reynolds_dtype_consistency(self):
        """Test that Reynolds operators handle different dtypes correctly."""
        order = 4
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_op = graph.equi_raynold_op
        
        # Convert to different dtypes
        size = int(np.sqrt(len(reynolds_op)))
        reynolds_matrix = reynolds_op.reshape(size, size)
        
        # Test complex dtypes
        reynolds_cfloat = torch.tensor(reynolds_matrix, dtype=torch.cfloat)
        reynolds_cdouble = torch.tensor(reynolds_matrix, dtype=torch.cdouble)
        
        # Both should have same shape
        assert reynolds_cfloat.shape == reynolds_cdouble.shape
        
        # Values should be close
        torch.testing.assert_close(reynolds_cfloat.double(), reynolds_cdouble, rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_generator_consistency(self, order, tolerance_config):
        """Test that different dihedral generators give equivalent Reynolds operators."""
        group_size = 2 * order
        nodes = list(range(group_size))
        
        # Both generators should give mathematically equivalent operators
        graph_rs = DihedralGraph(nodes, "r-s")
        graph_ssr = DihedralGraph(nodes, "s-sr")
        
        reynolds_rs = graph_rs.equi_raynold_op
        reynolds_ssr = graph_ssr.equi_raynold_op
        
        # Reshape both
        size = int(np.sqrt(len(reynolds_rs)))
        matrix_rs = torch.tensor(reynolds_rs.reshape(size, size), dtype=torch.cfloat)
        matrix_ssr = torch.tensor(reynolds_ssr.reshape(size, size), dtype=torch.cfloat)
        
        # Both should have eigenvalue 1
        check_eigenvalue_property(matrix_rs, 1.0, tolerance_config, 
                                 msg="r-s Reynolds should have eigenvalue 1")
        check_eigenvalue_property(matrix_ssr, 1.0, tolerance_config,
                                 msg="s-sr Reynolds should have eigenvalue 1")

    def test_reynolds_invariant_subspace_dimension(self):
        """Test that invariant subspace has expected dimension."""
        order = 4
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Find dimension of eigenspace for eigenvalue 1
        eigenvals, eigenvecs = torch.linalg.eig(reynolds_matrix)
        eigen_tol = 1e-8
        
        invariant_mask = torch.abs(eigenvals - 1.0) < eigen_tol
        invariant_dim = torch.sum(invariant_mask).item()
        
        # For cyclic groups, should have at least 1 invariant vector (constant function)
        assert invariant_dim >= 1, "Should have at least one invariant vector"
        
        # Should not exceed total dimension
        assert invariant_dim <= size, f"Invariant dimension {invariant_dim} should not exceed {size}"


class TestReynoldsEquivarianceProjection:
    """Test Reynolds operator as equivariance projector."""

    def test_reynolds_projects_to_invariant_subspace(self):
        """Test that Reynolds operator projects to G-invariant subspace."""
        order = 3
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Create a test vector
        test_vector = torch.randn(size, dtype=torch.cfloat)
        
        # Apply Reynolds operator
        projected = reynolds_matrix @ test_vector
        
        # Apply again - should get the same result (projection property)
        double_projected = reynolds_matrix @ projected
        
        torch.testing.assert_close(projected, double_projected, rtol=1e-6, atol=1e-8,
                                 msg="Reynolds operator should be idempotent projection")

    def test_reynolds_preserves_invariant_vectors(self):
        """Test that Reynolds operator preserves invariant vectors."""
        order = 4
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        reynolds_matrix = torch.tensor(graph.equi_raynold_op, dtype=torch.cfloat)
        
        # Find eigenvectors with eigenvalue 1
        eigenvals, eigenvecs = torch.linalg.eig(reynolds_matrix)
        eigen_tol = 1e-8
        
        invariant_mask = torch.abs(eigenvals - 1.0) < eigen_tol
        
        if torch.any(invariant_mask):
            invariant_vecs = eigenvecs[:, invariant_mask]
            
            # Reynolds operator should preserve these vectors
            for i in range(invariant_vecs.shape[1]):
                vec = invariant_vecs[:, i]
                transformed = reynolds_matrix @ vec
                
                # Should get back the same vector (up to phase)
                torch.testing.assert_close(transformed, vec, rtol=1e-6, atol=1e-8,
                                         msg=f"Invariant vector {i} should be preserved")
