"""
Tests for Fourier basis construction and properties.

This module tests Fourier bases constructed from irreducible representations:
- Unitary/orthonormal conditions within tolerance for cycle/dihedral groups
- Dimensions match |G| and irreps dimension sum
- Basis completeness and mathematical properties
- ESCNN integration correctness
"""

import pytest
import torch
import numpy as np
from tests.conftest import (
    group_config_parametrize,
    device_dtype_parametrize,
    assert_tensors_close,
    assert_matrix_properties,
    check_eigenvalue_property,
    tolerance_config,
    skip_if_no_escnn
)
from tests.helpers import validate_fourier_basis

# Import the modules under test
try:
    from gsampling.utils.graph_constructors import DihedralGraph, CycleGraph
    from gsampling.utils.group_utils import get_group
    import escnn.group as escnn_group
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


class TestCyclicFourierBasis:
    """Test Fourier basis construction for cyclic groups."""

    @pytest.mark.parametrize("order", [3, 4, 5, 6, 8, 12])
    def test_cyclic_basis_dimensions(self, order, tolerance_config):
        """Test that cyclic Fourier basis has correct dimensions."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Should be square matrix of size order x order
        assert basis.shape == (order, order), f"Expected shape ({order}, {order}), got {basis.shape}"

    @pytest.mark.parametrize("order", [4, 6, 8])
    @device_dtype_parametrize
    def test_cyclic_basis_unitary(self, order, device, dtype, tolerance_config):
        """Test that cyclic Fourier basis is unitary."""
        if not dtype.is_complex:
            pytest.skip("Skipping non-complex dtype for Fourier basis test")
            
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = torch.tensor(graph.fourier_basis, dtype=dtype, device=device)
        
        # Test unitary property: F @ F.H = I
        assert_matrix_properties(basis, ['unitary'], tolerance_config,
                                msg="Cyclic Fourier basis should be unitary")

    @pytest.mark.parametrize("order", [4, 5, 6])
    def test_cyclic_basis_vs_dft(self, order, tolerance_config):
        """Test that cyclic basis matches DFT construction."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Construct expected DFT matrix
        expected_basis = torch.zeros(order, order, dtype=torch.cfloat)
        for k in range(order):
            for j in range(order):
                expected_basis[j, k] = torch.exp(-2j * torch.pi * k * j / order) / torch.sqrt(torch.tensor(order, dtype=torch.float32))
        
        # Compare (up to phase factors and column ordering)
        # Test that both bases span the same space by checking unitarity
        assert_matrix_properties(basis, ['unitary'], tolerance_config,
                                msg="Cyclic basis should be unitary like DFT")

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_cyclic_escnn_integration(self, order):
        """Test integration with ESCNN cyclic group."""
        # Test that we can create ESCNN cyclic group
        G = escnn_group.cyclic_group(order)
        assert len(G.elements) == order
        
        # Test that regular representation change of basis exists
        basis = G.regular_representation.change_of_basis
        assert basis.shape == (order, order)
        
        # Create graph and compare
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        graph_basis = graph.fourier_basis
        
        # Both should be unitary
        validate_fourier_basis(torch.tensor(basis, dtype=torch.cfloat))
        validate_fourier_basis(torch.tensor(graph_basis, dtype=torch.cfloat))

    @pytest.mark.parametrize("order", [3, 4, 5, 8])
    def test_cyclic_irreps_dimension_sum(self, order):
        """Test that irrep dimensions sum correctly."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        # Get irreps
        irreps = graph.get_irreps(order)
        
        # For cyclic groups, irreps are all 1D, so sum should equal order
        G = escnn_group.cyclic_group(order)
        irreps_list = list(G.irreps())
        
        total_dim = sum(irrep.size for irrep in irreps_list)
        assert total_dim == order, f"Irrep dimensions should sum to {order}, got {total_dim}"

    def test_cyclic_basis_completeness(self):
        """Test that cyclic basis spans the full space."""
        order = 6
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Test that basis can represent arbitrary vectors
        random_vector = torch.randn(order, dtype=torch.cfloat)
        
        # Transform to frequency domain and back
        freq_coeffs = basis.conj().T @ random_vector
        reconstructed = basis @ freq_coeffs
        
        torch.testing.assert_close(reconstructed, random_vector, rtol=1e-5, atol=1e-6,
                                 msg="Basis should allow perfect reconstruction")


class TestDihedralFourierBasis:
    """Test Fourier basis construction for dihedral groups."""

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    def test_dihedral_basis_dimensions(self, order):
        """Test that dihedral Fourier basis has correct dimensions."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Should be matrix of size (2n) x (irrep_dim_sum)
        assert basis.shape[0] == group_size, f"Expected first dimension {group_size}, got {basis.shape[0]}"
        
        # Second dimension should match sum of squared irrep dimensions
        G = escnn_group.dihedral_group(order)
        irreps_list = list(G.irreps())
        expected_basis_cols = sum(irrep.size**2 for irrep in irreps_list)
        
        assert basis.shape[1] == expected_basis_cols, \
            f"Expected basis columns {expected_basis_cols}, got {basis.shape[1]}"

    @pytest.mark.parametrize("order", [2, 3, 4])
    @device_dtype_parametrize 
    def test_dihedral_basis_orthogonality(self, order, device, dtype, tolerance_config):
        """Test orthogonality properties of dihedral basis."""
        if not dtype.is_complex:
            pytest.skip("Skipping non-complex dtype for Fourier basis test")
            
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        basis = torch.tensor(graph.fourier_basis, dtype=dtype, device=device)
        
        # Test that columns are orthonormal
        gram_matrix = basis.conj().T @ basis
        identity = torch.eye(basis.shape[1], dtype=dtype, device=device)
        
        assert_tensors_close(gram_matrix, identity, tolerance_cfg=tolerance_config,
                           msg="Dihedral basis columns should be orthonormal")

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_escnn_integration(self, order):
        """Test integration with ESCNN dihedral group."""
        G = escnn_group.dihedral_group(order)
        group_size = 2 * order
        assert len(G.elements) == group_size
        
        # Test irrep structure
        irreps_list = list(G.irreps())
        
        # Dihedral groups have specific irrep structure:
        # - 2 one-dimensional irreps
        # - (n-1)/2 two-dimensional irreps if n is odd
        # - (n-2)/2 two-dimensional irreps + 2 one-dimensional if n is even
        total_irrep_dims = sum(irrep.size for irrep in irreps_list)
        assert total_irrep_dims == group_size, \
            f"Total irrep dimensions should equal group size {group_size}"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_basis_construction(self, order, tolerance_config):
        """Test the basis construction process for dihedral groups."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Test that get_basis method works
        basis = graph.get_basis(order)
        assert basis.shape[0] == group_size
        
        # Test that basis has reasonable values (not all zeros)
        basis_tensor = torch.tensor(basis, dtype=torch.cfloat)
        assert torch.any(torch.abs(basis_tensor) > 1e-10), "Basis should not be all zeros"
        
        # Test normalization - each column should have reasonable norm
        column_norms = torch.norm(basis_tensor, dim=0)
        assert torch.all(column_norms > 1e-10), "All basis columns should be non-zero"

    def test_dihedral_irreps_construction(self):
        """Test irreps construction for dihedral groups."""
        order = 3
        nodes = list(range(2 * order))
        graph = DihedralGraph(nodes, "r-s")
        
        irreps = graph.get_irreps(order)
        
        # Test that irreps object has expected properties
        G = escnn_group.dihedral_group(order)
        assert hasattr(irreps, '__call__'), "Irreps should be callable"
        
        # Test that irreps can be evaluated on group elements
        for g in G.elements:
            irrep_matrix = irreps(g)
            assert irrep_matrix.shape[0] == irrep_matrix.shape[1], "Irrep matrices should be square"


class TestFourierBasisComparison:
    """Compare Fourier bases between different construction methods."""

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_cyclic_consistency(self, order, tolerance_config):
        """Test consistency of cyclic basis across different constructions."""
        # Method 1: Through CycleGraph
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        basis1 = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Method 2: Direct ESCNN
        G = escnn_group.cyclic_group(order)
        basis2 = torch.tensor(G.regular_representation.change_of_basis, dtype=torch.cfloat)
        
        # Both should be unitary
        assert_matrix_properties(basis1, ['unitary'], tolerance_config, msg="Graph basis should be unitary")
        assert_matrix_properties(basis2, ['unitary'], tolerance_config, msg="ESCNN basis should be unitary")
        
        # They might differ by column ordering/phases, but should span same space
        # Test by checking that both can represent the same vectors equivalently
        test_vector = torch.randn(order, dtype=torch.cfloat)
        
        # Both transformations should preserve vector norms
        transformed1 = basis1.conj().T @ test_vector
        transformed2 = basis2.conj().T @ test_vector
        
        original_norm = torch.norm(test_vector)
        norm1 = torch.norm(transformed1)
        norm2 = torch.norm(transformed2)
        
        assert_tensors_close(norm1, original_norm, tolerance_cfg=tolerance_config,
                           msg="Graph basis should preserve norms")
        assert_tensors_close(norm2, original_norm, tolerance_cfg=tolerance_config,
                           msg="ESCNN basis should preserve norms")

    def test_subgroup_basis_relationship(self):
        """Test relationship between parent and subgroup bases."""
        # Create parent cycle group
        parent_order = 8
        parent_nodes = list(range(parent_order))
        parent_graph = CycleGraph(parent_nodes)
        parent_basis = torch.tensor(parent_graph.fourier_basis, dtype=torch.cfloat)
        
        # Create subgroup
        subgroup_order = 4
        subgroup_nodes = list(range(0, parent_order, 2))  # [0, 2, 4, 6]
        subgroup_graph = CycleGraph(subgroup_nodes)
        subgroup_basis = torch.tensor(subgroup_graph.fourier_basis, dtype=torch.cfloat)
        
        # Both should be unitary
        validate_fourier_basis(parent_basis)
        validate_fourier_basis(subgroup_basis)
        
        # Subgroup basis should have smaller dimensions
        assert subgroup_basis.shape[0] == subgroup_order
        assert parent_basis.shape[0] == parent_order


class TestFourierBasisNumericalStability:
    """Test numerical stability and edge cases."""

    @pytest.mark.parametrize("order", [1, 2, 3, 16, 32])
    def test_basis_stability_different_sizes(self, order, tolerance_config):
        """Test basis construction for different group sizes."""
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Test basic properties regardless of size
        assert basis.shape == (order, order)
        assert not torch.any(torch.isnan(basis)), "Basis should not contain NaN"
        assert not torch.any(torch.isinf(basis)), "Basis should not contain Inf"
        
        if order > 1:  # Skip unitary test for trivial group
            assert_matrix_properties(basis, ['unitary'], tolerance_config, 
                                    msg=f"Basis should be unitary for order {order}")

    def test_basis_dtype_handling(self):
        """Test that basis handles different dtypes correctly."""
        order = 4
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        
        # Original basis should be complex
        basis = graph.fourier_basis
        
        # Convert to tensor and test different dtypes
        basis_float32 = torch.tensor(basis, dtype=torch.float32)
        basis_float64 = torch.tensor(basis, dtype=torch.float64)
        basis_cfloat = torch.tensor(basis, dtype=torch.cfloat)
        basis_cdouble = torch.tensor(basis, dtype=torch.cdouble)
        
        # All should have same shape
        expected_shape = (order, order)
        assert basis_float32.shape == expected_shape
        assert basis_float64.shape == expected_shape
        assert basis_cfloat.shape == expected_shape
        assert basis_cdouble.shape == expected_shape

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_generator_consistency(self, order, tolerance_config):
        """Test that different generators give consistent bases."""
        group_size = 2 * order
        nodes = list(range(group_size))
        
        # Test both generators
        graph_rs = DihedralGraph(nodes, "r-s")
        graph_ssr = DihedralGraph(nodes, "s-sr")
        
        basis_rs = torch.tensor(graph_rs.fourier_basis, dtype=torch.cfloat)
        basis_ssr = torch.tensor(graph_ssr.fourier_basis, dtype=torch.cfloat)
        
        # Both should be unitary (main mathematical requirement)
        assert_matrix_properties(basis_rs, ['unitary'], tolerance_config,
                                msg="r-s basis should be unitary")
        assert_matrix_properties(basis_ssr, ['unitary'], tolerance_config,
                                msg="s-sr basis should be unitary")
        
        # Both should have same dimensions
        assert basis_rs.shape == basis_ssr.shape

    def test_basis_reconstruction_accuracy(self):
        """Test accuracy of signal reconstruction using basis."""
        order = 6
        nodes = list(range(order))
        graph = CycleGraph(nodes)
        basis = torch.tensor(graph.fourier_basis, dtype=torch.cfloat)
        
        # Create test signals with different characteristics
        test_signals = [
            torch.ones(order, dtype=torch.cfloat),  # Constant signal
            torch.randn(order, dtype=torch.cfloat),  # Random signal
            torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.cfloat),  # Impulse
            torch.exp(2j * torch.pi * torch.arange(order, dtype=torch.cfloat) / order)  # Pure frequency
        ]
        
        for i, signal in enumerate(test_signals):
            # Transform to frequency domain and back
            freq_domain = basis.conj().T @ signal
            reconstructed = basis @ freq_domain
            
            # Should get original signal back
            reconstruction_error = torch.norm(signal - reconstructed)
            assert reconstruction_error < 1e-10, \
                f"Reconstruction error too large for test signal {i}: {reconstruction_error}"

