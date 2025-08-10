"""
Tests for graph construction and properties.

This module tests the current GraphConstructor, DihedralGraph, and CycleGraph
implementations to validate:
- Node count, edge count, and symmetry properties
- Adjacency vs directed adjacency consistency
- Smoother shape and sparsity
- Basic graph structure properties
"""

import pytest
import torch
import numpy as np
from tests.conftest import (
    group_config_parametrize, 
    subsampling_config_parametrize,
    device_real_dtype_parametrize,
    assert_tensors_close,
    assert_matrix_properties,
    tolerance_config
)
from tests.helpers import TensorLayoutHelper, validate_sampling_matrix

# Import the modules under test
try:
    from gsampling.utils.graph_constructors import GraphConstructor, DihedralGraph, CycleGraph, subsample
    from gsampling.utils.group_utils import get_group, get_sub_group_element
except ImportError as e:
    pytest.skip(f"Cannot import gsampling modules: {e}", allow_module_level=True)


class TestSubsampleFunction:
    """Test the subsample function for different group combinations."""

    @pytest.mark.parametrize("group_size,factor,expected_size", [
        (8, 2, 4),
        (12, 3, 4),
        (16, 4, 4),
    ])
    def test_cycle_to_cycle_subsampling(self, group_size, factor, expected_size):
        """Test cycle -> cycle subsampling produces correct indices."""
        indices = subsample(group_size, "cycle", None, "cycle", factor)
        
        assert len(indices) == expected_size
        assert indices == list(range(0, group_size, factor))
        assert all(0 <= idx < group_size for idx in indices)

    @pytest.mark.parametrize("group_size,factor", [
        (8, 2),   # D_4 -> D_2
        (12, 3),  # D_6 -> D_2 
        (16, 4),  # D_8 -> D_2
    ])
    def test_dihedral_to_dihedral_subsampling(self, group_size, factor):
        """Test dihedral -> dihedral subsampling produces correct indices."""
        indices = subsample(group_size, "dihedral", "r-s", "dihedral", factor)
        
        expected_size = group_size // factor
        assert len(indices) == expected_size
        assert indices == list(range(0, group_size, factor))
        assert all(0 <= idx < group_size for idx in indices)

    @pytest.mark.parametrize("group_size,factor", [
        (8, 2),   # D_4 -> C_2
        (12, 4),  # D_6 -> C_3 (factor must be even)
        (16, 8),  # D_8 -> C_2
    ])
    def test_dihedral_to_cycle_subsampling(self, group_size, factor):
        """Test dihedral -> cycle subsampling produces correct indices."""
        # Factor must be even for dihedral -> cycle
        if factor % 2 != 0:
            pytest.skip("Factor must be even for dihedral -> cycle")
            
        indices = subsample(group_size, "dihedral", "r-s", "cycle", factor)
        
        expected_size = group_size // factor
        assert len(indices) == expected_size
        
        # Should only include rotation elements (first half)
        assert all(idx < group_size // 2 for idx in indices)

    def test_dihedral_adihedral_subsampling(self):
        """Test the mysterious 'adihedral' subsampling option."""
        # This tests the undocumented branch
        indices = subsample(8, "dihedral", "r-s", "adihedral", 2)
        
        # Based on the code: nodes[:group_size//2:factor] + nodes[group_size//2+1::factor]
        # For group_size=8, factor=2: [0, 2] + [5, 7] = [0, 2, 5, 7]
        expected = [0, 2, 5, 7]
        assert indices == expected

    def test_invalid_combinations(self):
        """Test that invalid combinations raise appropriate errors."""
        with pytest.raises(NotImplementedError):
            subsample(8, "invalid_group", None, "cycle", 2)
            
        with pytest.raises(NotImplementedError):
            subsample(8, "dihedral", "r-s", "invalid_subgroup", 2)

    def test_assertion_errors(self):
        """Test that invalid parameters trigger assertions."""
        # Non-even factor for dihedral -> cycle should fail
        with pytest.raises(AssertionError):
            subsample(8, "dihedral", "r-s", "cycle", 3)  # Odd factor
            
        # Wrong generator for dihedral operations
        with pytest.raises(AssertionError):
            subsample(8, "dihedral", "s-sr", "dihedral", 2)


class TestCycleGraph:
    """Test CycleGraph construction and properties."""

    @pytest.mark.parametrize("group_size", [3, 4, 5, 6, 8, 12])
    def test_cycle_graph_structure(self, group_size, tolerance_config):
        """Test basic structural properties of cycle graphs."""
        nodes = list(range(group_size))
        graph = CycleGraph(nodes)
        
        # Check basic properties
        assert len(graph.nodes) == group_size
        assert len(graph.edges) == group_size
        
        # Check adjacency matrix properties
        adj = torch.tensor(graph.adjacency_matrix, dtype=torch.float32)
        
        # Should be symmetric
        assert_matrix_properties(adj, ['symmetric'], tolerance_config, 
                                msg="Cycle adjacency matrix should be symmetric")
        
        # Each node should have degree 2
        degrees = torch.sum(adj, dim=1)
        expected_degrees = torch.full((group_size,), 2.0)
        assert_tensors_close(degrees, expected_degrees, tolerance_cfg=tolerance_config,
                           msg="Each node in cycle should have degree 2")

    @pytest.mark.parametrize("group_size", [4, 6, 8])
    def test_cycle_graph_connectivity(self, group_size):
        """Test that cycle graph forms proper cycle connectivity."""
        nodes = list(range(group_size))
        graph = CycleGraph(nodes)
        
        # Check edges form a cycle: (0,1), (1,2), ..., (n-1,0)
        expected_edges = [(i, (i + 1) % group_size) for i in range(group_size)]
        assert set(graph.edges) == set(expected_edges)
        
        # Check directed adjacency
        dir_adj = graph.directed_adjacency_matrix
        for i in range(group_size):
            assert dir_adj[i, (i + 1) % group_size] == 1
            # Count outgoing edges - should be exactly 1 per node
            assert np.sum(dir_adj[i, :]) == 1

    @pytest.mark.parametrize("group_size", [4, 6, 8])
    def test_cycle_smoother_properties(self, group_size, tolerance_config):
        """Test properties of the smoother matrix."""
        nodes = list(range(group_size))
        graph = CycleGraph(nodes)
        
        smoother = torch.tensor(graph.smoother, dtype=torch.float32)
        
        # Smoother should equal directed adjacency for cycle graphs
        dir_adj = torch.tensor(graph.directed_adjacency_matrix, dtype=torch.float32)
        assert_tensors_close(smoother, dir_adj, tolerance_cfg=tolerance_config,
                           msg="Cycle smoother should equal directed adjacency")
        
        # Check sparsity - should have exactly group_size non-zero entries
        non_zero_count = torch.count_nonzero(smoother)
        assert non_zero_count == group_size, f"Expected {group_size} non-zero entries, got {non_zero_count}"


class TestDihedralGraph:
    """Test DihedralGraph construction and properties."""

    @pytest.mark.parametrize("order", [2, 3, 4, 6])
    @pytest.mark.parametrize("generator", ["r-s", "s-sr"])
    def test_dihedral_graph_structure(self, order, generator, tolerance_config):
        """Test basic structural properties of dihedral graphs."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, generator)
        
        # Check basic properties
        assert len(graph.nodes) == group_size
        assert len(graph.edges) > 0
        
        # Check adjacency matrix properties
        adj = torch.tensor(graph.adjacency_matrix, dtype=torch.float32)
        
        # Should be symmetric
        assert_matrix_properties(adj, ['symmetric'], tolerance_config,
                                msg="Dihedral adjacency matrix should be symmetric")

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_rs_generator_structure(self, order, tolerance_config):
        """Test specific structure for r-s generator."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Check generator-specific edge structure
        
        # Rotation edges should form cycles
        rotation_edges = [(i, (i + 1) % order) for i in range(order)]
        reflection_edges = [(order + i, order + ((i + 1) % order)) for i in range(order)]
        connection_edges = [(i, i + order) for i in range(order)]
        
        # Check that expected edges are present
        for edge in rotation_edges + reflection_edges + connection_edges:
            assert edge in graph.edges, f"Expected edge {edge} not found"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_dihedral_smoother_properties(self, order):
        """Test properties of dihedral smoother matrix."""
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        smoother = graph.smoother
        
        # For r-s generator, smoother is concatenation of two generator matrices
        expected_shape = (2 * group_size, group_size)
        assert smoother.shape == expected_shape, f"Expected smoother shape {expected_shape}, got {smoother.shape}"
        
        # Check that smoother is built from generator matrices
        gen1_matrix = graph.directed_adjacency_matrix_generator_1
        gen2_matrix = graph.directed_adjacency_matrix_generator_2
        
        expected_smoother = np.concatenate([gen1_matrix, gen2_matrix], axis=0)
        np.testing.assert_array_equal(smoother, expected_smoother,
                                    err_msg="Smoother should be concatenation of generator matrices")

    def test_dihedral_edge_consistency(self):
        """Test consistency between edge lists and adjacency matrices."""
        order = 3
        group_size = 2 * order
        nodes = list(range(group_size))
        graph = DihedralGraph(nodes, "r-s")
        
        # Check undirected adjacency consistency
        adj = graph.adjacency_matrix
        for i, j in graph.edges:
            assert adj[i, j] == 1, f"Edge ({i},{j}) not in adjacency matrix"
            assert adj[j, i] == 1, f"Reverse edge ({j},{i}) not in adjacency matrix"
        
        # Check directed adjacency consistency  
        dir_adj = graph.directed_adjacency_matrix
        for i, j in graph.edges:
            assert dir_adj[i, j] == 1, f"Directed edge ({i},{j}) not in directed adjacency matrix"


class TestGraphConstructor:
    """Test GraphConstructor orchestration of graph building."""

    @subsampling_config_parametrize
    def test_graph_constructor_initialization(self, group_type, order, subgroup_type, 
                                            subsampling_factor, generator):
        """Test GraphConstructor initializes correctly."""
        if group_type == "dihedral":
            group_size = 2 * order
        else:
            group_size = order
            
        constructor = GraphConstructor(
            group_size=group_size,
            group_type=group_type, 
            group_generator=generator,
            subgroup_type=subgroup_type,
            subsampling_factor=subsampling_factor
        )
        
        # Check basic properties
        assert constructor.group_size == group_size
        assert constructor.group_type == group_type
        assert constructor.subsampling_factor == subsampling_factor
        assert constructor.subgroup_size == group_size // subsampling_factor
        
        # Check graphs exist
        assert hasattr(constructor, 'graph')
        assert hasattr(constructor, 'subgroup_graph')
        
        # Check graph types
        if group_type == "dihedral":
            assert isinstance(constructor.graph, DihedralGraph)
        elif group_type in ["cycle", "cyclic"]:
            assert isinstance(constructor.graph, CycleGraph)
            
        if subgroup_type in ["cycle", "cyclic"]:
            assert isinstance(constructor.subgroup_graph, CycleGraph)
        elif subgroup_type == "dihedral":
            assert isinstance(constructor.subgroup_graph, DihedralGraph)

    def test_graph_constructor_divisibility_assertion(self):
        """Test that GraphConstructor enforces divisibility constraint."""
        with pytest.raises(AssertionError):
            GraphConstructor(
                group_size=7,  # Prime number
                group_type="cycle",
                group_generator=None,
                subgroup_type="cycle", 
                subsampling_factor=3  # 7 not divisible by 3
            )

    @pytest.mark.parametrize("group_size,subsampling_factor", [
        (8, 2), (12, 3), (16, 4)
    ])
    def test_cycle_subgroup_construction(self, group_size, subsampling_factor):
        """Test cycle -> cycle subgroup construction."""
        constructor = GraphConstructor(
            group_size=group_size,
            group_type="cycle",
            group_generator=None,
            subgroup_type="cycle",
            subsampling_factor=subsampling_factor
        )
        
        # Check subgroup properties
        expected_subgroup_size = group_size // subsampling_factor
        assert len(constructor.subgroup_graph.nodes) == expected_subgroup_size
        
        # Check that subgroup nodes are correctly sampled
        expected_nodes = list(range(0, group_size, subsampling_factor))
        assert constructor.subgroup_graph.nodes == expected_nodes[:expected_subgroup_size]

    def test_dihedral_subgroup_construction(self):
        """Test dihedral -> dihedral subgroup construction."""
        constructor = GraphConstructor(
            group_size=8,  # D_4
            group_type="dihedral",
            group_generator="r-s",
            subgroup_type="dihedral", 
            subsampling_factor=2
        )
        
        # Check subgroup properties
        assert len(constructor.subgroup_graph.nodes) == 4  # D_2
        
        # Check that subgroup nodes are correctly sampled (every 2nd element)
        expected_nodes = [0, 2, 4, 6]
        assert constructor.subgroup_graph.nodes == expected_nodes


class TestGraphValidation:
    """Validation tests for graph mathematical properties."""

    @pytest.mark.parametrize("graph_type,size", [
        ("cycle", 4), ("cycle", 6), ("cycle", 8),
        ("dihedral", 6), ("dihedral", 8), ("dihedral", 10)
    ])
    def test_adjacency_matrix_properties(self, graph_type, size, tolerance_config):
        """Test mathematical properties of adjacency matrices."""
        if graph_type == "cycle":
            nodes = list(range(size))
            graph = CycleGraph(nodes)
        else:
            # Dihedral group D_n has 2n elements
            assert size % 2 == 0, "Dihedral group size must be even"
            nodes = list(range(size))
            graph = DihedralGraph(nodes, "r-s")
        
        adj = torch.tensor(graph.adjacency_matrix, dtype=torch.float32)
        
        # Test symmetry
        assert_matrix_properties(adj, ['symmetric'], tolerance_config,
                                msg=f"{graph_type} adjacency should be symmetric")
        
        # Test that diagonal is zero (no self-loops)
        diagonal = torch.diag(adj)
        zero_diagonal = torch.zeros_like(diagonal)
        assert_tensors_close(diagonal, zero_diagonal, tolerance_cfg=tolerance_config,
                           msg="Adjacency matrix should have zero diagonal")
        
        # Test binary values
        unique_values = torch.unique(adj)
        assert len(unique_values) <= 2, "Adjacency matrix should be binary"
        assert torch.all((adj == 0) | (adj == 1)), "Adjacency values should be 0 or 1"

    @pytest.mark.parametrize("size", [4, 6, 8])
    def test_cycle_spectral_properties(self, size, tolerance_config):
        """Test spectral properties of cycle graphs."""
        nodes = list(range(size))
        graph = CycleGraph(nodes)
        
        adj = torch.tensor(graph.adjacency_matrix, dtype=torch.float32)
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(adj).real
        
        # For cycle graphs, eigenvalues should be 2*cos(2πk/n) for k=0,...,n-1
        expected_eigenvals = torch.tensor([2 * torch.cos(2 * torch.pi * k / size) for k in range(size)])
        
        # Sort both for comparison
        eigenvals_sorted = torch.sort(eigenvals)[0]
        expected_sorted = torch.sort(expected_eigenvals)[0]
        
        assert_tensors_close(eigenvals_sorted, expected_sorted, tolerance_cfg=tolerance_config,
                           msg="Cycle graph eigenvalues should match theoretical values")

    def test_graph_connectivity(self):
        """Test that graphs are connected."""
        # Test cycle graph connectivity
        graph = CycleGraph(list(range(5)))
        adj = graph.adjacency_matrix
        
        # Simple connectivity test: check if we can reach all nodes from node 0
        # This is a basic test - for more rigorous testing we'd use proper graph algorithms
        visited = set()
        stack = [0]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            
            # Add neighbors to stack
            for neighbor in range(len(adj)):
                if adj[node, neighbor] == 1 and neighbor not in visited:
                    stack.append(neighbor)
        
        assert len(visited) == len(graph.nodes), "Graph should be connected"

    def test_subgraph_relationship(self):
        """Test that subgraph nodes are subset of parent graph nodes."""
        constructor = GraphConstructor(
            group_size=8,
            group_type="cycle", 
            group_generator=None,
            subgroup_type="cycle",
            subsampling_factor=2
        )
        
        parent_nodes = set(constructor.graph.nodes)
        subgroup_nodes = set(constructor.subgroup_graph.nodes)
        
        # Subgroup nodes should be subset of parent nodes
        assert subgroup_nodes.issubset(parent_nodes), \
            "Subgroup nodes should be subset of parent group nodes"
