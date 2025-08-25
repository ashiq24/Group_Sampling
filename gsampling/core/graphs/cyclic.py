"""
Cyclic group Cayley graph implementation.

Implements the AbstractGroupGraph interface for cyclic groups C_n.
Constructs a simple cycle graph structure with DFT-based Fourier basis
and Reynolds operator for equivariant projections.
"""

import numpy as np
from typing import List, Optional
from .base import AbstractGroupGraph
from escnn.group import cyclic_group, directsum


class CycleGraph(AbstractGroupGraph):
    """
    Cayley graph implementation for cyclic groups C_n.
    
    **Group Structure:**
    Cyclic group C_n has n elements: {e, r, r², ..., r^(n-1)}
    Elements are labeled as integers [0, 1, ..., n-1].
    
    **Graph Structure:**
    Simple cycle: each element connects to the next via generator r.
    Edge connectivity: (0→1, 1→2, ..., (n-1)→0)
    
    **Mathematical Properties:**
    - Fourier basis is the DFT matrix (unitary)
    - Reynolds operator normalized by group order n
    - All irreps are 1-dimensional for cyclic groups
    """
    
    def __init__(self, nodes: List[int], generator: Optional[str] = None):
        """
        Initialize cyclic group graph.
        
        Args:
            nodes: List of integer node labels
            generator: Ignored for cyclic groups (only one generator r)
        """
        super().__init__(nodes, generator)
    
    def _build_graph_structure(self):
        """Build cycle graph structure."""
        group_size = len(self.nodes)
        
        # Build cycle edges: i → (i+1) mod n
        self.edges = []
        for i in range(group_size):
            self.edges.append((i, (i + 1) % group_size))
        
        # Build adjacency matrices
        self._build_adjacency_matrices(group_size)
        
        # For cycle graphs, smoother equals directed adjacency
        self._smoother = self.directed_adjacency_matrix.copy()
    
    def _build_adjacency_matrices(self, group_size: int):
        """Build symmetric and directed adjacency matrices for cycle."""
        # Symmetric adjacency matrix
        self._adjacency_matrix = np.zeros((group_size, group_size))
        for edge in self.edges:
            self._adjacency_matrix[edge[0], edge[1]] = 1
            self._adjacency_matrix[edge[1], edge[0]] = 1

        # Directed adjacency matrix
        self._directed_adjacency_matrix = np.zeros((group_size, group_size))
        for edge in self.edges:
            self._directed_adjacency_matrix[edge[0], edge[1]] = 1
    
    def _build_spectral_operators(self):
        """Build Fourier basis and Reynolds operator for cyclic group."""
        group_size = len(self.nodes)
        self._fourier_basis = self.get_basis(group_size)
        self._equi_raynold_op = self.get_equi_raynold(group_size)
    
    def get_basis(self, order: int) -> np.ndarray:
        """
        Get unitary Fourier basis from cyclic group irreps.
        
        **Implementation:**
        Uses ESCNN's cyclic_group regular representation change_of_basis,
        which provides the DFT matrix with proper normalization.
        
        **Mathematical Properties:**
        - Unitary: Φ @ Φ† = I
        - Square matrix: n × n for C_n
        - Equivalent to normalized DFT matrix
        """
        G = cyclic_group(order)
        return G.regular_representation.change_of_basis
    
    def get_irreps(self, order: int):
        """
        Get direct sum of irreps for cyclic group C_n.
        
        **Mathematical Background:**
        Cyclic groups have n irreps, all 1-dimensional:
        ρ_k(r^j) = exp(2πijk/n) for k,j = 0,...,n-1
        
        **Implementation:**
        Direct sum of all irreps without multiplicities (each irrep appears once).
        """
        G = cyclic_group(order)
        g_dir_sum = []
        
        for rho in G.irreps():
            g_dir_sum.append(rho)
        
        return directsum(g_dir_sum)
    
    def get_equi_raynold(self, order: int) -> np.ndarray:
        """
        Compute Reynolds operator for cyclic group C_n.
        
        **Formula:**
        R = (1/n) Σ_{g∈C_n} ρ(g) ⊗ ρ(g⁻¹)ᵀ
        
        where ρ is the direct sum of irreps acting on Fourier coefficients.
        
        **Properties:**
        - Normalized by group order n (not 2n like dihedral)
        - Projects operators to C_n-equivariant subspace
        - Simpler structure than dihedral due to abelian group property
        """
        G = cyclic_group(order)
        k = self.get_irreps(order)
        size = G.regular_representation.change_of_basis.shape[-1]
        
        # Initialize with correct dimensions for Kronecker products
        # Reynolds operator acts on vectorized matrices, so it's (size²) × (size²)
        kron_size = size * size
        equi_rey = np.zeros((kron_size, kron_size))
        for g in G.elements:
            kron_prod = np.kron(k(g), k(g.__invert__()).T)
            equi_rey = equi_rey + kron_prod
        
        equi_rey = equi_rey / order  # Normalize by group order
        return equi_rey
