"""
Dihedral group Cayley graph implementation.

Implements the AbstractGroupGraph interface for dihedral groups D_n.
Supports different generator conventions ('r-s', 's-sr') and constructs
the appropriate Cayley graph structure, Fourier basis, and Reynolds operator.
"""

import numpy as np
from typing import List, Optional
from .base import AbstractGroupGraph, InvalidGeneratorError
from escnn.group import dihedral_group, directsum


class DihedralGraph(AbstractGroupGraph):
    """
    Cayley graph implementation for dihedral groups D_n.
    
    **Group Structure:**
    Dihedral group D_n has 2n elements: n rotations and n reflections.
    Elements are labeled as integers [0, 1, ..., 2n-1] where:
    - [0, ..., n-1]: rotations {e, r, r², ..., r^(n-1)}
    - [n, ..., 2n-1]: reflections {s, sr, sr², ..., sr^(n-1)}
    
    **Generator Conventions:**
    - 'r-s': Standard presentation with rotation r and reflection s
    - 's-sr': Alternative presentation (less common)
    
    **Mathematical Properties:**
    - Cayley graph connectivity depends on generator choice
    - Fourier basis constructed from D_n irreps (mix of 1D and 2D irreps)
    - Reynolds operator normalized by group order 2n
    """
    
    def __init__(self, nodes: List[int], generator: str = "r-s"):
        """
        Initialize dihedral group graph.
        
        Args:
            nodes: List of integer node labels (must have even length)
            generator: Generator convention ('r-s' or 's-sr')
        """
        if generator not in ["r-s", "s-sr"]:
            raise InvalidGeneratorError(f"Invalid generator '{generator}' for dihedral group. Use 'r-s' or 's-sr'")
        
        if len(nodes) % 2 != 0:
            raise ValueError("Dihedral group must have even number of elements")
        
        super().__init__(nodes, generator)
    
    def _build_graph_structure(self):
        """Build Cayley graph structure for dihedral group."""
        group_size = len(self.nodes)
        self.edges = []
        self.edges_generator_1 = []
        self.edges_generator_2 = []
        
        if self.generator == "r-s":
            self._build_rs_generator_structure(group_size)
        else:  # s-sr
            self._build_ssr_generator_structure(group_size)
        
        # Build adjacency matrices
        self._build_adjacency_matrices(group_size)
    
    def _build_rs_generator_structure(self, group_size: int):
        """Build graph structure for r-s generator convention."""
        # Rotation edges (generator r)
        for i in range(group_size // 2):
            edge = (i, (i + 1) % (group_size // 2))
            self.edges.append(edge)
            self.edges_generator_1.append(edge)

        # Reflection edges (generator r on reflection subgroup)
        for i in range(group_size // 2):
            edge = (
                group_size // 2 + i,
                group_size // 2 + ((i + 1) % (group_size // 2)),
            )
            self.edges.append(edge)
            self.edges_generator_1.append(edge)

        # Connection edges (generator s)
        for i in range(group_size // 2):
            edge = (i, i + group_size // 2)
            self.edges.append(edge)
            self.edges_generator_2.append(edge)

        # Build smoother as concatenation of generator matrices
        self._build_rs_smoother(group_size)
    
    def _build_ssr_generator_structure(self, group_size: int):
        """Build graph structure for s-sr generator convention."""
        r = self.nodes[: group_size // 2]
        sr = self.nodes[group_size // 2 :]
        sr = sr[:1] + sr[:0:-1]  # Reverse order for sr elements
        cycle_node = []

        # Interleave r and sr elements
        for i in range(group_size // 2):
            cycle_node.append(r[i])
            cycle_node.append(sr[i])
        
        # Create cycle from interleaved elements
        for i in range(group_size):
            self.edges.append((cycle_node[i], cycle_node[(i + 1) % group_size]))

        # Smoother is the directed adjacency for this case
        self._smoother = np.zeros((group_size, group_size))
        for edge in self.edges:
            self._smoother[edge[0], edge[1]] = 1
    
    def _build_rs_smoother(self, group_size: int):
        """Build smoother for r-s generator as concatenation of generator matrices."""
        # Generator 1 matrix (rotation generators)
        gen1_matrix = np.zeros((group_size, group_size))
        for edge in self.edges_generator_1:
            gen1_matrix[edge[0], edge[1]] = 1

        # Generator 2 matrix (reflection generators)
        gen2_matrix = np.zeros((group_size, group_size))
        for edge in self.edges_generator_2:
            gen2_matrix[edge[0], edge[1]] = 1
            gen2_matrix[edge[1], edge[0]] = 1  # Reflection is self-inverse

        # Concatenate generators for smoother
        self._smoother = np.concatenate([gen1_matrix, gen2_matrix], axis=0)
    
    def _build_adjacency_matrices(self, group_size: int):
        """Build symmetric and directed adjacency matrices."""
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
        """Build Fourier basis and Reynolds operator for dihedral group."""
        order = len(self.nodes) // 2  # D_n has 2n elements, n is the rotation order
        self._fourier_basis = self.get_basis(order)
        self._equi_raynold_op = self.get_equi_raynold(order)
    
    def get_basis(self, order: int) -> np.ndarray:
        """
        Construct unitary Fourier basis from dihedral group irreps.
        
        **Mathematical Construction:**
        1. Get all irreps of D_n from ESCNN
        2. For each irrep ρ, compute ρ(g) for all g ∈ D_n
        3. Reshape and normalize: (moveaxis, reshape, scale by √d/(2n)^0.5)
        4. Concatenate all irrep contributions
        
        **Normalization:**
        Each irrep contribution is scaled by √d/(2n)^0.5 where d = irrep.size
        """
        G = dihedral_group(order)
        basis = []
        
        for rho in G.irreps():
            d = rho.size ** 0.5
            rho_g = np.stack([rho(g) for g in G.elements], axis=0)
            
            # Critical reshape for proper basis construction
            rho_g = (
                np.moveaxis(rho_g, -2, -1).reshape(rho_g.shape[0], -1)
                * d
                / (2 * order) ** 0.5
            )
            basis.append(rho_g)
        
        return np.concatenate(basis, axis=1)
    
    def get_irreps(self, order: int):
        """
        Get direct sum of irreps for dihedral group D_n.
        
        **Implementation:**
        Creates direct sum with proper multiplicities for each irrep.
        For D_n, irreps include 1D and 2D representations.
        """
        G = dihedral_group(order)
        g_dir_sum = []
        
        for rho in G.irreps():
            d = rho.size
            # Add each irrep with multiplicity equal to its dimension
            for i in range(d):
                g_dir_sum.append(rho)
        
        return directsum(g_dir_sum)
    
    def get_equi_raynold(self, order: int) -> np.ndarray:
        """
        Compute Reynolds operator for dihedral group D_n.
        
        **Formula:**
        R = (1/(2n)) Σ_{g∈D_n} ρ(g) ⊗ ρ(g⁻¹)ᵀ
        
        where ρ is the direct sum of irreps acting on Fourier coefficients.
        
        **Properties:**
        - Normalized by group order 2n
        - Projects operators to D_n-equivariant subspace
        - Used for enforcing equivariance constraints in anti-aliasing
        """
        G = dihedral_group(order)
        k = self.get_irreps(order)
        size = G.regular_representation.change_of_basis.shape[-1]
        
        # Initialize with correct dimensions for Kronecker products
        # Reynolds operator acts on vectorized matrices, so it's (size²) × (size²)
        kron_size = size * size
        equi_rey = np.zeros((kron_size, kron_size))
        for g in G.elements:
            equi_rey = equi_rey + np.kron(k(g), k(g.__invert__()).T)
        
        equi_rey = equi_rey / (2 * order)  # Normalize by group order
        return equi_rey  # Already the correct shape
