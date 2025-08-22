"""
Octahedral group Cayley graph implementations.

Implements the AbstractGroupGraph interface for octahedral groups O and O_h.
The octahedral group O has 24 elements (rotational symmetries of cube/octahedron),
while the full octahedral group O_h has 48 elements (including inversions).
"""

import numpy as np
from typing import List, Optional
from .base import AbstractGroupGraph
from escnn.group import octa_group, full_octa_group, directsum


class OctahedralGraph(AbstractGroupGraph):
    """
    Cayley graph implementation for octahedral group O.
    
    **Group Structure:**
    Octahedral group O has 24 elements representing the rotational symmetries
    of a cube or octahedron:
    - 1 identity
    - 6 face rotations (90°, 180°, 270° around face-to-face axes)
    - 8 vertex rotations (120°, 240° around vertex-to-vertex axes)  
    - 9 edge rotations (180° around edge-to-edge axes)
    
    **Mathematical Properties:**
    - Non-abelian group with complex irrep structure
    - Multiple 1D, 2D, and 3D irreducible representations
    - Rich subgroup lattice (contains tetrahedral, dihedral, cyclic subgroups)
    
    **Applications:**
    - 3D crystallographic symmetries
    - Molecular symmetry analysis
    - 3D equivariant neural networks
    """
    
    def __init__(self, nodes: List[int], generator: Optional[str] = None):
        """
        Initialize octahedral group graph.
        
        Args:
            nodes: List of integer node labels (must have 24 elements)
            generator: Ignored for octahedral groups (fixed group structure)
        """
        if len(nodes) != 24:
            raise ValueError("Octahedral group must have exactly 24 elements")
        
        super().__init__(nodes, generator)
    
    def _build_graph_structure(self):
        """Build Cayley graph structure for octahedral group."""
        group_size = len(self.nodes)
        
        # For octahedral group, we build a complete connectivity based on
        # the group multiplication table from ESCNN
        self._build_octahedral_edges(group_size)
        self._build_adjacency_matrices(group_size)
        
        # Smoother is the directed adjacency for octahedral
        self._smoother = self.directed_adjacency_matrix.copy()
    
    def _build_octahedral_edges(self, group_size: int):
        self.G = octa_group()
        self.edges = []
        elements = list(self.G.elements)
        element_to_index = {g: i for i, g in enumerate(elements)}
        
        # Use group's generators for Cayley graph edges
        generators = self.G.generators
        for i, g in enumerate(elements):
            for generator in generators:
                h = g @ generator  # Group composition
                j = element_to_index[h]
                self.edges.append((i, j))
    
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
        """Build Fourier basis and Reynolds operator for octahedral group."""
        # Octahedral group has 24 elements, no order parameter needed
        self._fourier_basis = self.get_basis(24)
        self._equi_raynold_op = self.get_equi_raynold(24)
    
    def get_basis(self, order: int) -> np.ndarray:
        """
        Construct unitary Fourier basis from octahedral group irreps.
        
        **Mathematical Construction:**
        1. Get all irreps of O from ESCNN
        2. For each irrep ρ, compute ρ(g) for all g ∈ O
        3. Reshape and normalize properly for 3D group structure
        4. Concatenate all irrep contributions
        
        **Octahedral Group Irreps:**
        - A1: 1D trivial representation
        - A2: 1D alternating representation  
        - E: 2D representation
        - T1: 3D representation
        - T2: 3D representation
        
        **Normalization:**
        Each irrep contribution is scaled by √d/√|G| where d = irrep.size, |G| = 24
        """
        G = octa_group()
        basis = []
        
        for rho in G.irreps():
            d = rho.size ** 0.5
            rho_g = np.stack([rho(g) for g in G.elements], axis=0)
            
            # Reshape for proper basis construction (similar to dihedral)
            rho_g = (
                np.moveaxis(rho_g, -2, -1).reshape(rho_g.shape[0], -1)
                * d
                / (24) ** 0.5  # Normalize by group order
            )
            basis.append(rho_g)
        
        return np.concatenate(basis, axis=1)
    
    def get_irreps(self, order: int):
        """
        Get direct sum of irreps for octahedral group O.
        
        **Mathematical Background:**
        Octahedral group has 5 irrep classes:
        - A1, A2: 1-dimensional irreps
        - E: 2-dimensional irrep
        - T1, T2: 3-dimensional irreps
        
        **Implementation:**
        Direct sum with proper multiplicities for each irrep.
        """
        G = octa_group()
        g_dir_sum = []
        
        for rho in G.irreps():
            d = rho.size
            # Add each irrep with multiplicity equal to its dimension
            for i in range(d):
                g_dir_sum.append(rho)
        
        return directsum(g_dir_sum)
    
    def get_equi_raynold(self, order: int) -> np.ndarray:
        """
        Compute Reynolds operator for octahedral group O.
        
        **Formula:**
        R = (1/24) Σ_{g∈O} ρ(g) ⊗ ρ(g⁻¹)ᵀ
        
        where ρ is the direct sum of irreps acting on Fourier coefficients.
        
        **Properties:**
        - Normalized by group order 24
        - Projects operators to O-equivariant subspace
        - More complex than 2D groups due to 3D irrep structure
        """
        G = octa_group()
        k = self.get_irreps(order)
        size = G.regular_representation.change_of_basis.shape[-1]
        
        # Initialize with correct dimensions for Kronecker products
        # Reynolds operator acts on vectorized matrices, so it's (size²) × (size²)
        kron_size = size * size
        equi_rey = np.zeros((kron_size, kron_size))
        for g in G.elements:
            equi_rey = equi_rey + np.kron(k(g), k(g.__invert__()).T)
        
        equi_rey = equi_rey / 24  # Normalize by group order
        return equi_rey


class FullOctahedralGraph(AbstractGroupGraph):
    """
    Cayley graph implementation for full octahedral group O_h.
    
    **Group Structure:**
    Full octahedral group O_h has 48 elements representing the complete
    symmetry group of a cube or octahedron:
    - 24 rotations (same as octahedral group O)
    - 24 rotoreflections (rotations combined with inversion)
    
    **Mathematical Properties:**
    - Contains octahedral group O as a subgroup
    - Includes inversion symmetry
    - Even richer irrep structure than O
    - Important for crystallographic point groups
    
    **Applications:**
    - Full crystallographic symmetries
    - Inversion-symmetric molecular systems
    - Complete 3D symmetry analysis
    """
    
    def __init__(self, nodes: List[int], generator: Optional[str] = None):
        """
        Initialize full octahedral group graph.
        
        Args:
            nodes: List of integer node labels (must have 48 elements)
            generator: Ignored for full octahedral groups (fixed group structure)
        """
        if len(nodes) != 48:
            raise ValueError("Full octahedral group must have exactly 48 elements")
        
        super().__init__(nodes, generator)
    
    def _build_graph_structure(self):
        """Build Cayley graph structure for full octahedral group."""
        group_size = len(self.nodes)
        
        # Build connectivity for full octahedral group
        self._build_full_octahedral_edges(group_size)
        self._build_adjacency_matrices(group_size)
        
        # Smoother is the directed adjacency
        self._smoother = self.directed_adjacency_matrix.copy()
    
    def _build_full_octahedral_edges(self, group_size: int):
        """Build edges for full octahedral group based on group structure."""
        self.G = full_octa_group()
        self.edges = []
        elements = list(self.G.elements)
        element_to_index = {g: i for i, g in enumerate(elements)}
        
        # Use group's generators for Cayley graph edges
        generators = self.G.generators
        for i, g in enumerate(elements):
            for generator in generators:
                h = g @ generator  # Group composition
                j = element_to_index[h]
                self.edges.append((i, j))
    
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
        """Build Fourier basis and Reynolds operator for full octahedral group."""
        # Full octahedral group has 48 elements
        self._fourier_basis = self.get_basis(48)
        self._equi_raynold_op = self.get_equi_raynold(48)
    
    def get_basis(self, order: int) -> np.ndarray:
        """
        Construct unitary Fourier basis from full octahedral group irreps.
        
        **Mathematical Construction:**
        Similar to octahedral group but with additional irreps due to inversion symmetry.
        Full octahedral group O_h has more irrep classes than O.
        
        **Normalization:**
        Each irrep contribution is scaled by √d/√48 where d = irrep.size
        """
        G = full_octa_group()
        basis = []
        
        for rho in G.irreps():
            d = rho.size ** 0.5
            rho_g = np.stack([rho(g) for g in G.elements], axis=0)
            
            # Reshape for proper basis construction
            rho_g = (
                np.moveaxis(rho_g, -2, -1).reshape(rho_g.shape[0], -1)
                * d
                / (48) ** 0.5  # Normalize by group order
            )
            basis.append(rho_g)
        
        return np.concatenate(basis, axis=1)
    
    def get_irreps(self, order: int):
        """
        Get direct sum of irreps for full octahedral group O_h.
        
        **Mathematical Background:**
        Full octahedral group has 10 irrep classes (more than O due to inversion):
        - A1g, A1u, A2g, A2u: 1-dimensional irreps
        - Eg, Eu: 2-dimensional irreps
        - T1g, T1u, T2g, T2u: 3-dimensional irreps
        
        **Implementation:**
        Direct sum with proper multiplicities for each irrep.
        """
        G = full_octa_group()
        g_dir_sum = []
        
        for rho in G.irreps():
            d = rho.size
            # Add each irrep with multiplicity equal to its dimension
            for i in range(d):
                g_dir_sum.append(rho)
        
        return directsum(g_dir_sum)
    
    def get_equi_raynold(self, order: int) -> np.ndarray:
        """
        Compute Reynolds operator for full octahedral group O_h.
        
        **Formula:**
        R = (1/48) Σ_{g∈O_h} ρ(g) ⊗ ρ(g⁻¹)ᵀ
        
        where ρ is the direct sum of irreps acting on Fourier coefficients.
        
        **Properties:**
        - Normalized by group order 48
        - Projects operators to O_h-equivariant subspace
        - Includes inversion symmetry constraints
        """
        G = full_octa_group()
        k = self.get_irreps(order)
        size = G.regular_representation.change_of_basis.shape[-1]
        
        # Initialize with correct dimensions for Kronecker products
        kron_size = size * size
        equi_rey = np.zeros((kron_size, kron_size))
        for g in G.elements:
            equi_rey = equi_rey + np.kron(k(g), k(g.__invert__()).T)
        
        equi_rey = equi_rey / 48  # Normalize by group order
        return equi_rey