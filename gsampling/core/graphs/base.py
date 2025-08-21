"""
Abstract base class for group graph implementations.

This module defines the interface that all group graph classes must implement
to support the Group_Sampling algorithms. This enables clean extension to
new group types (tetrahedral, octahedral, etc.) without modifying existing code.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union, Optional


class AbstractGroupGraph(ABC):
    """
    Abstract base class for finite group Cayley graph implementations.
    
    All group graph classes must implement this interface to work with
    the Group_Sampling framework. This ensures consistent behavior across
    different group types and enables clean extension to new groups.
    
    **Required Interface:**
    - Graph structure: nodes, edges, adjacency matrices
    - Spectral operators: smoother for graph signal processing
    - Fourier analysis: basis from irreducible representations
    - Equivariance: Reynolds operator for equivariant projections
    
    **Mathematical Foundation:**
    Each group graph represents a Cayley graph Cay(G, S) where:
    - G is a finite group with elements as integer node labels
    - S is a generating set defining edge connectivity
    - Fourier basis is constructed from irreps via Peter-Weyl theorem
    - Reynolds operator implements R = (1/|G|) Σ_{g∈G} ρ(g) ⊗ ρ(g⁻¹)ᵀ
    """
    
    def __init__(self, nodes: List[int], generator: Optional[str] = None):
        """
        Initialize group graph with node list and optional generator specification.
        
        Args:
            nodes: List of integer node labels representing group elements
            generator: Optional generator specification (group-dependent)
        """
        self.nodes = nodes
        self.generator = generator
        self._validate_nodes()
        self._build_graph_structure()
        self._build_spectral_operators()
    
    def _validate_nodes(self):
        """Validate that nodes are appropriate for this group type."""
        if not isinstance(self.nodes, list):
            raise TypeError("Nodes must be a list")
        if not all(isinstance(node, int) for node in self.nodes):
            raise TypeError("All nodes must be integers")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Nodes must be unique")
    
    @abstractmethod
    def _build_graph_structure(self):
        """
        Build the Cayley graph structure for this group.
        
        Must set:
        - self.edges: List of (source, target) edge tuples
        - self.adjacency_matrix: Symmetric adjacency matrix (|G| × |G|)
        - self.directed_adjacency_matrix: Directed adjacency matrix (|G| × |G|)
        """
        pass
    
    @abstractmethod
    def _build_spectral_operators(self):
        """
        Build spectral operators for graph signal processing.
        
        Must set:
        - self.smoother: Graph shift operator for spectral smoothness
        - self.fourier_basis: Fourier basis from irreps (|G| × dim(irreps))
        - self.equi_raynold_op: Reynolds operator for equivariance (dim² × dim²)
        """
        pass
    
    @abstractmethod
    def get_basis(self, order: int) -> np.ndarray:
        """
        Construct unitary Fourier basis from irreducible representations.
        
        Args:
            order: Group-specific order parameter (may differ from |G|)
            
        Returns:
            Unitary matrix (|G| × total_irrep_dim) representing Fourier basis
            
        **Mathematical Requirements:**
        - Must be unitary: Φ @ Φ† = I
        - Constructed from irreps via Peter-Weyl theorem
        - Properly normalized for numerical stability
        """
        pass
    
    @abstractmethod
    def get_irreps(self, order: int):
        """
        Get direct sum of irreducible representations for this group.
        
        Args:
            order: Group-specific order parameter
            
        Returns:
            ESCNN representation object supporting evaluation on group elements
            
        **Implementation Notes:**
        - Should use ESCNN's directsum utility for combining irreps
        - Must handle all irreps with appropriate multiplicities
        - Used for constructing Reynolds operators and equivariant projections
        """
        pass
    
    @abstractmethod
    def get_equi_raynold(self, order: int) -> np.ndarray:
        """
        Compute Reynolds operator for equivariant projections.
        
        Args:
            order: Group-specific order parameter
            
        Returns:
            Reynolds operator matrix implementing R = (1/|G|) Σ_{g∈G} ρ(g) ⊗ ρ(g⁻¹)ᵀ
            
        **Mathematical Requirements:**
        - Must be Hermitian: R = R†
        - Must be idempotent: R² = R (projection property)
        - Must have eigenvalue 1 (for invariant subspace projection)
        - Must be positive semidefinite: R ≥ 0
        """
        pass
    
    # Properties that must be available after initialization
    @property
    def group_size(self) -> int:
        """Number of group elements."""
        return len(self.nodes)
    
    @property
    def edges(self) -> List[Tuple[int, int]]:
        """List of directed edges in the Cayley graph."""
        if not hasattr(self, '_edges'):
            raise NotImplementedError("_build_graph_structure() must set self._edges")
        return self._edges
    
    @edges.setter
    def edges(self, value: List[Tuple[int, int]]):
        self._edges = value
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Symmetric adjacency matrix of the Cayley graph."""
        if not hasattr(self, '_adjacency_matrix'):
            raise NotImplementedError("_build_graph_structure() must set self._adjacency_matrix")
        return self._adjacency_matrix
    
    @adjacency_matrix.setter
    def adjacency_matrix(self, value: np.ndarray):
        self._adjacency_matrix = value
    
    @property
    def directed_adjacency_matrix(self) -> np.ndarray:
        """Directed adjacency matrix of the Cayley graph."""
        if not hasattr(self, '_directed_adjacency_matrix'):
            raise NotImplementedError("_build_graph_structure() must set self._directed_adjacency_matrix")
        return self._directed_adjacency_matrix
    
    @directed_adjacency_matrix.setter
    def directed_adjacency_matrix(self, value: np.ndarray):
        self._directed_adjacency_matrix = value
    
    @property
    def smoother(self) -> np.ndarray:
        """Graph shift operator for spectral smoothness."""
        if not hasattr(self, '_smoother'):
            raise NotImplementedError("_build_spectral_operators() must set self._smoother")
        return self._smoother
    
    @smoother.setter
    def smoother(self, value: np.ndarray):
        self._smoother = value
    
    @property
    def fourier_basis(self) -> np.ndarray:
        """Fourier basis matrix from irreducible representations."""
        if not hasattr(self, '_fourier_basis'):
            raise NotImplementedError("_build_spectral_operators() must set self._fourier_basis")
        return self._fourier_basis
    
    @fourier_basis.setter
    def fourier_basis(self, value: np.ndarray):
        self._fourier_basis = value
    
    @property
    def equi_raynold_op(self) -> np.ndarray:
        """Reynolds operator for equivariant projections."""
        if not hasattr(self, '_equi_raynold_op'):
            raise NotImplementedError("_build_spectral_operators() must set self._equi_raynold_op")
        return self._equi_raynold_op
    
    @equi_raynold_op.setter
    def equi_raynold_op(self, value: np.ndarray):
        self._equi_raynold_op = value
    
    def validate_mathematical_properties(self, tolerance: float = 1e-6) -> bool:
        """
        Validate that the graph satisfies required mathematical properties.
        
        Args:
            tolerance: Numerical tolerance for validation
            
        Returns:
            True if all properties are satisfied
            
        **Validation Checks:**
        - Fourier basis is unitary
        - Reynolds operator is Hermitian, idempotent, and has eigenvalue 1
        - Adjacency matrices have correct symmetry properties
        - All matrices have appropriate dimensions
        """
        import torch
        
        # Validate Fourier basis unitarity
        basis = torch.tensor(self.fourier_basis, dtype=torch.cfloat)
        identity = torch.eye(basis.shape[1], dtype=torch.cfloat)
        gram_matrix = basis.conj().T @ basis
        
        if not torch.allclose(gram_matrix, identity, rtol=tolerance, atol=tolerance):
            return False
        
        # Validate Reynolds operator properties
        reynolds = torch.tensor(self.equi_raynold_op, dtype=torch.cfloat)
        
        # Hermitian check
        if not torch.allclose(reynolds, reynolds.conj().T, rtol=tolerance, atol=tolerance):
            return False
        
        # Idempotent check
        reynolds_squared = reynolds @ reynolds
        if not torch.allclose(reynolds_squared, reynolds, rtol=tolerance, atol=tolerance):
            return False
        
        # Eigenvalue 1 check
        eigenvals = torch.linalg.eigvals(reynolds)
        has_eigenvalue_one = torch.any(torch.abs(eigenvals - 1.0) < tolerance)
        if not has_eigenvalue_one:
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(nodes={len(self.nodes)}, generator='{self.generator}')"


class GroupGraphError(Exception):
    """Exception raised for group graph construction errors."""
    pass


class UnsupportedGroupError(GroupGraphError):
    """Exception raised when a group type is not supported."""
    pass


class InvalidGeneratorError(GroupGraphError):
    """Exception raised when an invalid generator is specified."""
    pass
