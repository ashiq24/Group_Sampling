"""
Helper Utilities for Group Equivariant Anti-Aliasing Layers

This module provides utility classes and functions for constructing and manipulating
graph-based smoothness operators, Reynolds projectors, and Fourier transforms used
in group equivariant anti-aliasing operations.

Mathematical Foundation:
------------------------
The utilities in this module support:

1. Graph Smoothness Operators:
   - Laplacian: L = D - A (degree matrix - adjacency matrix)
   - Normalized Laplacian: L_norm = D^-½ L D^-½
   - Row-normalized Adjacency: A_norm = A / diag(sum(A))

2. Reynolds Projection:
   - Computes equivariant projector: P = Q(QᵀQ)⁻¹Qᵀ
   - Where Q contains eigenvectors with eigenvalue ≈ 1

3. L1 Projection:
   - Projects to invariant subspace of mapping matrix M
   - L1 = V(VᵀV)⁻¹Vᵀ where V spans range(M)

4. Fourier Transforms:
   - Forward: X̂ = Bᵀx (complex case: B̄ᵀx)
   - Inverse: x = BX̂

Key Features:
- Centralized operator construction for reusability
- Support for multiple smoothness operators
- Efficient Fourier transform operations
- Numerical stability considerations

Author: Group Sampling Team
"""

import torch
from scipy.optimize import minimize, LinearConstraint
import numpy as np
from einops import rearrange



class SmoothOperatorFactory:
    """Factory to construct graph smoothness/shift operators.

    This factory centralizes the construction of various graph-based smoothness operators
    used in spectral anti-aliasing. It provides a unified interface for creating
    Laplacian, adjacency, and custom graph shift operators.

    Mathematical Operations:
    - Adjacency: Row-normalized stochastic matrix
    - Laplacian: L = D - A (degree matrix - adjacency matrix)
    - Normalized Laplacian: L_norm = D^-½ L D^-½
    - Graph Shift: Custom operator provided by user
    """

    @staticmethod
    def build(adjacency_matrix: torch.Tensor,  # Cayley graph adjacency matrix (|G| x |G|)
              smooth_operator: str = "laplacian",  # Type of smoothness operator
              graph_shift: torch.Tensor | None = None,  # Custom graph shift operator
              dtype: torch.dtype = torch.cfloat  # Output data type
              ) -> torch.Tensor:
        """Construct a graph smoothness operator from adjacency matrix.
        
        Args:
            adjacency_matrix: Cayley graph adjacency matrix of the group
            smooth_operator: Type of operator ("adjacency", "laplacian", "normalized_laplacian", "graph_shift")
            graph_shift: Custom graph shift operator (required for "graph_shift" mode)
            dtype: Output tensor data type
            
        Returns:
            Smoothness operator tensor in specified dtype
        """
        if smooth_operator == "adjacency":
            # Row-normalized adjacency matrix: A_norm[i,j] = A[i,j] / Σ_k A[i,k]
            # This creates a stochastic matrix where each row sums to 1
            # Used for random walk-based smoothness regularization
            smoother = adjacency_matrix / torch.sum(adjacency_matrix, dim=1, keepdim=True)
        elif smooth_operator == "laplacian":
            # Standard graph Laplacian: L = D - A
            # where D[i,i] = Σ_j A[i,j] (degree matrix)
            # L captures local smoothness: (Lf)[i] = Σ_j A[i,j](f[i] - f[j])
            degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
            smoother = degree_matrix - adjacency_matrix
        elif smooth_operator == "normalized_laplacian":
            # Normalized Laplacian: L_norm = D^-½ @ L @ D^-½
            # This normalizes the Laplacian by node degrees, making it scale-invariant
            # Eigenvalues are bounded in [0, 2] for connected graphs
            degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
            smoother = degree_matrix - adjacency_matrix
            # Compute D^-½ with numerical stability
            degree_matrix_power = torch.sqrt(1.0 / degree_matrix)
            degree_matrix_power[degree_matrix_power == float("inf")] = 0
            smoother = degree_matrix_power @ smoother @ degree_matrix_power
        elif smooth_operator == "graph_shift" and graph_shift is not None:
            # Custom graph shift operator provided by user
            # This allows for specialized spectral operators beyond standard choices
            smoother = torch.tensor(graph_shift)
        else:
            raise ValueError("Invalid smooth operator:", smooth_operator)

        return smoother.to(dtype)


class ReynoldsProjectorHelper:
    """Utilities for Reynolds operator and equivariant projection.

    This class provides utilities for constructing and applying Reynolds projectors
    that enforce equivariance constraints in group-equivariant operations.

    Mathematical Foundation:
    - Reynolds operator: R = (1/|G|) Σ_{g∈G} ρ(g) where ρ(g) is group representation
    - Equivariant projector: P = Q(QᵀQ)⁻¹Qᵀ where Q contains invariant eigenvectors
    - Projection: P̃ = P·vec(Φ) where Φ is the operator to be projected
    """

    @staticmethod
    def build_projector(raynold_op: np.ndarray | torch.Tensor,  # Reynolds operator matrix
                        dtype: torch.dtype  # Output data type
                        ) -> torch.Tensor:
        """Build equivariant projector from Reynolds operator.
        
        Mathematical Process:
        1. Eigendecomposition: R = QΛQ⁻¹
        2. Identify invariant subspace: Q[:,|λ-1|<ε]
        3. Construct projector: P = Q(QᵀQ)⁻¹Qᵀ
        
        Args:
            raynold_op: Reynolds operator matrix
            dtype: Output tensor data type
            
        Returns:
            Equivariant projector matrix
        """
        if raynold_op is None:
            raise ValueError("raynold_op cannot be None when building projector")
        
        # Convert to tensor and perform eigendecomposition
        R = torch.tensor(raynold_op).to(dtype)
        ev, evec = torch.linalg.eigh(R)
        
        # Extract eigenvectors with eigenvalues close to 1 (invariant subspace)
        # These correspond to group-invariant functions
        evec = evec[:, torch.abs(ev - 1) < 1e-3]
        
        # Construct orthogonal projector onto invariant subspace
        # P = Q(QᵀQ)⁻¹Qᵀ where Q contains invariant eigenvectors
        # Keep original numpy-based pseudo-inverse for numerical stability
        return evec @ np.linalg.inv(evec.T @ evec) @ evec.T

    @staticmethod
    def project(projector: torch.Tensor,  # Equivariant projector matrix
                operator: torch.Tensor  # Operator to be projected
                ) -> torch.Tensor:
        """Apply equivariant projection to an operator.
        
        Mathematical Operation:
        P̃ = P·vec(Φ) where P is the Reynolds projector and Φ is the operator
        
        Args:
            projector: Equivariant projector matrix
            operator: Operator tensor to be projected
            
        Returns:
            Projected operator with same shape as input
        """
        # Flatten operator, apply projection, then reshape back
        # This enforces equivariance constraint: T(g)·Φ = Φ·ρ_H(g)
        return (projector @ operator.flatten()).reshape(operator.shape)


class L1ProjectorUtils:
    """Utilities to compute L1 projector from a learned mapping M.

    This class provides utilities for constructing L1 projectors that project
    signals to the invariant subspace of a learned spectral mapping matrix M.

    Mathematical Foundation:
    - M̄ = M(MᵀM)⁻¹Mᵀ (projection matrix onto range of M)
    - Eigendecomposition: M̄ = VΣV⁻¹
    - L1 projector: L1 = V(VᵀV)⁻¹Vᵀ where V spans range(M)
    """

    @staticmethod
    def compute_from_M(M: torch.Tensor,  # Learned spectral mapping matrix
                       dtype: torch.dtype  # Output data type
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute L1 projector and invariant eigenvectors from mapping matrix M.
        
        Mathematical Process:
        1. Compute M̄ = M(MᵀM)⁻¹Mᵀ (projection onto range of M)
        2. Eigendecomposition: M̄ = VΣV⁻¹
        3. Extract invariant subspace: V[:,|σ-1|<ε]
        4. Construct L1 projector: L1 = V(VᵀV)⁻¹Vᵀ
        
        Args:
            M: Learned spectral mapping matrix
            dtype: Output tensor data type
            
        Returns:
            Tuple of (invariant_eigenvectors, L1_projector)
        """
        # Compute projection matrix onto range space of M
        # M̄ = M(MᵀM)⁻¹Mᵀ is the orthogonal projector onto col(M)
        M_bar = M @ torch.linalg.inv(M.T @ M) @ M.T
        
        # Eigendecomposition of the projection matrix
        # M̄ has eigenvalues 1 for vectors in range(M) and 0 for vectors in null(Mᵀ)
        eigvals, eigvecs = torch.linalg.eig(M_bar)
        
        # Extract eigenvectors with eigenvalues close to 1 (invariant subspace)
        # These correspond to the range space of the mapping matrix M
        eigvecs = eigvecs[:, torch.abs(eigvals - 1) < 1e-7]
        
        # Sparsify eigenvectors by zeroing out small coefficients
        eigvecs[torch.abs(eigvecs) < 1e-6] = 0

        # Compute L1 projector using numpy for numerical stability
        # This preserves the exact computation from the original implementation
        M_np = M.detach().cpu().numpy()
        L1_np = L1ProjectorUtils._l1_projector_numpy(M_np)
        L1 = torch.tensor(L1_np).to(dtype)
        
        return eigvecs, L1

    @staticmethod
    def _l1_projector_numpy(M: np.ndarray) -> np.ndarray:
        """Compute L1 projector using numpy for numerical stability.
        
        Mathematical Steps:
        1. Compute M̄ = M(MᵀM)⁻¹Mᵀ
        2. Eigendecomposition: M̄ = VΣV⁻¹
        3. Select invariant subspace: V[:,|σ-1|<ε]
        4. Form L1 projector: V(VᵀV)⁻¹Vᵀ
        
        Args:
            M: Mapping matrix as numpy array
            
        Returns:
            L1 projector as numpy array
        """
        # Compute projection matrix onto range space of M
        M_bar = M @ np.linalg.inv(M.T @ M) @ M.T
        
        # Eigendecomposition of the projection matrix
        eigvals, eigvecs = np.linalg.eig(M_bar)
        
        # Select eigenvectors corresponding to eigenvalue 1 (invariant subspace)
        eigvecs = eigvecs[:, np.abs(eigvals - 1) < 1e-7]
        
        # Construct L1 projector: orthogonal projection onto invariant subspace
        L1 = eigvecs @ np.linalg.pinv(eigvecs)
        return L1


class FourierOps:
    """Fourier transform helpers for group-equivariant operations.

    This class provides centralized Fourier transform operations for converting
    between spatial and spectral domains in group-equivariant neural networks.

    Mathematical Foundation:
    - Forward Transform: X̂ = Bᵀx (real case) or X̂ = B̄ᵀx (complex case)
    - Inverse Transform: x = BX̂
    - Where B is the group's Fourier basis matrix
    """

    @staticmethod
    def forward(x: torch.Tensor,  # Input tensor (spatial domain)
                basis: torch.Tensor,  # Fourier basis matrix
                dtype: torch.dtype  # Data type for complex operations
                ) -> torch.Tensor:
        """Forward Fourier transform: spatial → spectral domain.
        
        Mathematical Operation:
        - Real case: X̂ = Bᵀx
        - Complex case: X̂ = B̄ᵀx (conjugate transpose)
        
        Args:
            x: Input tensor in spatial domain
            basis: Fourier basis matrix
            dtype: Data type for complex operations
            
        Returns:
            Tensor in spectral domain
        """
        # Construct the Fourier transform matrix B
        if dtype in [torch.cfloat, torch.cdouble]:
            # Complex case: use conjugate transpose for unitary transform
            B = torch.transpose(torch.conj(basis), 0, 1)
        elif dtype in [torch.float, torch.float64]:
            # Real case: use transpose for orthogonal transform
            B = torch.transpose(basis, 0, 1)
        else:
            raise ValueError("Invalid dtype:", dtype)

        # Apply Fourier transform based on tensor dimensions
        if len(x.shape) == 1:
            # 1D case: simple matrix-vector multiplication
            # X̂ = B @ x where x is a group signal
            return torch.matmul(B, x.to(basis.dtype))
        elif len(x.shape) == 5:
            # 5D case: (batch, channel, group_size, height, width)
            # Apply Fourier transform: X̂[fc] = Σ_g B[fg] * x[g]
            return torch.einsum("fg,bcghw->bcfhw", B, x.to(basis.dtype))
        elif len(x.shape) == 6:
            # 6D case: (batch, channel, group_size, depth, height, width)
            # 3D spatial data with group dimension
            return torch.einsum("fg,bcghwd->bcfhwd", B, x.to(basis.dtype))
        else:
            raise ValueError("Invalid shape:", x.shape)

    @staticmethod
    def inverse(x: torch.Tensor,  # Input tensor (spectral domain)
                basis: torch.Tensor  # Fourier basis matrix
                ) -> torch.Tensor:
        """Inverse Fourier transform: spectral → spatial domain.
        
        Mathematical Operation:
        x = BX̂ where B is the Fourier basis matrix
        
        Args:
            x: Input tensor in spectral domain
            basis: Fourier basis matrix
            
        Returns:
            Tensor in spatial domain
        """
        # Apply inverse Fourier transform based on tensor dimensions
        if len(x.shape) == 1:
            # 1D case: simple matrix-vector multiplication
            # x = B @ X̂ where X̂ is spectral coefficients
            return torch.matmul(basis, x)
        elif len(x.shape) == 5:
            # 5D case: (batch, channel, spectral_coeffs, height, width)
            # Inverse transform: x[g] = Σ_f basis[fg] * X̂[f]
            return torch.einsum("fg,bcghw->bcfhw", basis, x.to(basis.dtype))
        elif len(x.shape) == 6:
            # 6D case: (batch, channel, spectral_coeffs, depth, height, width)
            # 3D spatial data reconstruction from spectral coefficients
            return torch.einsum("fg,bcghwd->bcfhwd", basis, x.to(basis.dtype))
        else:
            raise ValueError("Invalid shape:", x.shape)

