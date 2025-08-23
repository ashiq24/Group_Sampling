"""
Test helper utilities for Group_Sampling test suite.

This module provides:
- Tensor layout helpers for reshaping between different group-structured formats
- Small graph builders for unit tests
- Mock objects and simplified versions of components for isolated testing
"""

import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass


# ============================================================================
# Tensor Layout Helpers
# ============================================================================

class TensorLayoutHelper:
    """Utilities for managing tensor layouts in group-equivariant networks."""
    
    @staticmethod
    def flatten_group_channels(x: torch.Tensor, group_size: int) -> torch.Tensor:
        """
        Convert from (B, C, |G|, H, W) to (B, C*|G|, H, W) layout.
        Convert from (B, C, |G|, H, W, D) to (B, C*|G|, H, W, D) layout.
        
        Args:
            x: Input tensor of shape (B, C, |G|, H, W) or (B, C, |G|, H, W, D)
            group_size: Size of the group |G|
            
        Returns:
            Tensor of shape (B, C*|G|, H, W) or (B, C*|G|, H, W, D)
        """
        if x.dim() == 5:  # (B, C, |G|, H, W)
            B, C, G, H, W = x.shape
            assert G == group_size, f"Expected group dimension {group_size}, got {G}"
            return x.reshape(B, C * G, H, W)
        elif x.dim() == 6:  # (B, C, |G|, H, W, D)
            B, C, G, H, W, D = x.shape
            assert G == group_size, f"Expected group dimension {group_size}, got {G}"
            return x.reshape(B, C * G, H, W, D)
        elif x.dim() == 3:  # (B, C, |G|) - 1D case
            B, C, G = x.shape
            assert G == group_size, f"Expected group dimension {group_size}, got {G}"
            return x.reshape(B, C * G)
        elif x.dim() == 2:  # (C, |G|) - single sample
            C, G = x.shape
            assert G == group_size, f"Expected group dimension {group_size}, got {G}"
            return x.reshape(C * G)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    
    @staticmethod
    def unflatten_group_channels(x: torch.Tensor, group_size: int, num_channels: int) -> torch.Tensor:
        """
        Convert from (B, C*|G|, H, W) to (B, C, |G|, H, W) layout.
        Convert from (B, C*|G|, H, W, D) to (B, C, |G|, H, W, D) layout.
        
        Args:
            x: Input tensor of shape (B, C*|G|, H, W) or (B, C*|G|, H, W, D)
            group_size: Size of the group |G|
            num_channels: Number of feature channels C
            
        Returns:
            Tensor of shape (B, C, |G|, H, W) or (B, C, |G|, H, W, D)
        """
        if x.dim() == 4:  # (B, C*|G|, H, W)
            B, CG, H, W = x.shape
            assert CG == num_channels * group_size, f"Expected {num_channels * group_size} channels, got {CG}"
            return x.reshape(B, num_channels, group_size, H, W)
        elif x.dim() == 5:  # (B, C*|G|, H, W, D)
            B, CG, H, W, D = x.shape
            assert CG == num_channels * group_size, f"Expected {num_channels * group_size} channels, got {CG}"
            return x.reshape(B, num_channels, group_size, H, W, D)
        elif x.dim() == 2:  # (B, C*|G|) - 1D case
            B, CG = x.shape
            assert CG == num_channels * group_size, f"Expected {num_channels * group_size} channels, got {CG}"
            return x.reshape(B, num_channels, group_size)
        elif x.dim() == 1:  # (C*|G|) - single sample
            CG, = x.shape
            assert CG == num_channels * group_size, f"Expected {num_channels * group_size} channels, got {CG}"
            return x.reshape(num_channels, group_size)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    
    @staticmethod
    def validate_group_tensor_shape(
        x: torch.Tensor, 
        expected_batch_size: int,
        expected_channels: int, 
        expected_group_size: int,
        expected_spatial_shape: Tuple[int, ...],
        layout: str = 'flat'
    ):
        """
        Validate tensor has expected shape for group-structured data.
        
        Args:
            x: Tensor to validate
            expected_batch_size: Expected batch dimension
            expected_channels: Expected number of feature channels
            expected_group_size: Expected group size
            expected_spatial_shape: Expected spatial dimensions (H, W) or (H,)
            layout: 'flat' for (B, C*|G|, H, W) or 'unflat' for (B, C, |G|, H, W)
        """
        if layout == 'flat':
            expected_shape = (expected_batch_size, expected_channels * expected_group_size, *expected_spatial_shape)
        elif layout == 'unflat':
            expected_shape = (expected_batch_size, expected_channels, expected_group_size, *expected_spatial_shape)
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        assert x.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {x.shape} for layout '{layout}'"
        )
    
    @staticmethod
    def extract_fiber(x: torch.Tensor, group_size: int, fiber_idx: int = 0) -> torch.Tensor:
        """
        Extract a single fiber (group orbit) from group-structured tensor.
        
        Args:
            x: Group-structured tensor (B, C*|G|, H, W) or (B, C, |G|, H, W) or (B, C*|G|, H, W, D) or (B, C, |G|, H, W, D)
            group_size: Size of the group
            fiber_idx: Index of fiber to extract (0 to C-1)
            
        Returns:
            Fiber tensor of shape (B, |G|, H, W) or (B, |G|) or (B, |G|, H, W, D)
        """
        if x.dim() == 4:  # (B, C*|G|, H, W)
            B, CG, H, W = x.shape
            num_channels = CG // group_size
            x_unflat = x.reshape(B, num_channels, group_size, H, W)
            return x_unflat[:, fiber_idx]  # (B, |G|, H, W)
        elif x.dim() == 5:  # (B, C*|G|, H, W, D) or (B, C, |G|, H, W)
            if x.shape[1] == group_size:  # (B, C, |G|, H, W)
                return x[:, fiber_idx]  # (B, |G|, H, W)
            else:  # (B, C*|G|, H, W, D)
                B, CG, H, W, D = x.shape
                num_channels = CG // group_size
                x_unflat = x.reshape(B, num_channels, group_size, H, W, D)
                return x_unflat[:, fiber_idx]  # (B, |G|, H, W, D)
        elif x.dim() == 6:  # (B, C, |G|, H, W, D)
            return x[:, fiber_idx]  # (B, |G|, H, W, D)
        elif x.dim() == 2:  # (B, C*|G|)
            B, CG = x.shape
            num_channels = CG // group_size
            x_unflat = x.reshape(B, num_channels, group_size)
            return x_unflat[:, fiber_idx]  # (B, |G|)
        elif x.dim() == 3:  # (B, C, |G|)
            return x[:, fiber_idx]  # (B, |G|)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")


# ============================================================================
# Small Graph Builders for Testing
# ============================================================================

@dataclass
class MockGroupSpec:
    """Simple group specification for testing."""
    group_type: str
    order: int
    generator: Optional[str] = None


@dataclass 
class MockGraphData:
    """Container for mock graph data."""
    nodes: List[int]
    edges: List[Tuple[int, int]]
    adjacency_matrix: torch.Tensor
    directed_adjacency_matrix: torch.Tensor
    fourier_basis: torch.Tensor
    reynolds_operator: Optional[torch.Tensor] = None


class MockGraphBuilder:
    """Builder for small test graphs."""
    
    @staticmethod
    def build_cycle_graph(n: int, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> MockGraphData:
        """
        Build a simple cycle graph C_n for testing.
        
        Args:
            n: Number of nodes in cycle
            dtype: Data type for matrices
            device: Device for tensors
            
        Returns:
            MockGraphData with graph structure and bases
        """
        nodes = list(range(n))
        
        # Cycle edges: i -> (i+1) % n
        edges = [(i, (i + 1) % n) for i in range(n)]
        
        # Adjacency matrix (undirected)
        adj = torch.zeros(n, n, dtype=dtype, device=device)
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1
        
        # Directed adjacency matrix
        dir_adj = torch.zeros(n, n, dtype=dtype, device=device)
        for i in range(n):
            dir_adj[i, (i + 1) % n] = 1
        
        # Simple DFT basis for cycle
        fourier_basis = torch.zeros(n, n, dtype=torch.cfloat, device=device)
        for k in range(n):
            for j in range(n):
                angle = -2 * torch.pi * k * j / n
                fourier_basis[j, k] = torch.exp(1j * torch.tensor(angle)) / torch.sqrt(torch.tensor(n, dtype=torch.float32))
        
        if dtype.is_floating_point:  # Convert to real if needed
            fourier_basis = fourier_basis.real.to(dtype)
        
        return MockGraphData(
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adj,
            directed_adjacency_matrix=dir_adj,
            fourier_basis=fourier_basis
        )
    
    @staticmethod
    def build_dihedral_graph(n: int, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> MockGraphData:
        """
        Build a simple dihedral graph D_n (2n elements) for testing.
        
        Args:
            n: Order of dihedral group (group has 2n elements)
            dtype: Data type for matrices
            device: Device for tensors
            
        Returns:
            MockGraphData with graph structure and bases
        """
        group_size = 2 * n
        nodes = list(range(group_size))
        
        # Simple connectivity: rotations form a cycle, reflections connect to adjacent rotations
        edges = []
        
        # Rotation edges (first n elements form cycle)
        for i in range(n):
            edges.append((i, (i + 1) % n))
        
        # Reflection edges (connect rotations to reflections)
        for i in range(n):
            edges.append((i, n + i))  # rotation i to reflection i
            edges.append((n + i, i))  # reflection i to rotation i
        
        # Adjacency matrix (undirected)
        adj = torch.zeros(group_size, group_size, dtype=dtype, device=device)
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Directed adjacency matrix
        dir_adj = torch.zeros(group_size, group_size, dtype=dtype, device=device)
        for i, j in edges:
            dir_adj[i, j] = 1
        
        # Simplified Fourier basis (identity for testing)
        fourier_basis = torch.eye(group_size, dtype=dtype, device=device)
        
        return MockGraphData(
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adj,
            directed_adjacency_matrix=dir_adj,
            fourier_basis=fourier_basis
        )
    
    @staticmethod
    def build_trivial_graph(n: int = 1, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> MockGraphData:
        """Build trivial graph with n disconnected nodes."""
        nodes = list(range(n))
        edges = []
        adj = torch.zeros(n, n, dtype=dtype, device=device)
        dir_adj = torch.zeros(n, n, dtype=dtype, device=device)
        fourier_basis = torch.eye(n, dtype=dtype, device=device)
        
        return MockGraphData(
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adj,
            directed_adjacency_matrix=dir_adj,
            fourier_basis=fourier_basis
        )


# ============================================================================
# Mock Components for Isolated Testing
# ============================================================================

class MockSamplingMatrix:
    """Mock sampling matrix for testing."""
    
    def __init__(self, parent_size: int, subgroup_size: int, subsample_indices: List[int]):
        self.parent_size = parent_size
        self.subgroup_size = subgroup_size
        self.subsample_indices = subsample_indices
        
        assert len(subsample_indices) == subgroup_size
        assert all(0 <= idx < parent_size for idx in subsample_indices)
    
    def to_tensor(self, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> torch.Tensor:
        """Convert to tensor representation."""
        S = torch.zeros(self.subgroup_size, self.parent_size, dtype=dtype, device=device)
        for i, idx in enumerate(self.subsample_indices):
            S[i, idx] = 1
        return S
    
    def pseudoinverse(self, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> torch.Tensor:
        """Compute pseudoinverse for upsampling."""
        S = self.to_tensor(dtype, device)
        return torch.pinverse(S)


class MockSmoothnessOperator:
    """Mock smoothness operator for testing."""
    
    def __init__(self, size: int, operator_type: str = 'identity'):
        self.size = size
        self.operator_type = operator_type
    
    def to_tensor(self, dtype: torch.dtype = torch.float32, device: str = 'cpu') -> torch.Tensor:
        """Convert to tensor representation."""
        if self.operator_type == 'identity':
            return torch.eye(self.size, dtype=dtype, device=device)
        elif self.operator_type == 'laplacian':
            # Simple discrete Laplacian (tridiagonal)
            L = torch.zeros(self.size, self.size, dtype=dtype, device=device)
            for i in range(self.size):
                L[i, i] = 2
                if i > 0:
                    L[i, i-1] = -1
                if i < self.size - 1:
                    L[i, i+1] = -1
            return L
        elif self.operator_type == 'adjacency':
            # Simple adjacency (cycle)
            A = torch.zeros(self.size, self.size, dtype=dtype, device=device)
            for i in range(self.size):
                A[i, (i + 1) % self.size] = 1
                A[(i + 1) % self.size, i] = 1
            return A
        else:
            raise ValueError(f"Unknown operator type: {self.operator_type}")


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_test_subsampling_indices(parent_size: int, factor: int) -> List[int]:
    """Generate simple stride-based subsampling indices."""
    return list(range(0, parent_size, factor))


def generate_test_perfect_reconstruction_data(
    parent_size: int,
    subgroup_size: int,
    dtype: torch.dtype = torch.cfloat,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test data that satisfies perfect reconstruction constraint.
    
    Returns:
        Tuple of (parent_basis, subgroup_basis, sampling_matrix)
    """
    # Simple bases (DFT-like)
    parent_basis = torch.fft.fft(torch.eye(parent_size, dtype=dtype, device=device), dim=0)
    parent_basis = parent_basis / torch.sqrt(torch.tensor(parent_size, dtype=torch.float32))
    
    # Subgroup basis (subset of parent)
    indices = generate_test_subsampling_indices(parent_size, parent_size // subgroup_size)[:subgroup_size]
    subgroup_basis = torch.fft.fft(torch.eye(subgroup_size, dtype=dtype, device=device), dim=0)
    subgroup_basis = subgroup_basis / torch.sqrt(torch.tensor(subgroup_size, dtype=torch.float32))
    
    # Sampling matrix
    S = torch.zeros(subgroup_size, parent_size, dtype=dtype, device=device)
    for i, idx in enumerate(indices):
        S[i, idx] = 1
    
    return parent_basis, subgroup_basis, S


def generate_bandlimited_signal(
    signal_length: int,
    cutoff_freq: int,
    dtype: torch.dtype = torch.cfloat,
    device: str = 'cpu',
    seed: int = None
) -> torch.Tensor:
    """Generate a bandlimited test signal."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random frequencies up to cutoff
    signal = torch.zeros(signal_length, dtype=dtype, device=device)
    for k in range(min(cutoff_freq, signal_length)):
        signal[k] = torch.randn(1, dtype=dtype, device=device)
    
    # Convert to time domain
    time_signal = torch.fft.ifft(signal)
    
    if not dtype.is_complex:
        time_signal = time_signal.real.to(dtype)
    
    return time_signal


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_sampling_matrix(S: torch.Tensor, parent_size: int, subgroup_size: int):
    """Validate properties of sampling matrix."""
    assert S.shape == (subgroup_size, parent_size), f"Expected shape ({subgroup_size}, {parent_size}), got {S.shape}"
    
    # Each row should have exactly one 1 and rest 0s
    row_sums = torch.sum(S, dim=1)
    assert torch.allclose(row_sums, torch.ones(subgroup_size, dtype=S.dtype, device=S.device)), \
        "Each row should sum to 1"
    
    # Each column should have at most one 1
    col_sums = torch.sum(S, dim=0)
    assert torch.all(col_sums <= 1), "Each column should have at most one 1"


def validate_fourier_basis(basis: torch.Tensor, tolerance: float = 1e-6):
    """Validate that basis is unitary/orthonormal."""
    n = basis.shape[0]
    identity = torch.eye(n, dtype=basis.dtype, device=basis.device)
    
    # Check if basis is unitary: F @ F.H = I
    product = basis @ basis.conj().transpose(-2, -1)
    assert torch.allclose(product, identity, atol=tolerance), \
        "Basis is not unitary"


def check_perfect_reconstruction_constraint(
    parent_basis: torch.Tensor,
    subgroup_basis: torch.Tensor, 
    sampling_matrix: torch.Tensor,
    M: torch.Tensor,
    tolerance: float = 1e-4
):
    """Check if perfect reconstruction constraint is satisfied: F_H^{-1} ≈ S @ F_G^{-1} @ M"""
    lhs = torch.inverse(subgroup_basis)
    rhs = sampling_matrix @ torch.inverse(parent_basis) @ M
    
    assert torch.allclose(lhs, rhs, atol=tolerance), \
        f"Perfect reconstruction constraint not satisfied. Max error: {torch.max(torch.abs(lhs - rhs))}"


# ============================================================================
# 4D Tensor Support for Fourier Operations (B, C, D, H, W)
# ============================================================================

def handle_4d_tensor_fourier(x: torch.Tensor, operation: str = 'fft') -> torch.Tensor:
    """
    Handle Fourier operations on 4D tensors (B, C, D, H, W).
    
    Args:
        x: Input tensor of shape (B, C, D, H, W)
        operation: Fourier operation ('fft', 'ifft', 'fft_shift', 'ifft_shift')
        
    Returns:
        Result tensor of same shape
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {x.dim()}D")
    
    if operation == 'fft':
        # Perform 3D FFT on spatial dimensions (D, H, W)
        return torch.fft.fftn(x, dim=(-3, -2, -1))
    elif operation == 'ifft':
        # Perform 3D IFFT on spatial dimensions (D, H, W)
        return torch.fft.ifftn(x, dim=(-3, -2, -1))
    elif operation == 'fft_shift':
        # Perform 3D FFT shift on spatial dimensions
        x_shifted = torch.fft.fftshift(x, dim=-3)
        x_shifted = torch.fft.fftshift(x_shifted, dim=-2)
        x_shifted = torch.fft.fftshift(x_shifted, dim=-1)
        return x_shifted
    elif operation == 'ifft_shift':
        # Perform 3D IFFT shift on spatial dimensions
        x_shifted = torch.fft.ifftshift(x, dim=-3)
        x_shifted = torch.fft.ifftshift(x_shifted, dim=-2)
        x_shifted = torch.fft.ifftshift(x_shifted, dim=-1)
        return x_shifted
    else:
        raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# Utility functions for testing equivariance
# ============================================================================

def apply_group_action(
    x: torch.Tensor, 
    group_element: int, 
    group_type: str, 
    group_size: int
) -> torch.Tensor:
    """
    Apply group action to a signal for testing equivariance.
    
    Args:
        x: Input signal (B, C*|G|, H, W) or similar
        group_element: Group element to apply (integer index)
        group_type: Type of group ('cycle' or 'dihedral')
        group_size: Size of the group
        
    Returns:
        Transformed signal
    """
    if group_type == 'cycle':
        return apply_cyclic_action(x, group_element, group_size)
    elif group_type == 'dihedral':
        return apply_dihedral_action(x, group_element, group_size)
    else:
        raise ValueError(f"Unknown group type: {group_type}")


def apply_cyclic_action(x: torch.Tensor, rotation: int, group_size: int) -> torch.Tensor:
    """Apply cyclic group action (rotation)."""
    if x.dim() == 4:  # (B, C*|G|, H, W)
        B, CG, H, W = x.shape
        C = CG // group_size
        x_grouped = x.reshape(B, C, group_size, H, W)
        
        # Rotate along group dimension
        x_rotated = torch.roll(x_grouped, rotation, dims=2)
        return x_rotated.reshape(B, CG, H, W)
    
    elif x.dim() == 2:  # (B, C*|G|)
        B, CG = x.shape
        C = CG // group_size
        x_grouped = x.reshape(B, C, group_size)
        
        # Rotate along group dimension
        x_rotated = torch.roll(x_grouped, rotation, dims=2)
        return x_rotated.reshape(B, CG)
    
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}")


def apply_dihedral_action(x: torch.Tensor, element: int, group_size: int) -> torch.Tensor:
    """Apply dihedral group action (rotation + possible reflection)."""
    n = group_size // 2  # D_n has 2n elements
    
    if element < n:
        # Pure rotation
        return apply_cyclic_action(x, element, n)
    else:
        # Reflection followed by rotation
        reflection_element = element - n
        
        # Apply reflection (flip spatial dimensions)
        if x.dim() == 4:  # (B, C*|G|, H, W)
            x_reflected = torch.flip(x, dims=[3])  # Flip W dimension
        else:
            x_reflected = x  # No spatial flip for lower dimensions
        
        # Then apply rotation
        return apply_cyclic_action(x_reflected, reflection_element, n)
