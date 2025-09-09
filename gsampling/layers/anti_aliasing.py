"""
Equivariant Anti-Aliasing Layer for Group-Structured Data

This module implements spectral anti-aliasing for group-equivariant neural networks using
non-commutative spectral methods. It provides bandlimited reconstruction and upsampling
capabilities while preserving group equivariance properties.

Mathematical Foundation:
------------------------
The anti-aliasing layer learns a spectral mapping matrix M that satisfies:

1. Anti-aliasing constraint: ||S·F_G·M - F_H||² minimized
   Where S is the subsampling matrix, F_G is the original group's Fourier basis,
   and F_H is the subgroup's Fourier basis.

2. Smoothness regularization: tr(Mᵀ·F_Gᵀ·L·F_G·M)
   Where L is the graph Laplacian or other spectral operator.

3. Equivariance constraint: T(g)·M = M·ρ_H(g) ∀g ∈ G
   Where T(g) is the group action on the original space and ρ_H(g) is the
   representation of g in the subgroup.

Key Features:
- Spectral bandlimiting through L1 projection
- Equivariant upsampling with energy preservation
- Support for various graph operators (Laplacian, adjacency, etc.)
- Reynolds operator for equivariance enforcement
- GPU-accelerated optimization for large groups

Author: Group Sampling Team
"""

import torch
import numpy as np
from einops import rearrange


class AntiAliasingLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        nodes: list,  # Original group G elements (length |G|)
        adjaceny_matrix: np.ndarray,  # Cayley graph adjacency matrix (|G| x |G|)
        basis: np.ndarray,  # Fourier basis for G from irreps (|G| x dim(Fourier_space))
        subsample_nodes: list,  # Subgroup H ⊂ G elements (length |H|)
        subsample_adjacency_matrix: np.ndarray,  # H's Cayley graph adjacency (|H| x |H|)
        sub_basis: np.ndarray,  # Fourier basis for H from irreps (|H| x dim(Sub_Fourier_space))
        smooth_operator: str = "laplacian",  # Spectral operator type for smoothness
        smoothness_loss_weight=1.0,  # Weight for spectral smoothness regularization
        iterations=500,  # Optimization iterations for learning M matrix
        device="cuda:0",  # Compute device for optimization
        dtype=torch.cfloat,  # Data type for complex irreps
        graph_shift: np.ndarray = None,  # Alternative graph shift operator (|G| x |G|)
        raynold_op: np.ndarray = None,  # Reynolds operator for equivariance projection
        equi_constraint: bool = True,  # Enforce T(g)·M = M·ρ_H(g) constraint
        equi_correction: bool = True,  # Project learned operator to equivariant subspace
        mode: str = "linear_optim",  # Matrix learning method
        threshold=1e-3  # Sparsification threshold for projection matrix
    ):
        """Equivariant anti-aliasing layer for group-structured data using non-commutative spectral methods.


        Parameters
        ----------
        nodes : list
            Elements of the original group G (length |G|)
        adjacency_matrix : np.ndarray
            Adjacency matrix of G's Cayley graph (|G| x |G| numpy array)
        basis : np.ndarray
            Fourier basis for G formed by direct sum of irreps (|G| x dim(Fourier_space) numpy array)
        subsample_nodes : list
            Elements of subgroup H ⊂ G (length |H|)
        subsample_adjacency_matrix : np.ndarray
            Adjacency matrix of H's Cayley graph (|H| x |H| numpy array)
        sub_basis : np.ndarray
            Fourier basis for H from its irreps (|H| x dim(Sub_Fourier_space) numpy array)
        smooth_operator : str, optional
            Spectral operator type for smoothness regularization, by default "laplacian"
            Options:
            - "laplacian": Group Laplacian Δ = D - A (degree matrix - adjacency)
            - "normalized_laplacian": Δ_norm = D^-½ΔD^-½
            - "adjacency": Plain Cayley graph adjacency
            - "graph_shift": Custom shift operator
        smoothness_loss_weight : float, optional
            Weight for spectral smoothness term, by default 0.01
        iterations : int, optional
            Iterations for the optimization, by default 50000
        device : str, optional
            Compute device, by default "cuda:0"
        dtype : torch.dtype, optional
            Data type for complex irreps, by default torch.cfloat
        graph_shift : np.ndarray, optional
            Alternative graph shift operator for G (|G| x |G|), by default None
        raynold_op : np.ndarray, optional
            Reynolds operator for G's action on Fourier coefficients, by default None
        equi_constraint : bool, optional
            Enforce T(g)·M = M·ρ_H(g) ∀g ∈ G (equivariance constraint), by default True
        equi_correction : bool, optional
            Project learned operator to equivariant subspace, by default True
        mode : str, optional
            Matrix learning method, by default "linear_optim"
            Options: "linear_optim" (constrained), "analytical" (pseudoinverse)
        threshold : float, optional
            Sparsification threshold for projection matrix, by default 1e-3


        Mathematical Foundation
        -----------------------
        Learns M  satisfying:
        1. Anti-aliasing: ||S·F_G·M - F_H||² minimized
        2. Smoothness:
        3. Equivariance:

        Where:
        - F_G = G's Fourier transform matrix (irreps direct sum)
        - F_H = H's Fourier transform
        - S: Subsampling matrix
        """

        super().__init__()

        # Store group structure information
        self.nodes = nodes  # Original group G elements
        self.subsample_nodes = subsample_nodes  # Subgroup H elements
        self.device = device  # Compute device for optimization
        self.dtype = dtype  # Data type for complex irreps
        self.iterations = iterations  # Optimization iterations
        self.mode = mode  # Matrix learning method
        self.smoothness_loss_weight = smoothness_loss_weight  # Smoothness regularization weight
        
        # Convert adjacency matrix to tensor for PyTorch operations
        # This represents the Cayley graph structure of the original group G
        self.adjacency_matrix = torch.tensor(adjaceny_matrix)
        
        # Equivariance constraint settings
        self.equi_constraint = equi_constraint  # Whether to enforce equivariance
        self.equi_raynold_op = None  # Reynolds operator (initialized later)
        self.threshold = threshold  # Sparsification threshold for projection matrix

        print(
            "Equi Constraint: ", equi_constraint, "Equi Correction: ", equi_correction
        )

        # Initialize spectral smoothness operator (Laplacian, adjacency, etc.)
        # This operator L is used in the smoothness regularization term: tr(Mᵀ·F_Gᵀ·L·F_G·M)
        self._init_smooth_selection_operator(
            graph_shift=graph_shift, smooth_operator=smooth_operator, dtype=dtype
        )
        
        # Register Fourier bases as buffers (persistent tensors)
        # F_G: Original group's Fourier basis (|G| x dim(Fourier_space))
        self.register_buffer("basis", torch.tensor(basis))
        # F_H: Subgroup's Fourier basis (|H| x dim(Sub_Fourier_space))
        self.register_buffer("sub_basis", torch.tensor(sub_basis))

        # Initialize Reynolds operator for equivariance projection
        # This operator projects learned matrices to the equivariant subspace
        self._init_raynold_op(raynold_op=raynold_op, dtype=dtype)

        # Convert subgroup adjacency matrix to tensor
        self.subsample_adjacency_matrix = torch.tensor(subsample_adjacency_matrix)
        
        # Construct subsampling matrix S: maps from original group to subgroup
        # S[i,j] = 1 if subsample_nodes[i] == nodes[j], 0 otherwise
        # This matrix implements the group homomorphism G → H
        self.sampling_matrix = torch.zeros(len(self.subsample_nodes), len(self.nodes))
        for i, node in enumerate(self.subsample_nodes):
            self.sampling_matrix[i, node] = 1

        # Learn the spectral mapping matrix M through optimization
        # M satisfies: ||S·F_G·M - F_H||² + α·tr(Mᵀ·F_Gᵀ·L·F_G·M) minimized
        self._init_anti_aliasing_operator()

        # Construct upsampling basis for spectral reconstruction
        # Formula: Φ_upsample = (Φ_G √|G| / √|H|) @ M
        # This preserves energy during upsampling: ||x_upsampled||² = ||x_sub||²
        self.register_buffer(
            "up_sampling_basis",
            ((self.basis.to(self.M.dtype) * self.basis.shape[0] ** 0.5) @ self.M
            / self.sub_basis.shape[0] ** 0.5).T,
        )

        # Apply equivariance correction to L1 projector if requested
        # This ensures the projector respects group symmetries
        if equi_correction:
            self.L1_projector = self.equi_correction(self.L1_projector)

        # Convert all tensors to the specified dtype for consistency
        self.basis = self.basis.to(self.dtype)
        self.sub_basis = self.sub_basis.to(self.dtype)
        self.sampling_matrix = self.sampling_matrix.to(self.dtype)

    def _init_smooth_selection_operator(
        self, graph_shift=None, smooth_operator="laplacian", dtype=torch.cfloat
    ):
        """Constructs graph frequency operator through spectral decomposition of graph structure.

        Mathematical Operations:
        - For adjacency: A = A / diag(sum(A)) (Row-normalized adjacency)
        - Laplacian: L = D - A where D = diag(sum(A))
        - Normalized Laplacian: L_norm = D^-½ @ L @ D^-½
        - Stores operator as L for subsequent frequency analysis
        """
        if smooth_operator == "adjacency":
            # Row-normalized adjacency matrix: A_norm[i,j] = A[i,j] / Σ_k A[i,k]
            # This creates a stochastic matrix where each row sums to 1
            # Used for random walk-based smoothness
            smoother = self.adjacency_matrix / torch.sum(
                self.adjacency_matrix, dim=1, keepdim=True
            )
        elif smooth_operator == "laplacian":
            # Standard graph Laplacian: L = D - A
            # where D[i,i] = Σ_j A[i,j] (degree matrix)
            # L captures local smoothness: (Lf)[i] = Σ_j A[i,j](f[i] - f[j])
            degree_matrix = torch.diag(torch.sum(self.adjacency_matrix, dim=1))
            smoother = degree_matrix - self.adjacency_matrix
        elif smooth_operator == "normalized_laplacian":
            # Normalized Laplacian: L_norm = D^-½ @ L @ D^-½
            # This normalizes the Laplacian by node degrees, making it scale-invariant
            # Eigenvalues are bounded in [0, 2] for connected graphs
            degree_matrix = torch.diag(torch.sum(self.adjacency_matrix, dim=1))
            smoother = degree_matrix - self.adjacency_matrix
            # Compute D^-½ with numerical stability
            degree_matrix_power = torch.sqrt(1.0 / degree_matrix)
            degree_matrix_power[degree_matrix_power == float("inf")] = 0
            smoother = degree_matrix_power @ smoother @ degree_matrix_power
        elif smooth_operator == "graph_shift" and graph_shift is not None:
            # Custom graph shift operator provided by user
            # This allows for specialized spectral operators beyond standard choices
            smoother = torch.tensor(graph_shift)
        else:
            raise ValueError("Invalid smooth operator: ", smooth_operator)

        # Register the smoothness operator as a buffer for persistence
        # This operator L is used in the smoothness term: tr(Mᵀ·F_Gᵀ·L·F_G·M)
        self.register_buffer("smoother", smoother.to(dtype))

    def _init_raynold_op(self, raynold_op, dtype):
        """Computes equivariance projection operator using Raynold's Reynolds operator.

        Mathematical Process:
        1. Eigendecomposition: R = QΛQ⁻¹
        2. Identify invariant subspace: Q[:,|λ-1|<ε]
        3. Construct projector: P = Q(QᵀQ)⁻¹Qᵀ
        Where R is the Reynolds operator input
        """
        if raynold_op is not None:
            # Convert Reynolds operator to tensor
            # The Reynolds operator R averages group actions: R = (1/|G|) Σ_{g∈G} ρ(g)
            # where ρ(g) is the representation of group element g
            raynold_tensor = torch.tensor(raynold_op).to(dtype)
            
            # Ensure we have a 2D matrix for eigendecomposition
            if raynold_tensor.ndim == 1:
                # Reshape 1D array to square matrix (assuming it's a flattened square matrix)
                size = int(np.sqrt(raynold_tensor.shape[0]))
                raynold_tensor = raynold_tensor.reshape(size, size)
            elif raynold_tensor.ndim == 0:
                # Handle scalar case (fallback to identity for trivial groups)
                raynold_tensor = torch.eye(1, dtype=dtype)
            elif raynold_tensor.ndim > 2:
                # Flatten higher dimensional arrays to 2D
                raynold_tensor = raynold_tensor.reshape(-1)
                size = int(np.sqrt(raynold_tensor.shape[0]))
                raynold_tensor = raynold_tensor.reshape(size, size)
            
            self.equi_raynold_op = raynold_tensor
            
            # Perform eigendecomposition if the matrix is large enough
            if self.equi_raynold_op.shape[0] >= 2:
                # Eigendecomposition: R = QΛQ⁻¹
                # The Reynolds operator has eigenvalue 1 for invariant vectors
                ev, evec = torch.linalg.eigh(self.equi_raynold_op)
                
                # Identify invariant subspace: eigenvectors with eigenvalues close to 1
                # These correspond to group-invariant functions
                evec = evec[:, torch.abs(ev - 1) < 1e-3]
                
                if evec.shape[1] > 0:
                    # Construct orthogonal projector onto invariant subspace
                    # P = Q(QᵀQ)⁻¹Qᵀ where Q contains invariant eigenvectors
                    self.register_buffer(
                        "equi_projector", evec @ np.linalg.inv(evec.T @ evec) @ evec.T
                    )
                else:
                    # No eigenvectors close to 1, use identity (no invariant subspace)
                    self.register_buffer(
                        "equi_projector", torch.eye(self.equi_raynold_op.shape[0], dtype=dtype)
                    )
            else:
                # For small matrices (1x1), use identity projector
                self.register_buffer(
                    "equi_projector", torch.eye(self.equi_raynold_op.shape[0], dtype=dtype)
                )

    def _init_anti_aliasing_operator(self):
        """Learns spectral mapping between original and subsampled graph Fourier bases.

        Core Computation:
        - Solves M = argminₘ ||S·FB·M - F̃||_2 + α·tr(MᵀFBᵀLFB·M) + |R . L1 - L1|_2 (see paper)
        Where:
        FB = original basis, F̃ = subsampled basis
        S = sampling matrix, L = smoothing operator
        Stores M and computes its invariant subspace projection
        """
        # Learn the spectral mapping matrix M through optimization
        # M minimizes: ||S·F_G·M - F_H||² + α·tr(Mᵀ·F_Gᵀ·L·F_G·M)
        # where S is subsampling matrix, F_G/F_H are Fourier bases, L is smoothness operator
        self.register_buffer(
            "M",
            self._calculate_M(
                iterations=self.iterations,
                device="cpu",
                smoothness_loss_weight=self.smoothness_loss_weight,
                mode=self.mode,
            ),
        )

        # Sparsify the learned matrix M by zeroing out small coefficients
        # This reduces computational complexity and improves numerical stability
        self.M[torch.abs(self.M) <= self.threshold] = 0
        
        # Compute M̄ = M(MᵀM)⁻¹Mᵀ (projection matrix onto range of M)
        # This matrix projects any vector onto the column space of M
        self.M_bar = self.M @ torch.linalg.inv(self.M.T @ self.M) @ self.M.T
        
        # Eigendecomposition of M̄ to find invariant subspace
        # M̄ has eigenvalues 1 for vectors in range(M) and 0 for vectors in null(Mᵀ)
        eigvals, eigvecs = torch.linalg.eig(self.M_bar)
        
        # Extract eigenvectors with eigenvalues close to 1 (invariant subspace)
        # These correspond to the range space of the mapping matrix M
        eigvecs = eigvecs[:, torch.abs(eigvals - 1) < 1e-7]
        
        # Sparsify eigenvectors by zeroing out small coefficients
        eigvecs[torch.abs(eigvecs) < 1e-6] = 0
        
        # Ensure M is in the correct dtype
        self.M.to(self.dtype)

        # Store invariant eigenvectors and L1 projector for spectral bandlimiting
        self.register_buffer("L1_eigs", eigvecs)
        self.register_buffer(
            "L1_projector",
            torch.tensor(self.l1_projector(self.M.numpy())).to(self.dtype),
        )

    def l1_projector(self, M):
        """
        Mathematical Steps:
        1. Compute: M̄ = M(MᵀM)⁻¹Mᵀ
        2. Eigendecomposition: M̄ = VΣV⁻¹
        3. Select invariant subspace: V[:,|σ-1|<ε]
        4. Form L1 projector: V(VᵀV)⁻¹Vᵀ
        Projects signals to M's range space while preserving algebraic structure
        """
        # Compute projection matrix onto range space of M
        # M̄ = M(MᵀM)⁻¹Mᵀ is the orthogonal projector onto col(M)
        # This matrix has eigenvalues 1 for vectors in range(M) and 0 for vectors in null(Mᵀ)
        M_bar = M @ np.linalg.inv(M.T @ M) @ M.T
        
        # Eigendecomposition of the projection matrix
        # M̄ = VΣV⁻¹ where Σ contains eigenvalues (mostly 1s and 0s)
        eigvals, eigvecs = np.linalg.eig(M_bar)
        
        # Select eigenvectors corresponding to eigenvalue 1 (invariant subspace)
        # These span the range space of the mapping matrix M
        eigvecs = eigvecs[:, np.abs(eigvals - 1) < 1e-7]
        
        # Construct L1 projector: orthogonal projection onto invariant subspace
        # L1 = V(VᵀV)⁻¹Vᵀ where V contains invariant eigenvectors
        # This projector preserves the algebraic structure while bandlimiting signals
        L1 = eigvecs @ np.linalg.pinv(eigvecs)
        return L1

    def equi_correction(self, operator):
        """Enforces equivariance constraint on operator through Reynolds projection.

        Operation:
        P̃ = P·vec(Φ) where P is Reynolds projector
        Reshapes flattened operator to maintain Φ's original dimensions
        Ensures P̃∘Φ = Φ∘P̃ (Equivariance commutative diagram)
        """
        # Flatten the operator to a vector for projection
        # vec(Φ) converts the operator matrix to a column vector
        operator_flat = operator.flatten()
        
        # Apply Reynolds projector to enforce equivariance
        # P·vec(Φ) projects the operator onto the equivariant subspace
        # This ensures T(g)·Φ = Φ·ρ_H(g) for all group elements g
        projected_flat = self.equi_projector @ operator_flat
        
        # Reshape back to original operator dimensions
        # This maintains the matrix structure while preserving equivariance
        return projected_flat.reshape(operator.shape)

    def _calculate_M(
        self,
        iterations=50000,
        device="cuda:0",
        smoothness_loss_weight=0.01,
        mode="optim",
    ):
        # Import the solver function for learning the spectral mapping matrix
        from gsampling.layers.solvers import solve_M
        
        # Use the unified solver that supports all modes including gpu_optim
        # The solver learns M that minimizes the objective:
        # ||S·F_G·M - F_H||² + α·tr(Mᵀ·F_Gᵀ·L·F_G·M) + equivariance_penalty
        # where:
        # - S is the subsampling matrix (group homomorphism G → H)
        # - F_G is the original group's Fourier basis
        # - F_H is the subgroup's Fourier basis  
        # - L is the smoothness operator (Laplacian, adjacency, etc.)
        # - α is the smoothness weight
        M = solve_M(
            mode=mode,  # Optimization method (linear_optim, analytical, gpu_optim)
            iterations=iterations,  # Number of optimization iterations
            device=device,  # Compute device for optimization
            smoothness_loss_weight=smoothness_loss_weight,  # Weight for smoothness term
            sampling_matrix=self.sampling_matrix,  # Subsampling matrix S
            basis=self.basis,  # Original group Fourier basis F_G
            sub_basis=self.sub_basis,  # Subgroup Fourier basis F_H
            smoother=self.smoother,  # Smoothness operator L
            dtype=self.dtype,  # Data type for computations
            equi_constraint=self.equi_constraint,  # Whether to enforce equivariance
            equi_raynold_op=self.equi_raynold_op if hasattr(self, 'equi_raynold_op') else None,  # Reynolds operator
        )
        
        return M

    def _forward_f_transform(self, x, basis):
        """Graph Fourier Transform: Projects spatial signals to spectral domain.

        Transform: X̂ = Bᵀx
        Where B = conjugate transpose of graph Fourier basis
        Handles batch-channel-group-spatial dimensions via Einstein summation
        """
        # Construct the Fourier transform matrix B
        # For complex irreps: B = B̄ᵀ (conjugate transpose)
        # For real irreps: B = Bᵀ (transpose)
        B = None
        if self.dtype == torch.cfloat or self.dtype == torch.cdouble:
            # Complex case: use conjugate transpose for unitary transform
            B = torch.transpose(torch.conj(basis), 0, 1)
        elif self.dtype == torch.float or self.dtype == torch.float64:
            # Real case: use transpose for orthogonal transform
            B = torch.transpose(basis, 0, 1)
        else:
            raise ValueError("Invalid dtype: ", self.dtype)

        if len(x.shape) == 1:
            # 1D case: simple matrix-vector multiplication
            # X̂ = B @ x where x is a group signal
            return torch.matmul(B, x.to(basis.dtype))
        elif len(x.shape) == 4:
            # 4D case: (batch, group_size*channel, height, width)
            # Need to separate group and channel dimensions first
            batch, group_channel, height, width = x.shape
            group_size = len(self.nodes)
            channel = group_channel // group_size
            # Reshape to (batch, channel, group_size, height, width)
            x_reshaped = x.view(batch, channel, group_size, height, width)
            # Apply Fourier transform: X̂[fc] = Σ_g B[fg] * x[g]
            return torch.einsum("fg,bcghw->bcfhw", B, x_reshaped.to(basis.dtype))
        elif len(x.shape) == 5:
            # 5D case: (batch, channel, group_size, height, width)
            # Direct application of Fourier transform
            return torch.einsum("fg,bcghw->bcfhw", B, x.to(basis.dtype))
        elif len(x.shape) == 6:
            # 6D case: (batch, channel, group_size, depth, height, width)
            # 3D spatial data with group dimension
            return torch.einsum("fg,bcgdhw->bcfdhw", B, x.to(basis.dtype))
        else:
            raise ValueError(f"Invalid shape: {x.shape}. Expected 1D, 4D, 5D, or 6D tensor.")

    def _inverse_f_transform(self, x, basis):
        """Inverse Graph Fourier Transform: Reconstructs spatial from spectral components.

        Inverse Transform: x = BX̂
        """
        if len(x.shape) == 1:
            # Check if this is upsampling (basis has more rows than columns)
            if basis.shape[0] < basis.shape[1]:
                # Upsampling case: x @ basis where basis is (input_size, output_size)
                # This occurs when reconstructing from a smaller spectral space to larger spatial space
                return torch.matmul(x.to(basis.dtype), basis)
            else:
                # Regular case: basis @ x where basis is (output_size, input_size)
                # Standard inverse Fourier transform: x = B @ X̂
                return torch.matmul(basis, x.to(basis.dtype))
        elif len(x.shape) == 4:
            # 4D case: Handle 2D spatial data with spectral coefficients
            # Inverse transform: x[g] = Σ_f basis[fg] * X̂[f]
            return torch.einsum("fg,bcfhw->bcghw", basis, x.to(basis.dtype))
        elif len(x.shape) == 5:
            # 5D case: (batch, channel, spectral_coeffs, height, width)
            # Reconstruct group signals from spectral domain
            return torch.einsum("fg,bcfhw->bcghw", basis, x.to(basis.dtype))
        elif len(x.shape) == 6:
            # 6D case: (batch, channel, spectral_coeffs, depth, height, width)
            # 3D spatial data reconstruction from spectral coefficients
            return torch.einsum("fg,bcfdhw->bcgdhw", basis, x.to(basis.dtype))
        else:
            raise ValueError(f"Invalid shape: {x.shape}. Expected 1D, 4D, 5D, or 6D tensor.")

    def fft(self, x):
        return self._forward_f_transform(x, self.basis)

    def ifft(self, x):
        return self._inverse_f_transform(x, self.basis)

    def anti_aliase(self, x):
        """Applies anti-aliasing through spectral bandlimiting and reconstruction.

        Processing Pipeline:
        1. FFT(x) = X̂
        2. Bandlimit Projection: X̃ = L1·X̂
        3. IFFT(X̃) = x_anti-aliased
        Where L1 projects to the invariant subspace of M (Generalized Anti-Aliasing Operation)
        """
        original_shape = x.shape
        reshaped = False
        
        # Handle different input shapes and separate group/channel dimensions
        if len(x.shape) == 4:
            # 4D case: (batch, group*channel, height, width)
            # Need to separate group and channel dimensions for proper Fourier transform
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.nodes))
            reshaped = True
        elif len(x.shape) == 5:
            # 5D case: (batch, group*channel, depth, height, width)
            # 3D spatial data with interleaved group and channel dimensions
            group_size = len(self.nodes)
            total_channels = x.shape[1]
            if total_channels % group_size == 0:
                num_features = total_channels // group_size
                x = rearrange(x, "b (c g) d h w -> b c g d h w", g=group_size)
                reshaped = True
            else:
                raise ValueError(f"Channel dimension {total_channels} not divisible by group size {group_size}")

        # Step 1: Forward Fourier Transform
        # Transform spatial group signals to spectral domain: X̂ = F_G^† x
        fh = self.fft(x)
        
        # Step 2: Spectral Bandlimiting
        # Apply L1 projector to remove high-frequency aliasing components
        # L1 projects to the invariant subspace of the learned mapping matrix M
        if len(fh.shape) == 5:
            # 5D case: (batch, channel, spectral_coeffs, height, width)
            # Apply projector: X̃[f] = Σ_g L1[fg] * X̂[g]
            fh_p = torch.einsum("fg,bcghw->bcfhw", self.L1_projector, fh)
        elif len(fh.shape) == 6:
            # 6D case: (batch, channel, spectral_coeffs, depth, height, width)
            # 3D spatial data with spectral coefficients
            fh_p = torch.einsum("fg,bcgdhw->bcfdhw", self.L1_projector, fh)
        else:
            # Fallback for 1D or other cases
            fh_p = self.L1_projector @ fh
            
        # Step 3: Inverse Fourier Transform
        # Reconstruct anti-aliased spatial signals: x_anti-aliased = F_G X̃
        x_out = self.ifft(fh_p)

        # Reshape back to original format if needed
        if reshaped:
            if len(original_shape) == 4:
                # 4D case: back to (batch, group*channel, height, width)
                x_out = rearrange(x_out, "b c g h w -> b (c g) h w")
            elif len(original_shape) == 5:
                # 5D case: back to (batch, group*channel, depth, height, width)
                x_out = rearrange(x_out, "b c g d h w -> b (c g) d h w")
                
        return x_out


    def up_sample(self, x):
        """Upsamples subsampled signals through spectral mapping with energy preservation.

        Mathematical Operations:
        1. Fourier transform:
            X̂_sub = Φ_sub^† x  (where Φ_sub is subgroup's Fourier basis)
        2. Spectral upsampling:
            X̂_full = (Φ_original √N_original / √N_subsampled) @ M @ X̂_sub (where Φ_original is original Groups Fourier basis)
        3. Inverse transform to original spatial domain:
            x_upsampled = Φ_upsample X̂_full
                        = (Φ_original M) X̂_sub * √(N_original/N_subsampled)
        """
        original_shape = x.shape
        reshaped = False
        
        # Handle different input shapes and separate subgroup/channel dimensions
        if len(x.shape) == 4:
            # 4D case: (batch, subgroup*channel, height, width)
            # Need to separate subgroup and channel dimensions
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.subsample_nodes))
            reshaped = True
        elif len(x.shape) == 5:
            # 5D case: (batch, subgroup*channel, depth, height, width)
            # 3D spatial data with interleaved subgroup and channel dimensions
            subgroup_size = len(self.subsample_nodes)
            total_channels = x.shape[1]
            if total_channels % subgroup_size == 0:
                x = rearrange(x, "b (c g) d h w -> b c g d h w", g=subgroup_size)
                reshaped = True
        
        # Step 1: Forward Fourier Transform on subgroup
        # Transform subgroup signals to spectral domain: X̂_sub = F_H^† x
        # where F_H is the subgroup's Fourier basis
        xh = self._forward_f_transform(x, self.sub_basis)
        
        # Step 2: Spectral Upsampling with Energy Preservation
        # Apply upsampling basis: X̂_full = Φ_upsample @ X̂_sub
        # where Φ_upsample = (Φ_G √|G| / √|H|) @ M
        # This preserves energy: ||x_upsampled||² = ||x_sub||²
        x_upsampled = self._inverse_f_transform(xh, self.up_sampling_basis)
        
        # Reshape back to original format if needed
        if reshaped:
            if len(original_shape) == 4:
                # 4D case: back to (batch, group*channel, height, width)
                x_upsampled = rearrange(x_upsampled, "b c g h w -> b (c g) h w")
            elif len(original_shape) == 5:
                # 5D case: back to (batch, group*channel, depth, height, width)
                x_upsampled = rearrange(x_upsampled, "b c g d h w -> b (c g) d h w")
        
        return x_upsampled

    def forward(self, x):
        """Forward pass applies anti-aliasing to input signals.
        
        This is the main entry point for the anti-aliasing layer.
        It applies spectral bandlimiting to prevent aliasing artifacts
        during group downsampling operations.
        """
        return self.anti_aliase(x)
