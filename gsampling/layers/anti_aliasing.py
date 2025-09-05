import torch
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import numpy as np
from einops import rearrange


class AntiAliasingLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        nodes: list,
        adjaceny_matrix: np.ndarray,
        basis: np.ndarray,
        subsample_nodes: list,
        subsample_adjacency_matrix: np.ndarray,
        sub_basis: np.ndarray,
        smooth_operator: str = "laplacian",
        smoothness_loss_weight=1.0,
        iterations=500,
        device="cuda:0",
        dtype=torch.cfloat,
        graph_shift: np.ndarray = None,
        raynold_op: np.ndarray = None,
        equi_constraint: bool = True,
        equi_correction: bool = True,
        mode: str = "linear_optim",
        threshold=1e-3
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

        self.nodes = nodes
        self.subsample_nodes = subsample_nodes
        self.device = device
        self.dtype = dtype
        self.iterations = iterations
        self.mode = mode
        self.smoothness_loss_weight = smoothness_loss_weight
        # make adjacency matrix from edges
        self.adjacency_matrix = torch.tensor(adjaceny_matrix)
        self.equi_constraint = equi_constraint
        self.equi_raynold_op = None
        self.threshold = threshold

        print(
            "Equi Constraint: ", equi_constraint, "Equi Correction: ", equi_correction
        )

        self._init_smooth_selection_operator(
            graph_shift=graph_shift, smooth_operator=smooth_operator, dtype=dtype
        )
        self.register_buffer("basis", torch.tensor(basis))
        self.register_buffer("sub_basis", torch.tensor(sub_basis))

        self._init_raynold_op(raynold_op=raynold_op, dtype=dtype)

        self.subsample_adjacency_matrix = torch.tensor(subsample_adjacency_matrix)
        self.sampling_matrix = torch.zeros(len(self.subsample_nodes), len(self.nodes))
        for i, node in enumerate(self.subsample_nodes):
            self.sampling_matrix[i, node] = 1

        self._init_anti_aliasing_operator()

        self.register_buffer(
            "up_sampling_basis",
            (self.basis * self.basis.shape[0] ** 0.5 @ self.M).to(self.dtype)
            / self.sub_basis.shape[0] ** 0.5,
        )

        if equi_correction:
            self.L1_projector = self.equi_correction(self.L1_projector)

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
            smoother = self.adjacency_matrix / torch.sum(
                self.adjacency_matrix, dim=1, keepdim=True
            )
        elif smooth_operator == "laplacian":
            degree_matrix = torch.diag(torch.sum(self.adjacency_matrix, dim=1))
            smoother = degree_matrix - self.adjacency_matrix
        elif smooth_operator == "normalized_laplacian":
            degree_matrix = torch.diag(torch.sum(self.adjacency_matrix, dim=1))
            smoother = degree_matrix - self.adjacency_matrix
            degree_matrix_power = torch.sqrt(1.0 / degree_matrix)
            degree_matrix_power[degree_matrix_power == float("inf")] = 0
            smoother = degree_matrix_power @ smoother @ degree_matrix_power
        elif smooth_operator == "graph_shift" and graph_shift is not None:
            smoother = torch.tensor(graph_shift)
        else:
            raise ValueError("Invalid smooth operator: ", smooth_operator)

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
            raynold_tensor = torch.tensor(raynold_op).to(dtype)
            
            # Ensure we have a 2D matrix for eigendecomposition
            if raynold_tensor.ndim == 1:
                # Reshape 1D array to square matrix
                size = int(np.sqrt(raynold_tensor.shape[0]))
                raynold_tensor = raynold_tensor.reshape(size, size)
            elif raynold_tensor.ndim == 0:
                # Handle scalar case (fallback to identity)
                raynold_tensor = torch.eye(1, dtype=dtype)
            elif raynold_tensor.ndim > 2:
                # Flatten higher dimensional arrays
                raynold_tensor = raynold_tensor.reshape(-1)
                size = int(np.sqrt(raynold_tensor.shape[0]))
                raynold_tensor = raynold_tensor.reshape(size, size)
            
            self.equi_raynold_op = raynold_tensor
            
            # Perform eigendecomposition if the matrix is large enough
            if self.equi_raynold_op.shape[0] >= 2:
                ev, evec = torch.linalg.eigh(self.equi_raynold_op)
                evec = evec[:, torch.abs(ev - 1) < 1e-3]
                if evec.shape[1] > 0:
                    self.register_buffer(
                        "equi_projector", evec @ np.linalg.inv(evec.T @ evec) @ evec.T
                    )
                else:
                    # No eigenvectors close to 1, use identity
                    self.register_buffer(
                        "equi_projector", torch.eye(self.equi_raynold_op.shape[0], dtype=dtype)
                    )
            else:
                # For small matrices, use identity
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
        # register buffer for M
        self.register_buffer(
            "M",
            self._calculate_M(
                iterations=self.iterations,
                device="cpu",
                smoothness_loss_weight=self.smoothness_loss_weight,
                mode=self.mode,
            ),
        )

        # 0 out smaller number in M
        self.M[torch.abs(self.M) <= self.threshold] = 0
        self.M_bar = self.M @ torch.linalg.inv(self.M.T @ self.M) @ self.M.T
        eigvals, eigvecs = torch.linalg.eig(self.M_bar)
        # only take eigenvectors with eigenvalues close to 1
        eigvecs = eigvecs[:, torch.abs(eigvals - 1) < 1e-7]
        # zero out smaller number in eigvecs
        eigvecs[torch.abs(eigvecs) < 1e-6] = 0
        self.M.to(self.dtype)

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
        M_bar = M @ np.linalg.inv(M.T @ M) @ M.T
        eigvals, eigvecs = np.linalg.eig(M_bar)
        # only take eigenvectors with eigenvalues close to 1
        eigvecs = eigvecs[:, np.abs(eigvals - 1) < 1e-7]
        L1 = eigvecs @ np.linalg.pinv(eigvecs)
        return L1

    def equi_correction(self, operator):
        """Enforces equivariance constraint on operator through Reynolds projection.

        Operation:
        P̃ = P·vec(Φ) where P is Reynolds projector
        Reshapes flattened operator to maintain Φ's original dimensions
        Ensures P̃∘Φ = Φ∘P̃ (Equivariance commutative diagram)
        """
        return (self.equi_projector @ operator.flatten()).reshape(operator.shape)

    def _calculate_M(
        self,
        iterations=50000,
        device="cuda:0",
        smoothness_loss_weight=0.01,
        mode="optim",
    ):
        if mode == "analytical":
            print("===Using Analytical Solution====")
            """
            used analytical solution to solve the optimization problem
            """
            M = (
                torch.linalg.pinv(
                    self.sampling_matrix.to(self.basis.dtype) @ self.basis
                )
                @ self.sub_basis
            )

            return M

        elif mode == "linear_optim":
            """
            used alternating lease square programming to solve the optimization problem
            """

            print("===Using Linear Optimization====")
            # setting high precision dtype for optimization
            high_precision_dtype = None
            if self.dtype == torch.cfloat:
                high_precision_dtype = torch.cfloat64
            elif self.dtype == torch.float:
                high_precision_dtype = torch.float64

            F = (
                self.sub_basis.clone().to(device, dtype=high_precision_dtype).numpy()
                * self.sub_basis.shape[0] ** 0.5
            )

            FB = (
                self.basis.clone().to(device, dtype=high_precision_dtype).numpy()
                * self.basis.shape[0] ** 0.5
            )

            S = (
                self.sampling_matrix.clone()
                .to(device, dtype=high_precision_dtype)
                .numpy()
            )
            L = self.smoother.to(device, dtype=high_precision_dtype).numpy()
            if self.equi_constraint:
                R = (
                    self.equi_raynold_op.clone()
                    .to(device, dtype=high_precision_dtype)
                    .numpy()
                )
            else:
                R = None
            M = torch.zeros(
                (S.shape[1], F.shape[1]), dtype=high_precision_dtype, device=device
            ).numpy()

            F0 = np.zeros_like(F) + 1e-7  # tolerance for linear constraint

            def smoothness_ob1(M, FB, L):
                """
                Objective function to minimize: \sum |FB - L * FB|_2 x M
                """

                M = M.reshape(FB.shape[1], F.shape[1])
                shift_f = np.dot(L, FB)
                shift_err = np.diag(FB.T @ (FB - shift_f))
                shift_err = shift_err.reshape(-1, M.shape[0])

                return smoothness_loss_weight * np.mean(shift_err @ np.abs(M))

            def objective_function4(M, FB, L, R):
                """
                Objective function to minimize: |FB - L * FB . M|_2 + |R . L1 - L1|_2
                """
                smoothness_loss = smoothness_ob1(M, FB, L)

                M = M.reshape(FB.shape[1], F.shape[1])
                equi_error = equivarinace_loss(M, R)

                return smoothness_loss + 1 * equi_error

            def equivarinace_loss(M, R):
                """
                Objective function to minimize: |R . L1 - L1|_2

                Assumes M is already shaped into 2D (group_size, feature_size).
                """
                # Get L1 projector as 2D matrix (group_size, group_size)
                L1_matrix = self.l1_projector(M)  # Shape: (group_size, group_size)
                
                # Apply Reynolds operator to each column of L1 (each feature)
                # R operates on the group dimension (first dimension)
                RL1 = R @ L1_matrix  # Shape: (group_size, group_size)
                error_matrix = RL1 - L1_matrix
                
                return np.linalg.norm(error_matrix, ord='fro')  # Frobenius norm

            initial_guess_M = np.random.randn(*M.shape).flatten()

            print("Initial guess M:", initial_guess_M.shape)

            # Define the linear equality constraint
            linear_constraint_matrix = np.kron(S @ FB, np.eye(F.shape[1]))
            print("Linear Constraint Matrix:", linear_constraint_matrix.shape)

            linear_constraint = LinearConstraint(
                linear_constraint_matrix, (F - F0).flatten(), (F + F0).flatten()
            )

            # Set bounds to enforce strict equality constraint
            bounds = [(None, None)] * (FB.shape[1] * F.shape[1])

            print("*** starting optimization ***")
            if self.equi_constraint:
                result = minimize(
                    objective_function4,
                    initial_guess_M,
                    args=(FB, L, R),
                    constraints=[linear_constraint],
                    bounds=bounds,
                    options={"maxiter": iterations, "disp": True},
                )
            else:
                result = minimize(
                    smoothness_ob1,
                    initial_guess_M,
                    args=(FB, L),
                    constraints=[linear_constraint],
                    bounds=bounds,
                    options={"maxiter": iterations, "disp": True},
                )
            print("*** optimization done ***")

            optimal_M = result.x

            print("Optimal objective value:", result.fun)

            M = optimal_M.reshape(FB.shape[1], F.shape[1])

            print(" Final Loss Reconstruction :", np.linalg.norm(F - S @ FB @ M))
            print(" Final Equivarinace loss :", equivarinace_loss(M, R))

            return torch.tensor(M, dtype=torch.float64)
        else:
            raise ValueError("Invalid mode: ", mode)

    def _forward_f_transform(self, x, basis):
        """Graph Fourier Transform: Projects spatial signals to spectral domain.

        Transform: X̂ = Bᵀx
        Where B = conjugate transpose of graph Fourier basis
        Handles batch-channel-group-spatial dimensions via Einstein summation
        """
        B = None
        if self.dtype == torch.cfloat or self.dtype == torch.cdouble:
            B = torch.transpose(torch.conj(basis), 0, 1)
        elif self.dtype == torch.float or self.dtype == torch.float64:
            B = torch.transpose(basis, 0, 1)
        else:
            raise ValueError("Invalid dtype: ", self.dtype)

        if len(x.shape) == 1:
            return torch.matmul(B, x.to(basis.dtype))
        elif len(x.shape) == 4:
            # Handle 2D spatial data: (batch, group_size*channel, height, width)
            # First reshape to separate group dimension
            batch, group_channel, height, width = x.shape
            group_size = len(self.nodes)
            channel = group_channel // group_size
            x_reshaped = x.view(batch, channel, group_size, height, width)
            return torch.einsum("fg,bcghw->bcfhw", B, x_reshaped.to(basis.dtype))
        elif len(x.shape) == 5:
            # Handle 3D spatial data: (batch, channel, group_size, height, width) or 
            # (batch, group_size, channel, depth, height, width) depending on convention
            if x.shape[2] == len(self.nodes):
                # Convention: (batch, channel, group_size, height, width)
                return torch.einsum("fg,bcghw->bcfhw", B, x.to(basis.dtype))
            else:
                # Convention: (batch, group_size, channel, depth, height, width)
                return torch.einsum("fg,bcgdhw->bcfdhw", B, x.to(basis.dtype))
        elif len(x.shape) == 6:
            # Handle 3D spatial data: (batch, channel, group_size, depth, height, width)
            return torch.einsum("fg,bcgdhw->bcfdhw", B, x.to(basis.dtype))
        else:
            raise ValueError(f"Invalid shape: {x.shape}. Expected 1D, 4D, 5D, or 6D tensor.")

    def _inverse_f_transform(self, x, basis):
        """Inverse Graph Fourier Transform: Reconstructs spatial from spectral components.

        Inverse Transform: x = BX̂
        """
        if len(x.shape) == 1:
            return torch.matmul(basis, x)
        elif len(x.shape) == 4:
            # Handle 2D spatial data
            return torch.einsum("fg,bcfhw->bcghw", basis, x.to(basis.dtype))
        elif len(x.shape) == 5:
            # Handle 3D spatial data: (batch, channel, group_size, height, width) or 
            # (batch, group_size, channel, depth, height, width)
            if x.shape[2] == len(self.nodes):
                # Convention: (batch, channel, group_size, height, width)
                return torch.einsum("fg,bcfhw->bcghw", basis, x.to(basis.dtype))
            else:
                # Convention: (batch, group_size, channel, depth, height, width)
                return torch.einsum("fg,bcfdhw->bcgdhw", basis, x.to(basis.dtype))
        elif len(x.shape) == 6:
            # Handle 3D spatial data: (batch, channel, group_size, depth, height, width)
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
        
        # Handle different input shapes
        if len(x.shape) == 4:
            # 2D spatial: (batch, group*channel, height, width)
            # Need to separate group and channel dimensions
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.nodes))
            reshaped = True
        elif len(x.shape) == 5:
            # 3D spatial: (batch, group*channel, depth, height, width)
            # Need to separate group and channel dimensions
            group_size = len(self.nodes)
            total_channels = x.shape[1]
            if total_channels % group_size == 0:
                num_features = total_channels // group_size
                x = rearrange(x, "b (c g) d h w -> b c g d h w", g=group_size)
                reshaped = True
            else:
                raise ValueError(f"Channel dimension {total_channels} not divisible by group size {group_size}")

        fh = self.fft(x)
        
        # Apply L1 projector based on tensor dimensions
        if len(fh.shape) == 5:
            # 2D spatial or (batch, channel, group, height, width)
            fh_p = torch.einsum("fg,bcghw->bcfhw", self.L1_projector, fh)
        elif len(fh.shape) == 6:
            # 3D spatial: (batch, channel, group, depth, height, width)
            fh_p = torch.einsum("fg,bcgdhw->bcfdhw", self.L1_projector, fh)
        else:
            # Fallback for 1D or other cases
            fh_p = self.L1_projector @ fh
            
        x_out = self.ifft(fh_p)

        # Reshape back to original format if needed
        if reshaped:
            if len(original_shape) == 4:
                # 2D spatial: back to (batch, group*channel, height, width)
                x_out = rearrange(x_out, "b c g h w -> b (c g) h w")
            elif len(original_shape) == 5:
                # 3D spatial: back to (batch, group*channel, depth, height, width)
                x_out = rearrange(x_out, "b c g d h w -> b (c g) d h w")
                
        return x_out

    def apply_subsample_matrix(self, x):
        """
        Applies subsampling matrix to input signals.
        """
        return self.sampling_matrix @ x

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
        
        if len(x.shape) == 4:
            # 2D spatial: (batch, subgroup*channel, height, width)
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.subsample_nodes))
            reshaped = True
        elif len(x.shape) == 5:
            # 3D spatial: (batch, subgroup*channel, depth, height, width)
            subgroup_size = len(self.subsample_nodes)
            total_channels = x.shape[1]
            if total_channels % subgroup_size == 0:
                x = rearrange(x, "b (c g) d h w -> b c g d h w", g=subgroup_size)
                reshaped = True
        
        xh = self._forward_f_transform(x, self.sub_basis)
        x_upsampled = self._inverse_f_transform(xh, self.up_sampling_basis)
        
        if reshaped:
            if len(original_shape) == 4:
                # 2D spatial: back to (batch, group*channel, height, width)
                x_upsampled = rearrange(x_upsampled, "b c g h w -> b (c g) h w")
            elif len(original_shape) == 5:
                # 3D spatial: back to (batch, group*channel, depth, height, width)
                x_upsampled = rearrange(x_upsampled, "b c g d h w -> b (c g) d h w")
        
        return x_upsampled

    def forward(self, x):
        return self.anti_aliase(x)
