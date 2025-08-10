import torch
from scipy.optimize import minimize, LinearConstraint
import numpy as np
from einops import rearrange
from .helper import SmoothOperatorFactory, ReynoldsProjectorHelper, L1ProjectorUtils, FourierOps
from .solvers import solve_M


class AntiAliasingLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        nodes: list,
        adjaceny_matrix: np.ndarray,
        basis: np.ndarray,
        subsample_nodes: list,
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

        # Build smooth/selection operator via factory
        self.register_buffer(
            "smoother",
            SmoothOperatorFactory.build(
                self.adjacency_matrix, smooth_operator, graph_shift, dtype
            ),
        )
        # Register bases with proper dtype upfront
        basis_tensor = torch.tensor(basis, dtype=dtype)
        sub_basis_tensor = torch.tensor(sub_basis, dtype=dtype)
        self.register_buffer("basis", basis_tensor)
        self.register_buffer("sub_basis", sub_basis_tensor)

        # Reynolds artifacts (if provided) — keep original mechanism intact
        if equi_constraint and raynold_op is not None:
            # Store raw Reynolds operator tensor for legacy optimization path
            self.equi_raynold_op = torch.tensor(raynold_op)
            # Also precompute projector for post-correction (same as before)
            self.register_buffer(
                "equi_projector", ReynoldsProjectorHelper.build_projector(raynold_op, dtype)
            )
        else:
            self.equi_projector = None
            
        self.sampling_matrix = torch.zeros(
            len(self.subsample_nodes), len(self.nodes), dtype=dtype
        )
        for i, node in enumerate(self.subsample_nodes):
            self.sampling_matrix[i, node] = 1

        self._init_anti_aliasing_operator()

        # Ensure consistent dtype when forming upsampling basis
        up_basis = (self.basis.to(self.dtype) * (self.basis.shape[0] ** 0.5)) @ self.M
        up_basis = up_basis / (self.sub_basis.shape[0] ** 0.5)
        self.register_buffer("up_sampling_basis", up_basis.to(self.dtype))

        if equi_correction:
            if self.equi_projector is not None:
                self.L1_projector = ReynoldsProjectorHelper.project(
                    self.equi_projector, self.L1_projector
                )

        # Already registered with correct dtype

    # Removed old in-class smooth initializer (replaced by factory)

    # Removed old Reynolds setup (replaced by helper)

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
            solve_M(
                mode=self.mode,
                iterations=self.iterations,
                device=self.device,
                smoothness_loss_weight=self.smoothness_loss_weight,
                sampling_matrix=self.sampling_matrix,
                basis=self.basis,
                sub_basis=self.sub_basis,
                smoother=self.smoother,
                dtype=self.dtype,
                equi_constraint=self.equi_constraint,
                equi_raynold_op=self.equi_raynold_op,
            ),
        )

        # 0 out smaller number in M
        self.M[torch.abs(self.M) <= self.threshold] = 0
        # Keep a high-precision copy for downstream sensitive computations
        self.register_buffer("M_high64", self.M.to(torch.float64))
        # Ensure learned mapping is in the configured dtype to avoid matmul dtype mismatches
        self.M = self.M.to(self.dtype)
        # L1 eigvecs and projector
        eigvecs, L1 = L1ProjectorUtils.compute_from_M(self.M, self.dtype)
        self.register_buffer("L1_eigs", eigvecs)
        self.register_buffer("L1_projector", L1)

    # Removed in-class l1_projector (moved to L1ProjectorUtils)

    # Removed in-class equi_correction (use ReynoldsProjectorHelper.project)

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
            # Mixed-precision strategy: numerics in FP32/complex64 for speed, final cast to float64
            if self.dtype in (torch.cfloat, torch.cdouble):
                high_precision_dtype = torch.cfloat  # complex64 compute
                out_high_dtype = torch.float64       # return in float64
            else:
                high_precision_dtype = torch.float32
                out_high_dtype = torch.float64

            F = (
                self.sub_basis.clone().to(device, dtype=high_precision_dtype).numpy()
                * float(self.sub_basis.shape[0]) ** 0.5
            )

            FB = (
                self.basis.clone().to(device, dtype=high_precision_dtype).numpy()
                * float(self.basis.shape[0]) ** 0.5
            )

            S = (
                self.sampling_matrix.clone()
                .to(device, dtype=high_precision_dtype)
                .numpy()
            )
            L = self.smoother.to(device, dtype=high_precision_dtype).numpy()
            if self.equi_constraint and getattr(self, "equi_raynold_op", None) is not None:
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

                Assumes M is already shaped into 2D.
                """
                # Reuse the original numpy L1 projector to preserve behavior
                L1 = L1ProjectorUtils._l1_projector_numpy(M).reshape(-1)
                error = R @ L1 - L1
                return np.linalg.norm(error, ord=2)

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

            # Return in high precision as requested to protect downstream numerics
            return torch.tensor(M, dtype=out_high_dtype)

        elif mode == "gpu_optim":
            """
            GPU-accelerated optimization using PyTorch (Adam) with optional mixed precision.
            Core mechanism is the same objective; this is a new mode for speed.
            """
            device = self.device if torch.cuda.is_available() else "cpu"
            # Use fp32/complex64 for compute; keep final fp64 copy
            if self.dtype in (torch.cfloat, torch.cdouble):
                work_dtype = torch.cfloat
                out_high_dtype = torch.float64
            else:
                work_dtype = torch.float32
                out_high_dtype = torch.float64

            S = self.sampling_matrix.to(device=device, dtype=work_dtype)
            FG = self.basis.to(device=device, dtype=work_dtype)
            FH = self.sub_basis.to(device=device, dtype=work_dtype)
            L = self.smoother.to(device=device, dtype=work_dtype)

            # Optional equivariance projector for L1(M)
            projector_device = None
            if getattr(self, "equi_projector", None) is not None:
                projector_device = self.equi_projector.to(device=device, dtype=work_dtype)

            m_rows, m_cols = FG.shape[1], FH.shape[1]
            M_param = torch.nn.Parameter(torch.zeros(m_rows, m_cols, device=device, dtype=work_dtype))
            optim = torch.optim.Adam([M_param], lr=1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=50, factor=0.5)

            best = None
            best_loss = float('inf')
            equi_interval = 100
            equi_weight = 0.01
            for it in range(min(self.iterations, 2000)):
                optim.zero_grad()
                recon = S @ FG @ M_param
                loss_recon = torch.mean((recon - FH).abs() ** 2)
                smooth = torch.trace(M_param.T @ FG.T @ L @ FG @ M_param).real
                loss = loss_recon + smoothness_loss_weight * smooth
                # Equivariance loss (occasional to reduce overhead) — stabilized Hermitian L1 computation
                if projector_device is not None and (it % equi_interval == 0):
                    eps = 1e-3
                    # Regularized Gram and Hermitian Mbar
                    if M_param.is_complex():
                        G = M_param.conj().T @ M_param
                    else:
                        G = M_param.T @ M_param
                    I = torch.eye(G.shape[0], device=device, dtype=G.dtype)
                    G_reg = G + eps * I
                    G_inv = torch.linalg.pinv(G_reg)
                    if M_param.is_complex():
                        Mbar = M_param @ G_inv @ M_param.conj().T
                    else:
                        Mbar = M_param @ G_inv @ M_param.T
                    Mbar = 0.5 * (Mbar + (Mbar.conj().T if Mbar.is_complex() else Mbar.T))

                    try:
                        eigh_dtype = torch.cdouble if Mbar.is_complex() else torch.float64
                        evals, evecs64 = torch.linalg.eigh(Mbar.to(eigh_dtype))
                        mask = (evals - 1.0).abs() < 1e-7
                        if mask.any():
                            V = evecs64[:, mask].to(work_dtype)
                            L1_now = V @ torch.linalg.pinv(V)
                            L1_v = L1_now.reshape(-1)
                            L1_proj_v = projector_device @ L1_v
                            loss_equi = torch.mean((L1_v - L1_proj_v).abs() ** 2).real
                            loss = loss + equi_weight * loss_equi
                    except RuntimeError:
                        print("Warning: Eigendecomposition failed, skipping equivariance term")
                        pass
                loss.backward()
                optim.step()
                scheduler.step(loss)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best = M_param.detach().clone()
                if it % 100 == 0:
                    print(f"[gpu_optim] iter={it} loss={loss.item():.6f}")

            M_out = (best if best is not None else M_param.detach()).to(out_high_dtype, copy=True).cpu()
            return M_out
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
        elif len(x.shape) == 5:
            # assuming tensor of shape (batch, group_size, channel, height, width)
            return torch.einsum("fg,bcghw->bcfhw", B, x.to(basis.dtype))
        else:
            raise ValueError("Invalid shape: ", x.shape)

    def _inverse_f_transform(self, x, basis):
        """Inverse Graph Fourier Transform: Reconstructs spatial from spectral components.

        Inverse Transform: x = BX̂
        """
        if len(x.shape) == 1:
            return torch.matmul(basis, x)
        elif len(x.shape) == 5:
            # assuming tensor of shape (batch, group_size , channel, height, width)
            return torch.einsum("fg,bcghw->bcfhw", basis, x.to(basis.dtype))
        else:
            raise ValueError("Invalid shape: ", x.shape)

    def fft(self, x):
        return FourierOps.forward(x, self.basis, self.dtype)

    def ifft(self, x):
        return FourierOps.inverse(x, self.basis)

    def anti_aliase(self, x):
        """Applies anti-aliasing through spectral bandlimiting and reconstruction.

        Processing Pipeline:
        1. FFT(x) = X̂
        2. Bandlimit Projection: X̃ = L1·X̂
        3. IFFT(X̃) = x_anti-aliased
        Where L1 projects to the invariant subspace of M (Generalized Anti-Aliasing Operation)
        """
        if len(x.shape) == 4:
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.nodes))

        fh = self.fft(x)
        if len(x.shape) == 5:
            fh_p = torch.einsum("fg,bcghw->bcfhw", self.L1_projector, fh)
        else:
            fh_p = self.L1_projector @ fh
        x_out = self.ifft(fh_p)

        if len(x.shape) == 5:
            x_out = rearrange(x_out, "b  c g h w -> b (c g) h w")
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
        if len(x.shape) == 4:
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.subsample_nodes))
        xh = self._forward_f_transform(x, self.sub_basis)
        x_upsampled = self._inverse_f_transform(xh, self.up_sampling_basis)
        if len(x.shape) == 5:
            x_upsampled = rearrange(x_upsampled, "b c g h w -> b (c g) h w")
        return x_upsampled
    def forward(self, x):
        y =self.anti_aliase(x)
        return y

