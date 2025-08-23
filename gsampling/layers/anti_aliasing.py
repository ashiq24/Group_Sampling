import torch
from scipy.optimize import minimize, LinearConstraint
import numpy as np
from einops import rearrange
from .helper import SmoothOperatorFactory, ReynoldsProjectorHelper, L1ProjectorUtils, FourierOps
from .solvers import solve_M
import pdb

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
        elif len(x.shape) == 5:
            x = rearrange(x, "b (c g) h w d -> b c g h w d", g=len(self.nodes))

        fh = self.fft(x)
        if len(x.shape) == 5:
            fh_p = torch.einsum("fg,bcghw->bcfhw", self.L1_projector, fh)
        elif len(x.shape) == 6:
            fh_p = torch.einsum("fg,bcghwd->bcfhwd", self.L1_projector, fh)
        else:
            fh_p = self.L1_projector @ fh
        x_out = self.ifft(fh_p)

        if len(x.shape) == 5:
            x_out = rearrange(x_out, "b  c g h w -> b (c g) h w")
        elif len(x.shape) == 6:
            x_out = rearrange(x_out, "b c g h w d -> b (c g) h w d")
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
        elif len(x.shape) == 5:
            x = rearrange(x, "b (c g) h w d -> b c g h w d", g=len(self.subsample_nodes))
        xh = FourierOps.forward(x, self.sub_basis, self.dtype)
        x_upsampled = FourierOps.inverse(xh, self.up_sampling_basis)
        if len(x.shape) == 5:
            x_upsampled = rearrange(x_upsampled, "b c g h w -> b (c g) h w")
        elif len(x.shape) == 6:
            x_upsampled = rearrange(x_upsampled, "b c g h w d -> b (c g) h w d")
        return x_upsampled
    def forward(self, x):
        y =self.anti_aliase(x)
        return y

