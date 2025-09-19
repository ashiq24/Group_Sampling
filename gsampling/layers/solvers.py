"""
Spectral Mapping Matrix Solvers for Group Equivariant Anti-Aliasing

This module provides optimization algorithms for learning the spectral mapping matrix M
that maps between different group representations while preserving equivariance and
smoothness properties.

Mathematical Foundation:
------------------------
The spectral mapping matrix M is learned by solving the optimization problem:

min_M ||S·F_G·M - F_H||² + α·tr(Mᵀ·F_Gᵀ·L·F_G·M) + β·equivariance_penalty

Where:
- S is the subsampling matrix (group homomorphism G → H)
- F_G is the original group's Fourier basis
- F_H is the subgroup's Fourier basis
- L is the smoothness operator (Laplacian, adjacency, etc.)
- α is the smoothness weight
- β is the equivariance weight

The optimization ensures:
1. Anti-aliasing: S·F_G·M ≈ F_H (spectral consistency)
2. Smoothness: tr(Mᵀ·F_Gᵀ·L·F_G·M) (spectral smoothness)
3. Equivariance: T(g)·M = M·ρ_H(g) (group equivariance)

Key Features:
- Multiple optimization modes (analytical, linear, GPU)
- Support for complex and real representations
- Equivariance constraint enforcement
- Numerical stability considerations

Author: Group Sampling Team
"""

import torch
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import pdb

def solve_M(
    *,
    mode: str,  # Optimization mode ("analytical", "linear_optim", "gpu_optim")
    iterations: int,  # Number of optimization iterations
    device: str,  # Compute device ("cpu" or "cuda")
    smoothness_loss_weight: float,  # Weight for smoothness regularization term
    sampling_matrix: torch.Tensor,  # Subsampling matrix S (|H| x |G|)
    basis: torch.Tensor,  # Original group Fourier basis F_G (|G| x dim(Fourier_space))
    sub_basis: torch.Tensor,  # Subgroup Fourier basis F_H (|H| x dim(Sub_Fourier_space))
    smoother: torch.Tensor,  # Smoothness operator L (|G| x |G|)
    dtype: torch.dtype,  # Data type for computations
    equi_constraint: bool,  # Whether to enforce equivariance constraint
    equi_raynold_op: torch.Tensor | None,  # Reynolds operator for equivariance
):
    """Solve for spectral mapping matrix M using selected optimization mode.
    
    This function learns the spectral mapping matrix M that minimizes the objective:
    ||S·F_G·M - F_H||² + α·tr(Mᵀ·F_Gᵀ·L·F_G·M) + β·equivariance_penalty
    
    Where:
    - S is the subsampling matrix (group homomorphism G → H)
    - F_G is the original group's Fourier basis
    - F_H is the subgroup's Fourier basis
    - L is the smoothness operator
    - α is the smoothness weight
    - β is the equivariance weight
    
    Args:
        mode: Optimization method ("analytical", "linear_optim", "gpu_optim")
        iterations: Number of optimization iterations
        device: Compute device for optimization
        smoothness_loss_weight: Weight for smoothness regularization
        sampling_matrix: Subsampling matrix S
        basis: Original group Fourier basis F_G
        sub_basis: Subgroup Fourier basis F_H
        smoother: Smoothness operator L
        dtype: Data type for computations
        equi_constraint: Whether to enforce equivariance
        equi_raynold_op: Reynolds operator for equivariance projection
        
    Returns:
        Learned spectral mapping matrix M as torch.Tensor
    """
    if mode == "analytical":
        # ANALYTICAL MODE: Direct pseudo-inverse solution
        # This mode provides the least-squares solution to the linear system
        # 
        # Mathematical derivation:
        # We want to minimize ||S·F_G·M - F_H||²
        # Taking the gradient and setting to zero:
        # (S·F_G)ᵀ(S·F_G·M - F_H) = 0
        # (S·F_G)ᵀ(S·F_G)·M = (S·F_G)ᵀ·F_H
        # M = ((S·F_G)ᵀ(S·F_G))⁻¹(S·F_G)ᵀ·F_H
        # M = (S·F_G)⁺·F_H  (where ⁺ denotes pseudo-inverse)
        #
        # This gives the best least-squares approximation without smoothness constraints
        
        # Compute S·F_G (subsampled original group basis)
        S_FG = sampling_matrix.to(basis.dtype) @ basis
        
        # Compute pseudo-inverse and solve for M
        # M = (S·F_G)⁺·F_H
        M = torch.linalg.pinv(S_FG) @ sub_basis
        return M

    elif mode == "linear_optim":
        # LINEAR OPTIMIZATION MODE: Constrained optimization with smoothness and equivariance
        # This mode solves the full optimization problem with constraints:
        # min_M ||S·F_G·M - F_H||² + α·tr(Mᵀ·F_Gᵀ·L·F_G·M) + β·equivariance_penalty
        # subject to linear constraints for numerical stability
        
        # Mixed precision compute for speed, final cast to float64 as requested
        if dtype in (torch.cfloat, torch.cdouble):
            work_dtype = torch.cfloat  # Use complex float for complex representations
            out_high_dtype = torch.float64  # Output in high precision
        else:
            work_dtype = torch.float32  # Use float32 for real representations
            out_high_dtype = torch.float64  # Output in high precision

        # Convert tensors to numpy arrays for scipy optimization
        # Apply normalization factors for numerical stability
        F = (sub_basis.clone().to(device, dtype=work_dtype).numpy() * float(sub_basis.shape[0]) ** 0.5)
        FB = (basis.clone().to(device, dtype=work_dtype).numpy() * float(basis.shape[0]) ** 0.5)
        S = sampling_matrix.clone().to(device, dtype=work_dtype).numpy()
        L = smoother.to(device, dtype=work_dtype).numpy()

        # Handle Reynolds operator for equivariance constraint
        if equi_constraint and equi_raynold_op is not None:
            R = equi_raynold_op.clone().to(device, dtype=work_dtype).numpy()
        else:
            R = None

        # Initialize optimization variables
        M0 = torch.zeros((S.shape[1], F.shape[1]), dtype=work_dtype, device=device).numpy()
        F0 = np.zeros_like(F) + 1e-7  # Small epsilon for constraint bounds

        def smoothness_obj(M_flat, FB, L):
            """Compute smoothness regularization term.
            
            Mathematical: tr(Mᵀ·F_Gᵀ·L·F_G·M)
            This term encourages smooth spectral mappings by penalizing
            high-frequency components in the learned mapping matrix.
            """
            M = M_flat.reshape(FB.shape[1], F.shape[1])
            # Compute shift operator applied to Fourier basis
            shift_f = np.dot(L, FB)
            # Compute shift error (difference between original and shifted basis)
            shift_err = np.diag(FB.T @ (FB - shift_f))
            shift_err = shift_err.reshape(-1, M.shape[0])
            # Apply smoothness weight and return mean error
            return smoothness_loss_weight * np.mean(shift_err @ np.abs(M))

        def equivariance_obj(M_flat, FB, L, R):
            """Compute equivariance constraint term.
            
            This term enforces that the learned mapping respects group symmetries:
            T(g)·M = M·ρ_H(g) for all group elements g ∈ G
            """
            M = M_flat.reshape(FB.shape[1], F.shape[1])
            # Compute projection matrix onto range of M
            M_bar = M @ np.linalg.pinv(M.T @ M + 1e-3 * np.eye(M.shape[1])) @ M.T
            # Eigendecomposition to find invariant subspace
            evals, evecs = np.linalg.eigh(0.5 * (M_bar + M_bar.T))
            mask = np.abs(evals - 1.0) < 1e-7
            if not np.any(mask):
                return 0.0
            # Extract invariant eigenvectors
            V = evecs[:, mask]
            # Construct L1 projector
            L1 = V @ np.linalg.pinv(V)
            # Compute equivariance error using Reynolds operator
            err = R @ L1.reshape(-1) - L1.reshape(-1)
            return np.mean(np.abs(err) ** 2)

        def objective(M_flat, FB, L, R):
            """Combined objective function.
            
            Combines smoothness and equivariance terms:
            smoothness_obj + equivariance_obj (if R is provided)
            """
            val = smoothness_obj(M_flat, FB, L)
            if R is not None:
                val += equivariance_obj(M_flat, FB, L, R)
            return val

        # Set up linear constraints for numerical stability
        # Constraint: S·F_G·M ≈ F_H (with small tolerance)
        # This ensures the learned mapping approximately satisfies the anti-aliasing condition
        A_eq = np.kron(S @ FB, np.eye(F.shape[1]))
        lin_con = LinearConstraint(A_eq, (F - F0).flatten(), (F + F0).flatten())
        bounds = [(None, None)] * (FB.shape[1] * F.shape[1])

        # Run constrained optimization
        res = minimize(
            objective if R is not None else smoothness_obj,  # Objective function
            M0.flatten(),  # Initial guess (flattened matrix)
            args=(FB, L, R) if R is not None else (FB, L),  # Additional arguments
            constraints=[lin_con],  # Linear constraints
            bounds=bounds,  # Variable bounds
            options={"maxiter": iterations, "disp": True},  # Optimization options
        )

        # Reshape solution back to matrix form and convert to tensor
        M_opt = res.x.reshape(FB.shape[1], F.shape[1])
        return torch.tensor(M_opt, dtype=out_high_dtype)

    elif mode == "gpu_optim":
        # GPU OPTIMIZATION MODE: PyTorch-based optimization with GPU acceleration
        # This mode uses PyTorch's automatic differentiation and GPU acceleration
        # for efficient optimization of the spectral mapping matrix
        
        # Determine compute device (GPU if available, otherwise CPU)
        dev = device if torch.cuda.is_available() else "cpu"
        
        # Set up data types for mixed precision computation
        if dtype in (torch.cfloat, torch.cdouble):
            work_dtype = torch.cfloat  # Use complex float for complex representations
            out_high_dtype = torch.float64  # Output in high precision
        else:
            work_dtype = torch.float32  # Use float32 for real representations
            out_high_dtype = torch.float64  # Output in high precision

        # Move tensors to compute device with appropriate dtype
        S = sampling_matrix.to(device=dev, dtype=work_dtype)
        FG = basis.to(device=dev, dtype=work_dtype)
        FH = sub_basis.to(device=dev, dtype=work_dtype)
        L = smoother.to(device=dev, dtype=work_dtype)

        # Build equivariance projector from Reynolds operator
        projector = None
        if equi_constraint and equi_raynold_op is not None:
            try:
                # Convert Reynolds operator to compute device
                R = equi_raynold_op.to(device=dev, dtype=work_dtype)
                # Eigendecomposition to find invariant subspace
                ev, evec = torch.linalg.eigh(R)
                evec = evec[:, torch.abs(ev - 1) < 1e-3]
                # Construct equivariance projector
                projector = evec @ torch.linalg.pinv(evec)
            except Exception:
                projector = None

        # Initialize optimization variables
        m_rows, m_cols = FG.shape[1], FH.shape[1]
        M_param = torch.nn.Parameter(torch.zeros(m_rows, m_cols, device=dev, dtype=work_dtype))
        
        # Set up optimizer and learning rate scheduler
        optim = torch.optim.Adam([M_param], lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=100, factor=0.5)

        # Initialize tracking variables for optimization
        best = None  # Best solution found so far
        best_loss = float("inf")  # Best loss value
        equi_interval = 50  # Interval for applying equivariance constraint
        equi_weight = 0.000001  # Weight for equivariance penalty
        
        # Main optimization loop
        for it in range(iterations):
            optim.zero_grad()
            
            # Compute reconstruction loss: ||S·F_G·M - F_H||²
            recon = S @ FG @ M_param
            loss_recon = torch.mean((recon - FH).abs() ** 2)
            
            # Compute smoothness loss: tr(Mᵀ·F_Gᵀ·L·F_G·M)
            smooth = torch.trace(M_param.T @ FG.T @ L @ FG @ M_param).real
            loss = loss_recon + smoothness_loss_weight * smooth

            # Apply equivariance constraint periodically
            if projector is not None and (it % equi_interval == 0):
                eps = 1e-3  # Regularization for numerical stability
                # Compute Gram matrix G = MᵀM
                G = (M_param.conj().T @ M_param) if M_param.is_complex() else (M_param.T @ M_param)
                I = torch.eye(G.shape[0], device=dev, dtype=G.dtype)
                G_reg = G + eps * I  # Regularized Gram matrix
                G_inv = torch.linalg.pinv(G_reg)
                
                # Compute projection matrix M̄ = M(MᵀM)⁻¹Mᵀ
                Mbar = (M_param @ G_inv @ (M_param.conj().T if M_param.is_complex() else M_param.T))
                Mbar = 0.5 * (Mbar + (Mbar.conj().T if Mbar.is_complex() else Mbar.T))  # Symmetrize
                
                try:
                    # Eigendecomposition to find invariant subspace
                    eigh_dtype = torch.cdouble if Mbar.is_complex() else torch.float64
                    evals, evecs64 = torch.linalg.eigh(Mbar.to(eigh_dtype))
                    mask = (evals - 1.0).abs() < 1e-7
                    if mask.any():
                        # Extract invariant eigenvectors
                        V = evecs64[:, mask].to(work_dtype)
                        # Construct L1 projector
                        L1_now = V @ torch.linalg.pinv(V)
                        L1_v = L1_now.reshape(-1)
                        # Apply equivariance projector
                        L1_proj_v = projector @ L1_v
                        # Compute equivariance loss
                        loss_equi = torch.mean((L1_v - L1_proj_v).abs() ** 2).real
                        loss = loss + equi_weight * loss_equi
                except RuntimeError:
                    pass  # Skip equivariance constraint if eigendecomposition fails

            # Backpropagation and optimization step
            loss.backward()
            optim.step()
            sched.step(loss)
            
            # Track best solution
            if loss.item() < best_loss:
                best_loss = loss.item()
                best = M_param.detach().clone()
            
            # Print progress periodically
            if it % 1000 == 0:
                print("[GPU_OPT] Iteration ", it, " loss: ", loss.item())

        # Return best solution in high precision
        M_out = (best if best is not None else M_param.detach()).to(out_high_dtype, copy=True).cpu()
        return M_out

    else:
        raise ValueError("Invalid mode:", mode)


