import torch
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import pdb

def solve_M(
    *,
    mode: str,
    iterations: int,
    device: str,
    smoothness_loss_weight: float,
    sampling_matrix: torch.Tensor,
    basis: torch.Tensor,
    sub_basis: torch.Tensor,
    smoother: torch.Tensor,
    dtype: torch.dtype,
    equi_constraint: bool,
    equi_raynold_op: torch.Tensor | None,
):
    """Solve for spectral mapping M using selected mode.

    Returns torch.Tensor with dtype float64 for linear/gpu modes (as requested) and
    basis dtype for analytical.
    """
    if mode == "analytical":
        # M = (S @ F_G)^† @ F_H
        S_FG = sampling_matrix.to(basis.dtype) @ basis
        M = torch.linalg.pinv(S_FG) @ sub_basis
        return M

    elif mode == "linear_optim":
        # Mixed precision compute for speed, final cast to float64 as requested
        if dtype in (torch.cfloat, torch.cdouble):
            work_dtype = torch.cfloat
            out_high_dtype = torch.float64
        else:
            work_dtype = torch.float32
            out_high_dtype = torch.float64

        F = (sub_basis.clone().to(device, dtype=work_dtype).numpy() * float(sub_basis.shape[0]) ** 0.5)
        FB = (basis.clone().to(device, dtype=work_dtype).numpy() * float(basis.shape[0]) ** 0.5)
        S = sampling_matrix.clone().to(device, dtype=work_dtype).numpy()
        L = smoother.to(device, dtype=work_dtype).numpy()

        if equi_constraint and equi_raynold_op is not None:
            R = equi_raynold_op.clone().to(device, dtype=work_dtype).numpy()
        else:
            R = None

        M0 = torch.zeros((S.shape[1], F.shape[1]), dtype=work_dtype, device=device).numpy()
        F0 = np.zeros_like(F) + 1e-7

        def smoothness_obj(M_flat, FB, L):
            M = M_flat.reshape(FB.shape[1], F.shape[1])
            shift_f = np.dot(L, FB)
            shift_err = np.diag(FB.T @ (FB - shift_f))
            shift_err = shift_err.reshape(-1, M.shape[0])
            return smoothness_loss_weight * np.mean(shift_err @ np.abs(M))

        def equivariance_obj(M_flat, FB, L, R):
            M = M_flat.reshape(FB.shape[1], F.shape[1])
            M_bar = M @ np.linalg.pinv(M.T @ M + 1e-3 * np.eye(M.shape[1])) @ M.T
            evals, evecs = np.linalg.eigh(0.5 * (M_bar + M_bar.T))
            mask = np.abs(evals - 1.0) < 1e-7
            if not np.any(mask):
                return 0.0
            V = evecs[:, mask]
            L1 = V @ np.linalg.pinv(V)
            err = R @ L1.reshape(-1) - L1.reshape(-1)
            return np.mean(np.abs(err) ** 2)

        def objective(M_flat, FB, L, R):
            val = smoothness_obj(M_flat, FB, L)
            if R is not None:
                val += equivariance_obj(M_flat, FB, L, R)
            return val

        A_eq = np.kron(S @ FB, np.eye(F.shape[1]))
        lin_con = LinearConstraint(A_eq, (F - F0).flatten(), (F + F0).flatten())
        bounds = [(None, None)] * (FB.shape[1] * F.shape[1])

        res = minimize(
            objective if R is not None else smoothness_obj,
            M0.flatten(),
            args=(FB, L, R) if R is not None else (FB, L),
            constraints=[lin_con],
            bounds=bounds,
            options={"maxiter": iterations, "disp": True},
        )

        M_opt = res.x.reshape(FB.shape[1], F.shape[1])
        return torch.tensor(M_opt, dtype=out_high_dtype)

    elif mode == "gpu_optim":
        dev = device if torch.cuda.is_available() else "cpu"
        if dtype in (torch.cfloat, torch.cdouble):
            work_dtype = torch.cfloat
            out_high_dtype = torch.float64
        else:
            work_dtype = torch.float32
            out_high_dtype = torch.float64

        S = sampling_matrix.to(device=dev, dtype=work_dtype)
        FG = basis.to(device=dev, dtype=work_dtype)
        FH = sub_basis.to(device=dev, dtype=work_dtype)
        L = smoother.to(device=dev, dtype=work_dtype)

        projector = None
        # If a projector is provided via equi_raynold_op, try to build one
        # The caller can pass precomputed projector if desired in future
        if equi_constraint and equi_raynold_op is not None:
            try:
                R = equi_raynold_op.to(device=dev, dtype=work_dtype)
                ev, evec = torch.linalg.eigh(R)
                evec = evec[:, torch.abs(ev - 1) < 1e-3]
                projector = evec @ torch.linalg.pinv(evec)
            except Exception:
                projector = None

        m_rows, m_cols = FG.shape[1], FH.shape[1]
        M_param = torch.nn.Parameter(torch.zeros(m_rows, m_cols, device=dev, dtype=work_dtype))
        optim = torch.optim.Adam([M_param], lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=100, factor=0.5)

        best = None
        best_loss = float("inf")
        equi_interval = 50
        equi_weight = 0.000001
        for it in range(iterations):
            optim.zero_grad()
            recon = S @ FG @ M_param
            loss_recon = torch.mean((recon - FH).abs() ** 2)
            smooth = torch.trace(M_param.T @ FG.T @ L @ FG @ M_param).real
            loss = loss_recon + smoothness_loss_weight * smooth

            if projector is not None and (it % equi_interval == 0):
                eps = 1e-3
                G = (M_param.conj().T @ M_param) if M_param.is_complex() else (M_param.T @ M_param)
                I = torch.eye(G.shape[0], device=dev, dtype=G.dtype)
                G_reg = G + eps * I
                G_inv = torch.linalg.pinv(G_reg)
                Mbar = (M_param @ G_inv @ (M_param.conj().T if M_param.is_complex() else M_param.T))
                Mbar = 0.5 * (Mbar + (Mbar.conj().T if Mbar.is_complex() else Mbar.T))
                try:
                    eigh_dtype = torch.cdouble if Mbar.is_complex() else torch.float64
                    evals, evecs64 = torch.linalg.eigh(Mbar.to(eigh_dtype))
                    mask = (evals - 1.0).abs() < 1e-7
                    if mask.any():
                        V = evecs64[:, mask].to(work_dtype)
                        L1_now = V @ torch.linalg.pinv(V)
                        L1_v = L1_now.reshape(-1)
                        L1_proj_v = projector @ L1_v
                        loss_equi = torch.mean((L1_v - L1_proj_v).abs() ** 2).real
                        loss = loss + equi_weight * loss_equi
                except RuntimeError:
                    pass

            loss.backward()
            optim.step()
            sched.step(loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best = M_param.detach().clone()
            if it % 1000 == 0:
                print("[GPU_OPT] Iteration ", it, " loss: ", loss.item())

        M_out = (best if best is not None else M_param.detach()).to(out_high_dtype, copy=True).cpu()
        return M_out

    else:
        raise ValueError("Invalid mode:", mode)


