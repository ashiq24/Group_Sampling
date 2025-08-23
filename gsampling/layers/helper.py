import torch
from scipy.optimize import minimize, LinearConstraint
import numpy as np
from einops import rearrange



class SmoothOperatorFactory:
    """Factory to construct graph smoothness/shift operators.

    Keeps the exact logic as the original `_init_smooth_selection_operator` while
    centralizing it for reuse and easier extension to new group types.
    """

    @staticmethod
    def build(adjacency_matrix: torch.Tensor,
              smooth_operator: str = "laplacian",
              graph_shift: torch.Tensor | None = None,
              dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
        if smooth_operator == "adjacency":
            smoother = adjacency_matrix / torch.sum(adjacency_matrix, dim=1, keepdim=True)
        elif smooth_operator == "laplacian":
            degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
            smoother = degree_matrix - adjacency_matrix
        elif smooth_operator == "normalized_laplacian":
            degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
            smoother = degree_matrix - adjacency_matrix
            degree_matrix_power = torch.sqrt(1.0 / degree_matrix)
            degree_matrix_power[degree_matrix_power == float("inf")] = 0
            smoother = degree_matrix_power @ smoother @ degree_matrix_power
        elif smooth_operator == "graph_shift" and graph_shift is not None:
            smoother = torch.tensor(graph_shift)
        else:
            raise ValueError("Invalid smooth operator: ", smooth_operator)

        return smoother.to(dtype)


class ReynoldsProjectorHelper:
    """Utilities for Reynolds operator and equivariant projection.

    Matches the prior code path to compute `equi_projector` from a provided
    Reynolds operator and apply it to any operator via vectorization.
    """

    @staticmethod
    def build_projector(raynold_op: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if raynold_op is None:
            raise ValueError("raynold_op cannot be None when building projector")
        R = torch.tensor(raynold_op).to(dtype)
        ev, evec = torch.linalg.eigh(R)
        evec = evec[:, torch.abs(ev - 1) < 1e-3]
        # Keep original numpy-based pseudo-inverse shape handling
        return evec @ np.linalg.inv(evec.T @ evec) @ evec.T

    @staticmethod
    def project(projector: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        return (projector @ operator.flatten()).reshape(operator.shape)


class L1ProjectorUtils:
    """Utilities to compute L1 projector from a learned mapping M.

    Preserves the same computation steps as in the original implementation.
    """

    @staticmethod
    def compute_from_M(M: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # M_bar in torch for eigen-analysis and eigvec thresholding (kept as in original layer)
        M_bar = M @ torch.linalg.inv(M.T @ M) @ M.T
        eigvals, eigvecs = torch.linalg.eig(M_bar)
        eigvecs = eigvecs[:, torch.abs(eigvals - 1) < 1e-7]
        eigvecs[torch.abs(eigvecs) < 1e-6] = 0

        # L1 projector via numpy routine (unchanged math):
        M_np = M.detach().cpu().numpy()
        L1_np = L1ProjectorUtils._l1_projector_numpy(M_np)
        L1 = torch.tensor(L1_np).to(dtype)
        return eigvecs, L1

    @staticmethod
    def _l1_projector_numpy(M: np.ndarray) -> np.ndarray:
        M_bar = M @ np.linalg.inv(M.T @ M) @ M.T
        eigvals, eigvecs = np.linalg.eig(M_bar)
        eigvecs = eigvecs[:, np.abs(eigvals - 1) < 1e-7]
        L1 = eigvecs @ np.linalg.pinv(eigvecs)
        return L1


class FourierOps:
    """Fourier transform helpers; mirrors original einsum/matmul logic.

    Centralized so extending to new group bases is straightforward.
    """

    @staticmethod
    def forward(x: torch.Tensor, basis: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if dtype in [torch.cfloat, torch.cdouble]:
            B = torch.transpose(torch.conj(basis), 0, 1)
        elif dtype in [torch.float, torch.float64]:
            B = torch.transpose(basis, 0, 1)
        else:
            raise ValueError("Invalid dtype: ", dtype)

        if len(x.shape) == 1:
            return torch.matmul(B, x.to(basis.dtype))
        elif len(x.shape) == 5:
            return torch.einsum("fg,bcghw->bcfhw", B, x.to(basis.dtype))
        elif len(x.shape) == 6:
            return torch.einsum("fg,bcghwd->bcfhwd", B, x.to(basis.dtype))
        else:
            raise ValueError("Invalid shape: ", x.shape)

    @staticmethod
    def inverse(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            return torch.matmul(basis, x)
        elif len(x.shape) == 5:
            return torch.einsum("fg,bcghw->bcfhw", basis, x.to(basis.dtype))
        elif len(x.shape) == 6:
            return torch.einsum("fg,bcghwd->bcfhwd", basis, x.to(basis.dtype))
        else:
            raise ValueError("Invalid shape: ", x.shape)

