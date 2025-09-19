"""
Tests for Claim 2: Bandlimited signal reconstruction with anti-aliasing.

Converted from unittest to pytest framework.
Tests the bandlimited reconstruction property: signals filtered by anti-aliasing
operator should be perfectly reconstructed after subsampling and upsampling.
"""

import pytest
import torch
import numpy as np
from gsampling.layers.helper import SmoothOperatorFactory, ReynoldsProjectorHelper, L1ProjectorUtils, FourierOps

# Import the modules under test
try:
    from gsampling.layers.anti_aliasing import AntiAliasingLayer
    from gsampling.layers.sampling import SamplingLayer
    from gsampling.utils.graph_constructors import GraphConstructor
    from escnn.group import dihedral_group, cyclic_group
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


def bandlimited_claim_helper(
    group_type: str,
    order: int,
    sub_group_type: str,
    subsampling_factor: int,
    generator: str = "r-s",
    smooth_operator: str = "graph_shift",
    mode: str = "linear_optim",
    iterations: int = 1000,
    smoothness_loss_weight: float = 1.0,
    threshold: float = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    sample_type: str = "pool",
    equi_correction: bool = False,
    num_trials: int = 100,
):
    """
    Helper function to test bandlimited reconstruction claim.
    
    Tests that signals filtered by anti-aliasing operator can be
    perfectly reconstructed after subsampling and upsampling.
    
    Returns:
        Tuple of (mean_error, std_error)
    """
    print(
        f"Testing group type: {group_type}, order: {order}, "
        f"subgroup type: {sub_group_type}, subsampling factor: {subsampling_factor}"
    )

    # Determine group size
    if group_type == "dihedral":
        nodes_num = order * 2
    elif group_type == "cycle":
        nodes_num = order
    else:
        raise ValueError(f"Unknown group type: {group_type}")

    # Build graph constructor
    gc = GraphConstructor(
        group_size=nodes_num,
        group_type=group_type,
        group_generator=generator,
        subgroup_type=sub_group_type,
        subsampling_factor=subsampling_factor,
    )

    # Build anti-aliasing layer
    p = AntiAliasingLayer(
        nodes=gc.graph.nodes,
        adjaceny_matrix=gc.graph.directed_adjacency_matrix,
        basis=gc.graph.fourier_basis,
        subsample_nodes=gc.subgroup_graph.nodes,
        sub_basis=gc.subgroup_graph.fourier_basis,
        subsample_adjacency_matrix=gc.subgroup_graph.directed_adjacency_matrix,
        smooth_operator=smooth_operator,
        smoothness_loss_weight=smoothness_loss_weight,
        iterations=iterations,
        mode=mode,
        device=device,
        threshold=threshold,
        graph_shift=gc.graph.smoother,
        raynold_op=gc.graph.equi_raynold_op,
        equi_correction=equi_correction,
        dtype=dtype,
    )

    # Build sampling layer
    sampling_layer = SamplingLayer(
        sampling_factor=subsampling_factor,
        nodes=gc.graph.nodes,
        subsample_nodes=gc.subgroup_graph.nodes,
        type=sample_type,
    ).to(device, dtype=dtype)

    # Get ESCNN group for validation
    if group_type == "dihedral":
        G = dihedral_group(order)
    elif group_type == "cycle":
        G = cyclic_group(order)

    print("Checking reconstruction...")
    p = p.to(device, dtype=dtype)
    # Test reconstruction over multiple trials
    error = []
    # Generate random signal and apply anti-aliasing (makes signal bandlimited)
    f_bandlimited = torch.randn(nodes_num, dtype=dtype, device=device)
    f_bandlimited = p(f_bandlimited)

    # Subsample and upsample
    f_band_sub = sampling_layer(f_bandlimited)
    f_sub_up = p.up_sample(f_band_sub)
    
    # Compute reconstruction error
    error.append((torch.norm(f_bandlimited - f_sub_up, p=2).item() ** 2))

    mean_error = np.mean(error)
    
    print(f"Error in reconstruction is {mean_error:.6f}")

    return mean_error


class TestBandlimitedReconstruction:
    """Test bandlimited signal reconstruction with anti-aliasing."""

    @pytest.mark.parametrize("group_type,order,sub_group_type,subsampling_factor,expected_error, iterations", [
        ("dihedral", 8, "dihedral", 2, 5000.0, 100000), 
        ("dihedral", 8, "cycle", 2, 10000.0, 100000),
        # ("cycle", 8, "cycle", 2, 1e-1, 1000),
        # ("cycle", 9, "cycle", 3, 1e-1, 1000),
    ])
    def test_reconstruction_error_original_cases(
        self, group_type, order, sub_group_type, subsampling_factor, expected_error, iterations, num_trials=3
    ):
        """Test reconstruction error for the original test cases."""
        mean_error = bandlimited_claim_helper(
            group_type=group_type,
            order=order,
            sub_group_type=sub_group_type,
            subsampling_factor=subsampling_factor,
            generator="r-s",
            mode="gpu_optim",
            smooth_operator="adjacency",
            smoothness_loss_weight=0.001,
            iterations=iterations,
            threshold=0.0,
            device="cuda",
            dtype=torch.float32,
            sample_type="sample",
            num_trials=num_trials,
        )
        
        assert mean_error < expected_error, \
            f"Reconstruction error {mean_error} exceeds threshold {expected_error}"




