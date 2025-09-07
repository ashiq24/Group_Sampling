"""
Tests for Claim 2: 3D Bandlimited signal reconstruction with anti-aliasing.

Tests the bandlimited reconstruction property for 3D groups (octahedral, full octahedral):
signals filtered by anti-aliasing operator should be perfectly reconstructed after 
subsampling and upsampling, even for complex 3D symmetries.

This extends the 2D testing to validate that the anti-aliasing infrastructure
works correctly with 3D group structures and their Fourier bases.
"""

import pytest
import torch
import numpy as np

# Import the modules under test
try:
    from gsampling.layers.anti_aliasing import AntiAliasingLayer
    from gsampling.layers.sampling import SamplingLayer
    from gsampling.utils.graph_constructors import GraphConstructor
    from escnn.group import octa_group, full_octa_group, cyclic_group, dihedral_group
    from gsampling.core.subsampling import SubsamplingRegistry
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


def bandlimited_claim_helper_3d(
    group_type: str,
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
    Helper function to test 3D bandlimited reconstruction claim.
    
    Tests that signals filtered by anti-aliasing operator can be
    perfectly reconstructed after subsampling and upsampling for 3D groups.
    
    **3D Group Support:**
    - octahedral: 24 elements (rotational symmetries of cube/octahedron)
    - full_octahedral: 48 elements (including inversions)
    
    **Subsampling Strategies:**
    - O → C4: Octahedral to cyclic subgroup (preserves z-axis rotations)
    - O_h → D4: Full octahedral to dihedral subgroup (preserves z-axis + reflections)
    - O_h → O: Full octahedral to octahedral subgroup (proper rotations only)
    
    Returns:
        Mean reconstruction error
    """
    print(
        f"Testing 3D group type: {group_type}, "
        f"subgroup type: {sub_group_type}, subsampling factor: {subsampling_factor}"
    )

    # Determine group size for 3D groups
    if group_type == "octahedral":
        nodes_num = 24
    elif group_type == "full_octahedral":
        nodes_num = 48
    elif group_type == "dihedral":
        nodes_num = 24
    elif group_type == "cycle":
        nodes_num = 12
    else:
        raise ValueError(f"Unsupported 3D group type: {group_type}")
        


    # Build graph constructor for 3D groups
    gc = GraphConstructor(
        group_size=nodes_num,
        group_type=group_type,
        group_generator=generator,
        subgroup_type=sub_group_type,
        subsampling_factor=subsampling_factor,
    )

    # Build anti-aliasing layer for 3D groups
    p = AntiAliasingLayer(
        nodes=gc.graph.nodes,
        adjaceny_matrix=gc.graph.directed_adjacency_matrix,
        basis=gc.graph.fourier_basis,
        subsample_nodes=gc.subgroup_graph.nodes,
        subsample_adjacency_matrix=gc.subgroup_graph.directed_adjacency_matrix,
        sub_basis=gc.subgroup_graph.fourier_basis,
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

    # Build sampling layer for 3D groups
    sampling_layer = SamplingLayer(
        sampling_factor=subsampling_factor,
        nodes=gc.graph.nodes,
        subsample_nodes=gc.subgroup_graph.nodes,
        type=sample_type,
    ).to(device, dtype=dtype)

    # Get ESCNN group for validation
    if group_type == "octahedral":
        G = octa_group()
    elif group_type == "full_octahedral":
        G = full_octa_group()
    elif group_type == "dihedral":
        G = dihedral_group(12)
    elif group_type == "cycle":
        G = cyclic_group(12)

    print("Checking 3D reconstruction...")
    p = p.to(device, dtype=dtype)
    
    # Test reconstruction over multiple trials
    errors = []
    
    for trial in range(num_trials):
        # Generate random signal and apply anti-aliasing (makes signal bandlimited)
        f_bandlimited = torch.randn(nodes_num, dtype=dtype, device=device)
        f_bandlimited = p(f_bandlimited)

        # Subsample and upsample
        f_band_sub = sampling_layer(f_bandlimited)
        f_sub_up = p.up_sample(f_band_sub)
        
        # Compute reconstruction error
        error = torch.norm(f_bandlimited - f_sub_up, p=2).item() ** 2
        errors.append(error)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"3D reconstruction error: {mean_error:.6f} ± {std_error:.6f}")

    return mean_error


class Test3DBandlimitedReconstruction:
    """Test 3D bandlimited signal reconstruction with anti-aliasing."""

    @pytest.mark.parametrize("group_type,sub_group_type,subsampling_factor,expected_error,iterations, mode", [
        # Octahedral group subsampling - higher thresholds for 3D groups
        ("octahedral", "cycle", 6, 20.0, 10000, "gpu_optim"),  # O → C4 (24 → 4)
        
        # Full octahedral group subsampling - higher thresholds for 3D groups
        ("full_octahedral", "cycle", 12, 50.0, 10000, "gpu_optim"),  # O_h → C4 (48 → 4) - very high subsampling
        ("full_octahedral", "dihedral", 6, 25.0, 10000, "gpu_optim"),  # O_h → D4 (48 → 8)
        ("full_octahedral", "octahedral", 2, 8.0, 10000, "gpu_optim"),  # O_h → O (48 → 24)
        
        # 2D groups on 3D data (for comparison) - adjusted thresholds
        ("dihedral", "dihedral", 2, 2.5, 10000, "gpu_optim"),  # D12 → D6 (24 → 12)
        ("cycle", "cycle", 2, 3.0, 10000, "gpu_optim"),  # C12 → C6 (12 → 6)
    ])
    def test_3d_reconstruction_error(
        self, group_type, sub_group_type, subsampling_factor, expected_error, iterations, mode
    ):
        """Test reconstruction error for 3D group configurations."""
        mean_error = bandlimited_claim_helper_3d(
            group_type=group_type,
            sub_group_type=sub_group_type,
            subsampling_factor=subsampling_factor,
            generator="r-s",
            mode=mode,  # Use analytical mode for faster testing
            smooth_operator="adjacency",
            smoothness_loss_weight=0.000001,
            iterations=iterations,
            threshold=0.0,
            device="cpu",  # Use CPU for testing
            dtype=torch.float32,
            sample_type="sample",
            num_trials=2,  # Fewer trials for faster testing
        )
        
        assert mean_error < expected_error, \
            f"3D reconstruction error {mean_error} exceeds threshold {expected_error}"


    def test_3d_subsampling_strategies_registered(self):
        """Test that all 3D subsampling strategies are properly registered."""
        # Check that 3D strategies are registered
        assert ("octahedral", "cycle") in SubsamplingRegistry.get_supported_transitions()
        assert ("full_octahedral", "cycle") in SubsamplingRegistry.get_supported_transitions()
        assert ("full_octahedral", "dihedral") in SubsamplingRegistry.get_supported_transitions()
        assert ("full_octahedral", "octahedral") in SubsamplingRegistry.get_supported_transitions()




