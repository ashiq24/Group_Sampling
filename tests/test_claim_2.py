from gsampling.layers.anti_aliasing import AntiAliasingLayer
from gsampling.layers.sampling import SamplingLayer
import torch
from escnn.group import *
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from escnn import gspaces
import torch.nn.functional as F
import networkx as nx
from escnn import nn
import numpy as np
import unittest
from gsampling.utils.graph_constructors import GraphConstructor
import matplotlib.pyplot as plt


def test_bandlimited_claim(
    group_type: str,
    order: int,
    sub_group_type: str,
    subsampling_factor: int,
    generator: str = "r-s",
    smooth_operator: str = "graph_shift",
    mode: str = "linear_optim",
    iterations: int = 100000,
    smoothness_loss_weight: float = 1.0,
    threshold: float = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    sample_type: str = "pool",
    equi_correction: bool = False,
):
    print(
        f"Testing group type: {group_type}, order: {order}, subgroup type: {sub_group_type}, subsampling factor: {subsampling_factor}"
    )

    if group_type == "dihedral":
        nodes_num = order * 2
    elif group_type == "cycle":
        nodes_num = order

    gc = GraphConstructor(
        group_size=nodes_num,
        group_type=group_type,
        group_generator=generator,
        subgroup_type=sub_group_type,
        subsampling_factor=subsampling_factor,
    )

    p = AntiAliasingLayer(
        nodes=gc.graph.nodes,
        adjaceny_matrix=gc.graph.directed_adjacency_matrix,
        basis=gc.graph.fourier_basis,
        subsample_nodes=gc.subgroup_graph.nodes,
        subsample_adjacency_matrix=gc.subgroup_graph.adjacency_matrix,
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

    sampling_layer = SamplingLayer(
        sampling_factor=subsampling_factor,
        nodes=gc.graph.nodes,
        subsample_nodes=gc.subgroup_graph.nodes,
        type=sample_type,
    ).to(device, dtype=dtype)

    if group_type == "dihedral":
        G = dihedral_group(order)
    elif group_type == "cycle":
        G = cyclic_group(order)

    print("Checking recontruction !!")
    error = []
    for i in range(100):
        f_bandlimited = torch.randn(nodes_num, dtype=dtype, device=device)
        f_bandlimited = p(f_bandlimited)

        f_band_sub = sampling_layer(f_bandlimited)
        f_sub_up = p.up_sample(f_band_sub)
        error.append((torch.norm(f_bandlimited - f_sub_up, p=2).item() ** 2))

    print(f"Error in reconstruction is", np.mean(error))

    return np.mean(error), np.std(error)


class TestReconstructionError(unittest.TestCase):
    def test_reconstruction(self):
        order = 8
        ing = "dihedral"
        outg = "dihedral"
        sampling_factor = 2
        error, std = test_bandlimited_claim(
            ing,
            order,
            outg,
            sampling_factor,
            generator="r-s",
            mode="linear_optim",
            smooth_operator="adjacency",
            smoothness_loss_weight=5.0,
            iterations=10000,
            threshold=0.0,
            device="cpu",
            dtype=torch.double,
            sample_type="sample",
        )
        assert error < 10e-4

        order = 12
        ing = "dihedral"
        outg = "dihedral"
        sampling_factor = 2
        error, std = test_bandlimited_claim(
            ing,
            order,
            outg,
            sampling_factor,
            generator="r-s",
            mode="linear_optim",
            smooth_operator="adjacency",
            smoothness_loss_weight=5.0,
            iterations=10000,
            threshold=0.0,
            device="cpu",
            dtype=torch.double,
            sample_type="sample",
        )
        assert error < 10e-4

        order = 8
        ing = "dihedral"
        outg = "cycle"
        sampling_factor = 2
        error, std = test_bandlimited_claim(
            ing,
            order,
            outg,
            sampling_factor,
            generator="r-s",
            mode="linear_optim",
            smooth_operator="adjacency",
            smoothness_loss_weight=5.0,
            iterations=10000,
            threshold=0.0,
            device="cpu",
            dtype=torch.double,
            sample_type="sample",
        )
        assert error < 10e-4

        order = 8
        ing = "cycle"
        outg = "cycle"
        sampling_factor = 2
        error, std = test_bandlimited_claim(
            ing,
            order,
            outg,
            sampling_factor,
            generator="r-s",
            mode="linear_optim",
            smooth_operator="adjacency",
            smoothness_loss_weight=5.0,
            iterations=10000,
            threshold=0.0,
            device="cpu",
            dtype=torch.double,
            sample_type="sample",
        )
        assert error < 10e-4

        order = 9
        ing = "cycle"
        outg = "cycle"
        sampling_factor = 3
        error, std = test_bandlimited_claim(
            ing,
            order,
            outg,
            sampling_factor,
            generator="r-s",
            mode="linear_optim",
            smooth_operator="adjacency",
            smoothness_loss_weight=5.0,
            iterations=10000,
            threshold=0.0,
            device="cpu",
            dtype=torch.double,
            sample_type="sample",
        )
        assert error < 10e-4
