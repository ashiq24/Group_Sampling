from models.model_handler import get_model
import torch
import pytest


def test_model_creation():
    """
    Test the creation of the Gcnn model with default parameters.
    """
    num_channels = [32, 64]
    model = get_model(input_channel=3, num_channels=num_channels)
    # random input
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 10)

    final_feature = model.get_feature(x)
    assert final_feature.shape[-1] == num_channels[-1]

    dwn_group_types = [["dihedral", "cycle"], ["cycle", "cycle"]]
    model = get_model(
        input_channel=3, num_channels=num_channels, dwn_group_types=dwn_group_types, subsampling_factors=[2, 1]
    )
    # random input
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 10)

    final_feature = model.get_feature(x)
    assert final_feature.shape[-1] == num_channels[-1]
